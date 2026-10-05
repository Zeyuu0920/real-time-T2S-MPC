"""Utilities for parameterized first-order Neural MPC."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from src.models import BoundedTwoSpeedMLP, MLP, TwoSpeedMLP


@dataclass(frozen=True)
class RealtimeTrainingJob:
    version: int
    trigger_step: int
    trigger_wall_time: float
    inputs: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class RealtimeTrainingResult:
    version: int
    trigger_step: int
    trigger_wall_time: float
    ready_wall_time: float
    train_seconds: float
    losses: tuple[float, ...]
    state_dict: dict[str, np.ndarray] | None
    error: str | None = None


def first_order_parameters(
    model: torch.nn.Module,
    expansion_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return acados parameters, exact values, and Jacobians at each point.

    RealTimeL4CasADi's symbolic parameter vector is ``[a, f(a), vec_F(J)]``.
    CasADi vectorizes matrices column-major, so the Jacobian is explicitly
    packed in Fortran order rather than relying on NumPy's default ordering.
    """
    points = np.asarray(expansion_points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("expansion_points must have shape (batch, input_dim)")

    tensor = torch.as_tensor(points, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if isinstance(model, TwoSpeedMLP):
            # Exact batched chain rule for the fixed three-layer T2S network.
            # ``vmap(jacrev)`` constructs a fresh autograd graph every MPC
            # cycle and was slower than the paper controller's 5 ms period.
            # This produces the identical Jacobian without autograd overhead.
            first_pre = model.layer1(tensor)
            first = model.act(first_pre)
            second_pre = model.layer2(first)
            second = model.act(second_pre)
            output_pre = model.layer3(second)
            first_jacobian = (
                model.layer1.weight.unsqueeze(0)
                * (first_pre > 0).unsqueeze(-1)
            )
            second_jacobian = torch.matmul(
                model.layer2.weight.unsqueeze(0), first_jacobian
            ) * (second_pre > 0).unsqueeze(-1)
            jacobians = torch.matmul(
                model.layer3.weight.unsqueeze(0), second_jacobian
            )
            if isinstance(model, BoundedTwoSpeedMLP):
                values = torch.tanh(output_pre)
                jacobians = jacobians * (1.0 - values.square()).unsqueeze(-1)
            else:
                values = output_pre
        else:
            values = model(tensor)
            jacobians = torch.func.vmap(torch.func.jacrev(model))(tensor)

    values_np = values.detach().cpu().numpy().astype(np.float64, copy=False)
    jacobians_np = (
        jacobians.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    jacobian_fortran = jacobians_np.transpose(0, 2, 1).reshape(
        len(points), -1
    )
    parameters = np.concatenate(
        [points.astype(np.float64), values_np, jacobian_fortran],
        axis=1,
    )
    return parameters, values_np, jacobians_np


def evaluate_first_order(
    expansion_points: np.ndarray,
    values: np.ndarray,
    jacobians: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Evaluate the local first-order model at one query per expansion point."""
    expansion_points = np.asarray(expansion_points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    jacobians = np.asarray(jacobians, dtype=np.float64)
    delta = query_points - expansion_points
    return values + np.einsum("boi,bi->bo", jacobians, delta)


def exact_model_output(
    model: torch.nn.Module,
    points: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        result = model(torch.as_tensor(points, dtype=torch.float32))
    return result.detach().cpu().numpy().astype(np.float64, copy=False)


def relu_activation_patterns(
    model: torch.nn.Module,
    points: np.ndarray,
) -> np.ndarray:
    """Return ReLU patterns for the repository MLP or TwoSpeedMLP."""
    if hasattr(model, "layer1") and hasattr(model, "layer2"):
        hidden = torch.as_tensor(points, dtype=torch.float32)
        with torch.no_grad():
            first_pre_activation = model.layer1(hidden)
            hidden = model.act(first_pre_activation)
            second_pre_activation = model.layer2(hidden)
        return np.concatenate(
            [
                (first_pre_activation > 0).cpu().numpy(),
                (second_pre_activation > 0).cpu().numpy(),
            ],
            axis=1,
        )
    if not hasattr(model, "net"):
        raise TypeError(
            "Activation diagnostics require an MLP or TwoSpeedMLP"
        )
    hidden = torch.as_tensor(points, dtype=torch.float32)
    patterns = []
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                hidden = layer(hidden)
            elif isinstance(layer, torch.nn.ReLU):
                patterns.append((hidden > 0).cpu().numpy())
                hidden = layer(hidden)
            else:
                hidden = layer(hidden)
    if not patterns:
        return np.empty((len(points), 0), dtype=bool)
    return np.concatenate(patterns, axis=1)


def load_numpy_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, np.ndarray],
) -> None:
    tensors = {
        name: torch.as_tensor(value).clone()
        for name, value in state_dict.items()
    }
    model.load_state_dict(tensors)
    model.eval()


def _training_process(
    jobs,
    results,
    ready_event,
    initial_state: dict[str, np.ndarray],
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    learning_rate: float,
    epochs: int,
    torch_threads: int,
) -> None:
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    load_numpy_state_dict(model, initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    ready_event.set()

    while True:
        job = jobs.get()
        if job is None:
            return
        start = time.perf_counter()
        losses: list[float] = []
        state = None
        error = None
        try:
            inputs = torch.as_tensor(job.inputs, dtype=torch.float32)
            targets = torch.as_tensor(job.targets, dtype=torch.float32)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            state = {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in model.state_dict().items()
            }
        except Exception:
            error = traceback.format_exc()
        ready = time.perf_counter()
        results.put(
            RealtimeTrainingResult(
                version=job.version,
                trigger_step=job.trigger_step,
                trigger_wall_time=job.trigger_wall_time,
                ready_wall_time=ready,
                train_seconds=ready - start,
                losses=tuple(losses),
                state_dict=state,
                error=error,
            )
        )


class AsyncRealtimeTrainer:
    """One persistent training process with a single in-flight job."""

    def __init__(
        self,
        initial_model: torch.nn.Module,
        *,
        input_dim: int = 8,
        output_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        torch_threads: int = 1,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if epochs <= 0 or torch_threads <= 0:
            raise ValueError("epochs and torch_threads must be positive")
        initial_state = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in initial_model.state_dict().items()
        }
        context = mp.get_context("spawn")
        self._jobs = context.Queue(maxsize=1)
        self._results = context.Queue(maxsize=1)
        self._ready_event = context.Event()
        self._busy = False
        self._closed = False
        self._process = context.Process(
            target=_training_process,
            args=(
                self._jobs,
                self._results,
                self._ready_event,
                initial_state,
                input_dim,
                output_dim,
                hidden_dim,
                num_layers,
                learning_rate,
                epochs,
                torch_threads,
            ),
            name="realtime-neural-training",
            daemon=False,
        )
        self._process.start()
        if not self._ready_event.wait(timeout=30.0):
            self._process.join(timeout=0.1)
            raise RuntimeError("Training process failed to initialize")

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, job: RealtimeTrainingJob) -> bool:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed trainer")
        if self._busy:
            return False
        self._jobs.put_nowait(job)
        self._busy = True
        return True

    def poll(self) -> RealtimeTrainingResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        return result

    def finish(self, timeout: float = 30.0) -> list[RealtimeTrainingResult]:
        if self._closed:
            return []
        pending = []
        if self._busy:
            pending.append(self._results.get(timeout=timeout))
            self._busy = False
        self._jobs.put(None)
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            raise TimeoutError("Training process did not stop cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"Training process exited with code {self._process.exitcode}"
            )
        self._jobs.close()
        self._results.close()
        self._jobs.join_thread()
        self._results.join_thread()
        self._closed = True
        return pending
