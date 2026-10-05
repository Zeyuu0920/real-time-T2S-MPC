"""Asynchronous two-timescale training for RealTime T2S MPC."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass

import numpy as np
import torch

from src.models import BoundedTwoSpeedMLP, TwoSpeedMLP
from src.realtime_neural import load_numpy_state_dict


def sample_replay_without_replacement(
    rng: np.random.Generator,
    inputs,
    targets,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a replay mini-batch using distinct buffer indices."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(inputs) != len(targets):
        raise ValueError("replay inputs and targets must have equal length")
    if batch_size > len(inputs):
        raise ValueError(
            f"cannot draw {batch_size} distinct samples from {len(inputs)} entries"
        )
    indices = rng.choice(len(inputs), size=batch_size, replace=False)
    return (
        np.asarray([inputs[index] for index in indices], dtype=np.float32),
        np.asarray([targets[index] for index in indices], dtype=np.float32),
    )


@dataclass(frozen=True)
class RealtimeT2SJob:
    """One atomic fast/slow update request.

    Either batch may be absent.  When both are present, the worker preserves
    the original T2S ordering: update the fast output layer first, then the
    two slow representation layers.
    """

    version: int
    trigger_step: int
    trigger_wall_time: float
    fast_inputs: np.ndarray | None = None
    fast_targets: np.ndarray | None = None
    slow_inputs: np.ndarray | None = None
    slow_targets: np.ndarray | None = None


@dataclass(frozen=True)
class RealtimeT2SResult:
    version: int
    trigger_step: int
    trigger_wall_time: float
    ready_wall_time: float
    train_seconds: float
    fast_seconds: float
    slow_seconds: float
    fast_losses: tuple[float, ...]
    slow_losses: tuple[float, ...]
    state_dict: dict[str, np.ndarray] | None
    error: str | None = None


def _set_trainable(model: TwoSpeedMLP, *, fast: bool) -> None:
    for parameter in model.layer1.parameters():
        parameter.requires_grad = not fast
    for parameter in model.layer2.parameters():
        parameter.requires_grad = not fast
    for parameter in model.layer3.parameters():
        parameter.requires_grad = fast


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Restore Adam's state to its exact pre-training values after warmup."""
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()


def _training_process(
    jobs,
    results,
    ready_event,
    initial_state: dict[str, np.ndarray],
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    fast_learning_rate: float,
    slow_learning_rate: float,
    fast_epochs: int,
    slow_epochs: int,
    torch_threads: int,
    bounded_output: bool,
    cpu_core: int | None,
) -> None:
    if cpu_core is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity requires os.sched_setaffinity")
        os.sched_setaffinity(0, {int(cpu_core)})
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    model_class = BoundedTwoSpeedMLP if bounded_output else TwoSpeedMLP
    model = model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    load_numpy_state_dict(model, initial_state)
    fast_optimizer = torch.optim.Adam(
        model.layer3.parameters(), lr=fast_learning_rate
    )
    slow_optimizer = torch.optim.Adam(
        list(model.layer1.parameters()) + list(model.layer2.parameters()),
        lr=slow_learning_rate,
    )
    criterion = torch.nn.MSELoss()

    # Warm PyTorch autograd and both lazy Adam paths before the controller is
    # armed.  Restore both weights and optimizer state afterward, so this has
    # no learning effect but removes one-time initialization from job 1.
    warm_inputs = torch.zeros((4, input_dim), dtype=torch.float32)
    warm_targets = torch.zeros((4, output_dim), dtype=torch.float32)
    _set_trainable(model, fast=True)
    fast_optimizer.zero_grad(set_to_none=True)
    criterion(model(warm_inputs), warm_targets).backward()
    fast_optimizer.step()
    _set_trainable(model, fast=False)
    slow_optimizer.zero_grad(set_to_none=True)
    criterion(model(warm_inputs), warm_targets).backward()
    slow_optimizer.step()
    load_numpy_state_dict(model, initial_state)
    _reset_optimizer_state(fast_optimizer)
    _reset_optimizer_state(slow_optimizer)
    model.zero_grad(set_to_none=True)
    ready_event.set()

    while True:
        job = jobs.get()
        if job is None:
            return
        start = time.perf_counter()
        fast_seconds = 0.0
        slow_seconds = 0.0
        fast_losses: list[float] = []
        slow_losses: list[float] = []
        state = None
        error = None
        try:
            model.train()
            if job.fast_inputs is not None:
                fast_start = time.perf_counter()
                inputs = torch.as_tensor(job.fast_inputs, dtype=torch.float32)
                targets = torch.as_tensor(job.fast_targets, dtype=torch.float32)
                _set_trainable(model, fast=True)
                for _ in range(fast_epochs):
                    fast_optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    fast_optimizer.step()
                    fast_losses.append(float(loss.detach()))
                fast_seconds = time.perf_counter() - fast_start

            if job.slow_inputs is not None:
                slow_start = time.perf_counter()
                inputs = torch.as_tensor(job.slow_inputs, dtype=torch.float32)
                targets = torch.as_tensor(job.slow_targets, dtype=torch.float32)
                _set_trainable(model, fast=False)
                for _ in range(slow_epochs):
                    slow_optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    slow_optimizer.step()
                    slow_losses.append(float(loss.detach()))
                slow_seconds = time.perf_counter() - slow_start

            model.eval()
            state = {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in model.state_dict().items()
            }
        except Exception:
            error = traceback.format_exc()
        ready = time.perf_counter()
        results.put(
            RealtimeT2SResult(
                version=job.version,
                trigger_step=job.trigger_step,
                trigger_wall_time=job.trigger_wall_time,
                ready_wall_time=ready,
                train_seconds=ready - start,
                fast_seconds=fast_seconds,
                slow_seconds=slow_seconds,
                fast_losses=tuple(fast_losses),
                slow_losses=tuple(slow_losses),
                state_dict=state,
                error=error,
            )
        )


class AsyncRealtimeT2STrainer:
    """Persistent T2S worker with one atomic update in flight."""

    def __init__(
        self,
        initial_model: TwoSpeedMLP,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        fast_learning_rate: float = 1e-3,
        slow_learning_rate: float = 1e-2,
        fast_epochs: int = 10,
        slow_epochs: int = 20,
        torch_threads: int = 1,
        bounded_output: bool = False,
        cpu_core: int | None = None,
    ) -> None:
        if fast_epochs <= 0 or slow_epochs <= 0 or torch_threads <= 0:
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
                hidden_dim,
                output_dim,
                fast_learning_rate,
                slow_learning_rate,
                fast_epochs,
                slow_epochs,
                torch_threads,
                bounded_output,
                cpu_core,
            ),
            name="realtime-t2s-training",
            daemon=False,
        )
        self._process.start()
        if not self._ready_event.wait(timeout=30.0):
            self._process.join(timeout=0.1)
            raise RuntimeError("T2S training process failed to initialize")

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, job: RealtimeT2SJob) -> bool:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed trainer")
        if self._busy:
            return False
        if job.fast_inputs is None and job.slow_inputs is None:
            raise ValueError("A T2S job must contain a fast or slow batch")
        self._jobs.put_nowait(job)
        self._busy = True
        return True

    def poll(self) -> RealtimeT2SResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        return result

    def finish(self, timeout: float = 30.0) -> list[RealtimeT2SResult]:
        if self._closed:
            return []
        pending = []
        if self._busy:
            pending.append(self._results.get(timeout=timeout))
            self._busy = False
        self._jobs.put(None)
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            raise TimeoutError("T2S training process did not stop cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"T2S training process exited with code {self._process.exitcode}"
            )
        self._jobs.close()
        self._results.close()
        self._jobs.join_thread()
        self._results.join_thread()
        self._closed = True
        return pending
