"""Task-matched online KNODE residual learning for the shared 3-D MPC.

The controller, reference horizon, plant state and motor-thrust interface are
owned by ``realtime_quadrotor3d_experiment``.  This module contains only the
learning mechanism that distinguishes the KNODE baseline: fresh collection
windows, a queue of at most three Tanh residual blocks, exponential forgetting
and training of only the newest block.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


INPUT_DIM = 16
OUTPUT_DIM = 6
MAX_MODELS = 3
MAX_HIDDEN_DIM = 20
NEW_MODEL_HIDDEN_DIM = 16


def exponential_queue_weights(size: int) -> np.ndarray:
    if not 1 <= size <= MAX_MODELS:
        raise ValueError(f"queue size must lie in [1, {MAX_MODELS}]")
    return np.exp(np.arange(size, dtype=np.float32) - (size - 1))


class KNODEQueueBlock(nn.Module):
    """Fixed-shape block supporting the authors' 20/16 hidden-unit queue."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(INPUT_DIM, MAX_HIDDEN_DIM)
        self.layer2 = nn.Linear(MAX_HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, value: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.layer1(value)) * hidden_mask
        return self.layer2(hidden)


class TaskMatchedKNODEResidual(nn.Module):
    """Three-slot KNODE queue with a fixed graph for RealTimeL4CasADi."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(KNODEQueueBlock() for _ in range(MAX_MODELS))
        self.register_buffer("queue_weights", torch.zeros(MAX_MODELS))
        self.register_buffer(
            "hidden_masks", torch.zeros(MAX_MODELS, MAX_HIDDEN_DIM)
        )
        self.register_buffer("active_count", torch.zeros((), dtype=torch.int64))
        for block in self.blocks:
            self._initialize_block(block, NEW_MODEL_HIDDEN_DIM)

    @staticmethod
    def _initialize_block(block: KNODEQueueBlock, hidden_dim: int) -> None:
        if hidden_dim not in (NEW_MODEL_HIDDEN_DIM, MAX_HIDDEN_DIM):
            raise ValueError("KNODE hidden dimension must be 16 or 20")
        with torch.no_grad():
            block.layer1.weight.zero_()
            block.layer1.bias.zero_()
            block.layer2.weight.zero_()
            block.layer2.bias.zero_()
            nn.init.xavier_uniform_(block.layer1.weight[:hidden_dim])
            nn.init.xavier_uniform_(block.layer2.weight[:, :hidden_dim])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = value.new_zeros((*value.shape[:-1], OUTPUT_DIM))
        for index, block in enumerate(self.blocks):
            result = result + self.queue_weights[index] * block(
                value, self.hidden_masks[index]
            )
        return result

    def activate_newest_block(self) -> int:
        """Append a new block, dropping the oldest when the queue is full."""
        count = int(self.active_count.item())
        if count == MAX_MODELS:
            for target, source in ((0, 1), (1, 2)):
                self.blocks[target].load_state_dict(self.blocks[source].state_dict())
                self.hidden_masks[target].copy_(self.hidden_masks[source])
            newest_index = MAX_MODELS - 1
        else:
            newest_index = count
            count += 1

        hidden_dim = MAX_HIDDEN_DIM if count == 1 else NEW_MODEL_HIDDEN_DIM
        self._initialize_block(self.blocks[newest_index], hidden_dim)
        self.hidden_masks[newest_index].zero_()
        self.hidden_masks[newest_index, :hidden_dim] = 1.0
        self.active_count.fill_(count)
        self.queue_weights.zero_()
        weights = torch.as_tensor(exponential_queue_weights(count))
        self.queue_weights[:count].copy_(weights)
        return newest_index


@dataclass(frozen=True)
class RealtimeKNODEJob:
    version: int
    trigger_step: int
    trigger_wall_time: float
    inputs: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class RealtimeKNODEResult:
    version: int
    trigger_step: int
    trigger_wall_time: float
    ready_wall_time: float
    train_seconds: float
    losses: tuple[float, ...]
    state_dict: dict[str, np.ndarray] | None
    error: str | None = None


def _numpy_state_dict(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }


def _load_numpy_state_dict(model: nn.Module, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict(
        {name: torch.as_tensor(value).clone() for name, value in state.items()}
    )


def _training_process(
    jobs,
    results,
    ready_event,
    initial_state: dict[str, np.ndarray],
    epochs: int,
    learning_rate: float,
    regularization: float,
    torch_threads: int,
) -> None:
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = TaskMatchedKNODEResidual()
    _load_numpy_state_dict(model, initial_state)
    ready_event.set()

    while True:
        job = jobs.get()
        if job is None:
            return
        started = time.perf_counter()
        losses: list[float] = []
        state = None
        error = None
        try:
            newest_index = model.activate_newest_block()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            newest_parameters = list(model.blocks[newest_index].parameters())
            for parameter in newest_parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.Adam(newest_parameters, lr=learning_rate)
            inputs = torch.as_tensor(job.inputs, dtype=torch.float32)
            targets = torch.as_tensor(job.targets, dtype=torch.float32)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                data_loss = torch.mean((prediction - targets).square())
                l2_norm = sum(
                    parameter.square().sum()
                    for index in range(int(model.active_count.item()))
                    for parameter in model.blocks[index].parameters()
                )
                objective = data_loss + regularization * l2_norm
                objective.backward()
                optimizer.step()
                losses.append(float(objective.detach()))
            model.eval()
            state = _numpy_state_dict(model)
        except Exception:
            error = traceback.format_exc()
        finished = time.perf_counter()
        results.put(
            RealtimeKNODEResult(
                version=job.version,
                trigger_step=job.trigger_step,
                trigger_wall_time=job.trigger_wall_time,
                ready_wall_time=finished,
                train_seconds=finished - started,
                losses=tuple(losses),
                state_dict=state,
                error=error,
            )
        )


class AsyncRealtimeKNODETrainer:
    """Persistent worker for sequential, fresh-window KNODE updates."""

    def __init__(
        self,
        initial_model: TaskMatchedKNODEResidual,
        *,
        epochs: int = 60,
        learning_rate: float = 1e-2,
        regularization: float = 1e-7,
        torch_threads: int = 1,
    ) -> None:
        if epochs <= 0 or torch_threads <= 0:
            raise ValueError("epochs and torch_threads must be positive")
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
                _numpy_state_dict(initial_model),
                epochs,
                learning_rate,
                regularization,
                torch_threads,
            ),
            name="realtime-knode-training",
            daemon=False,
        )
        self._process.start()
        if not self._ready_event.wait(timeout=30.0):
            self._process.join(timeout=0.1)
            raise RuntimeError("KNODE training process failed to initialize")

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, job: RealtimeKNODEJob) -> bool:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed trainer")
        if self._busy:
            return False
        self._jobs.put_nowait(job)
        self._busy = True
        return True

    def poll(self) -> RealtimeKNODEResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        return result

    def finish(self, timeout: float = 30.0) -> list[RealtimeKNODEResult]:
        if self._closed:
            return []
        pending = []
        if self._busy:
            pending.append(self._results.get(timeout=timeout))
            self._busy = False
        self._jobs.put(None)
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            raise TimeoutError("KNODE training process did not stop cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"KNODE training process exited with code {self._process.exitcode}"
            )
        self._jobs.close()
        self._results.close()
        self._jobs.join_thread()
        self._results.join_thread()
        self._closed = True
        return pending
