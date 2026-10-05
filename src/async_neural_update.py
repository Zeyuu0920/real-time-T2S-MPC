"""Single-worker asynchronous training for double-buffered Neural MPC."""

from __future__ import annotations

import copy
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class NeuralUpdateJob:
    """Immutable snapshot submitted by the control loop."""

    version: int
    trigger_step: int
    trigger_wall_time: float
    staging_index: int
    samples: np.ndarray


@dataclass(frozen=True)
class NeuralUpdateResult:
    """Completed candidate model and its wall-clock timing."""

    version: int
    trigger_step: int
    trigger_wall_time: float
    staging_index: int
    ready_wall_time: float
    train_seconds: float
    publish_seconds: float
    total_seconds: float
    losses: tuple[float, ...]
    error: str | None = None


class AsyncNeuralUpdater:
    """Train one canonical model and publish to one inactive model slot.

    Only this worker mutates the training model and optimizer.  The control
    loop chooses an inactive publisher when submitting a job and switches to
    that slot only after receiving a successful result.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        nominal_func: Any,
        publishers: Sequence[Any],
        epochs: int = 20,
    ) -> None:
        if len(publishers) != 2:
            raise ValueError("Double buffering requires exactly two publishers")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.nominal_func = nominal_func
        self.publishers = tuple(publishers)
        self.epochs = int(epochs)

        self._jobs: queue.Queue[NeuralUpdateJob | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[NeuralUpdateResult] = queue.Queue()
        self._state_lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="neural-mpc-update-worker",
            daemon=False,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy

    def submit(self, job: NeuralUpdateJob) -> bool:
        """Submit without blocking; return False when a job is already active."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed updater")
            if self._busy:
                return False
            self._busy = True
        self._jobs.put_nowait(job)
        return True

    def poll(self) -> NeuralUpdateResult | None:
        """Return one completed result without blocking."""
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._state_lock:
            self._busy = False
        return result

    def finish(self) -> list[NeuralUpdateResult]:
        """Wait for the active job, stop the worker, and return pending results."""
        with self._state_lock:
            if self._closed:
                return []
        self._jobs.join()
        pending = []
        while True:
            result = self.poll()
            if result is None:
                break
            pending.append(result)
        with self._state_lock:
            self._closed = True
        self._jobs.put(None)
        self._thread.join()
        return pending

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return

            start = time.perf_counter()
            train_seconds = 0.0
            publish_seconds = 0.0
            losses: list[float] = []
            error = None
            try:
                data = np.asarray(job.samples, dtype=np.float32)
                x_batch = torch.tensor(data[:, :8], dtype=torch.float32)
                y_true = data[:, 8:]
                nominal = np.array(
                    [
                        self.nominal_func(row[:6], row[6:8]).full().flatten()
                        for row in data
                    ]
                )
                y_nominal = nominal[:, [1, 3, 5]]
                y_target = torch.tensor(
                    y_true - y_nominal,
                    dtype=torch.float32,
                )

                self.model.train()
                for parameter in self.model.parameters():
                    parameter.requires_grad = True
                for _ in range(self.epochs):
                    self.optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(x_batch)
                    loss = self.criterion(prediction, y_target)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                train_seconds = time.perf_counter() - start

                candidate = copy.deepcopy(self.model).eval()
                for parameter in candidate.parameters():
                    parameter.requires_grad = False
                publish_start = time.perf_counter()
                self.publishers[job.staging_index].update(candidate)
                publish_seconds = time.perf_counter() - publish_start
            except Exception:
                error = traceback.format_exc()

            ready = time.perf_counter()
            self._results.put(
                NeuralUpdateResult(
                    version=job.version,
                    trigger_step=job.trigger_step,
                    trigger_wall_time=job.trigger_wall_time,
                    staging_index=job.staging_index,
                    ready_wall_time=ready,
                    train_seconds=train_seconds,
                    publish_seconds=publish_seconds,
                    total_seconds=ready - start,
                    losses=tuple(losses),
                    error=error,
                )
            )
            self._jobs.task_done()
