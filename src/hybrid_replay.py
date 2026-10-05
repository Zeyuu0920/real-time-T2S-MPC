"""Bounded recent FIFO plus a uniform reservoir over evicted history."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayEntry:
    inputs: np.ndarray
    targets: np.ndarray
    source_step: int
    source_time: float


class RecentReservoirReplay:
    """Keep recent samples exactly and Algorithm-R samples of older history.

    Only FIFO evictions enter the reservoir stream. Thus the pools are disjoint
    and each of the n evicted samples has inclusion probability min(1, H/n),
    where H is the reservoir capacity. Admission has its own random generator,
    independent of training-batch selection and the simulated disturbances.
    """

    def __init__(self, recent_capacity=50, history_capacity=50, seed=0):
        if recent_capacity <= 0 or history_capacity <= 0:
            raise ValueError("Both replay capacities must be positive")
        self.recent_capacity = int(recent_capacity)
        self.history_capacity = int(history_capacity)
        self.recent = deque()
        self.history = []
        self.history_seen = 0
        self.last_source_step = -1
        self._rng = np.random.default_rng(seed)

    def append(self, inputs, targets, source_step, source_time):
        if source_step <= self.last_source_step:
            raise ValueError("Replay source steps must be strictly increasing")
        entry = ReplayEntry(
            np.asarray(inputs, dtype=np.float32).copy(),
            np.asarray(targets, dtype=np.float32).copy(),
            int(source_step), float(source_time),
        )
        self.last_source_step = int(source_step)
        if len(self.recent) == self.recent_capacity:
            evicted = self.recent.popleft()
            self.history_seen += 1
            if len(self.history) < self.history_capacity:
                self.history.append(evicted)
            else:
                slot = int(self._rng.integers(self.history_seen))
                if slot < self.history_capacity:
                    self.history[slot] = evicted
        self.recent.append(entry)

    def can_sample(self, recent_count, history_count):
        return (
            len(self.recent) >= recent_count
            and len(self.history) >= history_count
        )

    def latest(self, count):
        if count <= 0 or count > len(self.recent):
            raise ValueError("Not enough recent samples")
        return list(self.recent)[-count:]

    def sample(self, rng, recent_count, history_count):
        if recent_count <= 0 or history_count <= 0:
            raise ValueError("Both batch contributions must be positive")
        if not self.can_sample(recent_count, history_count):
            raise ValueError("Not enough distinct samples in both replay pools")
        recent = list(self.recent)
        recent_ids = rng.choice(len(recent), recent_count, replace=False)
        history_ids = rng.choice(len(self.history), history_count, replace=False)
        return (
            [recent[int(i)] for i in recent_ids],
            [self.history[int(i)] for i in history_ids],
        )

    @staticmethod
    def arrays(entries):
        return (
            np.asarray([entry.inputs for entry in entries], dtype=np.float32),
            np.asarray([entry.targets for entry in entries], dtype=np.float32),
        )
