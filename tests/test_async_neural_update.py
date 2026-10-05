import time

import numpy as np
import torch

from src.async_neural_update import AsyncNeuralUpdater, NeuralUpdateJob


class _NominalResult:
    def full(self):
        return np.zeros((6, 1), dtype=float)


def _nominal(_state, _control):
    return _NominalResult()


class _Publisher:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.models = []

    def update(self, model):
        time.sleep(self.delay)
        self.models.append(model)


def test_single_worker_rejects_overlap_and_publishes_inactive_slot():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(8, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    publishers = [_Publisher(), _Publisher(delay=0.02)]
    updater = AsyncNeuralUpdater(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        nominal_func=_nominal,
        publishers=publishers,
        epochs=2,
    )
    samples = np.zeros((32, 11), dtype=np.float32)
    job = NeuralUpdateJob(
        version=1,
        trigger_step=50,
        trigger_wall_time=time.perf_counter(),
        staging_index=1,
        samples=samples,
    )

    assert updater.submit(job)
    assert not updater.submit(job)
    pending = updater.finish()

    assert len(pending) == 1
    result = pending[0]
    assert result.error is None
    assert result.version == 1
    assert result.staging_index == 1
    assert result.publish_seconds >= 0.015
    assert len(publishers[0].models) == 0
    assert len(publishers[1].models) == 1
    assert publishers[1].models[0] is not model
