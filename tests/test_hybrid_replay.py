import numpy as np
import pytest

from src.hybrid_replay import RecentReservoirReplay


def append_number(buffer, index):
    buffer.append([index, index * 0.02], [-index], index, index * 0.02)


def test_bounded_pools_disjoint_and_timestamp_preserving():
    buffer = RecentReservoirReplay(seed=42)
    for index in range(1000):
        append_number(buffer, index)
        assert len(buffer.recent) <= 50 and len(buffer.history) <= 50
        recent = {entry.source_step for entry in buffer.recent}
        history = {entry.source_step for entry in buffer.history}
        assert recent.isdisjoint(history)
    assert [e.source_step for e in buffer.recent] == list(range(950, 1000))
    assert buffer.history_seen == 950
    assert min(e.source_step for e in buffer.history) < 100
    for entry in list(buffer.recent) + buffer.history:
        assert entry.source_time == entry.source_step * 0.02
        assert entry.inputs[0] == -entry.targets[0] == entry.source_step


def test_batches_are_distinct_balanced_and_fast_is_recent():
    buffer = RecentReservoirReplay(seed=42)
    for index in range(1000):
        append_number(buffer, index)
    recent, history = buffer.sample(np.random.default_rng(7), 32, 32)
    assert len({e.source_step for e in recent + history}) == 64
    assert all(e.source_step >= 950 for e in recent)
    assert all(e.source_step < 950 for e in history)
    assert [e.source_step for e in buffer.latest(4)] == [996, 997, 998, 999]
    inputs, targets = buffer.arrays(recent + history)
    np.testing.assert_array_equal(inputs[:, 0], -targets[:, 0])


def test_warmup_does_not_duplicate_samples_to_fill_batch():
    buffer = RecentReservoirReplay(seed=0)
    for index in range(81):
        append_number(buffer, index)
    assert not buffer.can_sample(32, 32)
    with pytest.raises(ValueError, match="distinct"):
        buffer.sample(np.random.default_rng(0), 32, 32)
    append_number(buffer, 81)
    assert buffer.can_sample(32, 32)
    with pytest.raises(ValueError, match="strictly increasing"):
        append_number(buffer, 81)


def test_reservoir_is_uniform_over_evicted_stream_not_recency_biased():
    # Across independent runs, first and last portions of the evicted stream
    # have the same inclusion probability. A sliding-window implementation
    # or a denominator based only on admitted samples fails this test.
    counts = np.zeros(100, dtype=int)
    for seed in range(1000):
        buffer = RecentReservoirReplay(10, 10, seed=seed)
        for index in range(110):
            append_number(buffer, index)
        for entry in buffer.history:
            counts[entry.source_step] += 1
    assert np.all(np.abs(counts.reshape(5, 20).sum(axis=1) - 2000) < 200)


def test_sampling_does_not_change_future_reservoir_admission():
    first = RecentReservoirReplay(5, 5, seed=12)
    second = RecentReservoirReplay(5, 5, seed=12)
    sample_rng = np.random.default_rng(0)
    for index in range(100):
        append_number(first, index)
        append_number(second, index)
        if first.can_sample(2, 2):
            first.sample(sample_rng, 2, 2)
    assert [e.source_step for e in first.history] == [e.source_step for e in second.history]
