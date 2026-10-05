"""Go2 adapter for the unchanged Quadrotor recent/reservoir replay class.

No physical truth is consumed: only the existing MPC-boundary features,
residual labels and their source times. Scheduling remains the Go2 schedule,
including no new pre-slow reservation or end-of-episode suppression.
"""
import numpy as np
from src.hybrid_replay import RecentReservoirReplay
from src.go2_icra_protocol import due_updates


class Go2T2SHybridReplay:
    def __init__(self, config):
        self.config = config
        self.replay = RecentReservoirReplay(50, 50, seed=config.seed + 600000)

    def append(self, inputs, targets, timestamp):
        step = round(timestamp / self.config.dt)
        if not np.isclose(step * self.config.dt, timestamp, rtol=0, atol=1e-9):
            raise ValueError("A replay sample must retain its MPC-boundary source time")
        self.replay.append(inputs, targets, step, timestamp)

    def due(self, cycle):
        pool = self.replay
        fast, slow = due_updates(cycle, len(pool.recent) + len(pool.history), self.config)
        return (fast and len(pool.recent) >= self.config.fast_batch,
                slow and pool.can_sample(self.config.slow_batch // 2, self.config.slow_batch // 2))

    def batches(self, rng, fast_due, slow_due):
        pool = self.replay
        fast = pool.latest(self.config.fast_batch) if fast_due else []
        recent, history = pool.sample(rng, self.config.slow_batch // 2,
            self.config.slow_batch // 2) if slow_due else ([], [])
        fast_x, fast_y = pool.arrays(fast) if fast else (None, None)
        slow_x, slow_y = pool.arrays(recent + history) if recent else (None, None)
        audit = dict(recent_size=len(pool.recent), history_size=len(pool.history),
            history_seen=pool.history_seen,
            fast_source_steps=[entry.source_step for entry in fast],
            slow_recent_source_steps=[entry.source_step for entry in recent],
            slow_history_source_steps=[entry.source_step for entry in history],
            fast_source_times=[entry.source_time for entry in fast],
            slow_recent_source_times=[entry.source_time for entry in recent],
            slow_history_source_times=[entry.source_time for entry in history])
        return fast_x, fast_y, slow_x, slow_y, audit
