"""Frozen, explicit physical and scheduling configuration for the Go2 study.

This is a new protocol; old 200 Hz and liquid result directories remain intact.
The main controller is 50 Hz, the prediction grid is independently 30 ms.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math


@dataclass(frozen=True)
class Go2ICRAConfig:
    method: str = "nominal"
    scenario: str = "drain"
    seed: int = 1001
    duration: float = 60.0
    speed: float = 0.5
    height: float = 0.30
    mpc_hz: int = 50
    low_level_hz: int = 500
    physics_hz: int = 1000
    warmup: float = 0.02
    payload_mass: float = 4.0
    timing: str = "paced_pipeline"
    delay_predictor: str = "hold"
    fast_every: int = 2
    slow_every: int = 50
    fast_batch: int = 4
    slow_batch: int = 64
    fast_epochs: int = 5
    slow_epochs: int = 20
    fast_lr: float = 2e-4
    slow_lr: float = 2e-4
    replay_capacity: int = 100
    ssi_learning_rate: float = 0.003
    gp_inducing: int = 80
    gp_spatial_lengthscale: float = 1.5
    gp_temporal_lengthscale: float = 5.0
    gp_noise: float = 0.36
    wrench_scales: tuple = (40., 40., 40., 4., 4., 4.)

    def validate(self):
        if self.method not in ("nominal", "ssi", "stgp", "t2s"):
            raise ValueError("unknown method")
        if self.scenario not in ("clean", "drain", "friction", "combined"):
            raise ValueError("unknown scenario")
        if self.timing not in ("paced_pipeline", "latency_replay_pipeline", "ideal"):
            raise ValueError("unknown timing mode")
        if self.delay_predictor not in ("model", "kinematic", "hold"):
            raise ValueError("unknown delay predictor")
        if self.mpc_hz != 50 or self.low_level_hz != 500 or self.physics_hz != 1000:
            raise ValueError("This protocol fixes 50/500/1000 Hz; use a new protocol for rate studies")
        for name in ("duration", "speed", "height", "payload_mass", "fast_every",
                     "slow_every", "fast_batch", "slow_batch", "fast_epochs",
                     "slow_epochs", "fast_lr", "slow_lr", "replay_capacity",
                     "ssi_learning_rate", "gp_inducing", "gp_spatial_lengthscale",
                     "gp_temporal_lengthscale", "gp_noise"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if max(self.fast_batch, self.slow_batch) > self.replay_capacity:
            raise ValueError("batch exceeds buffer")
        if len(self.wrench_scales) != 6 or any(not math.isfinite(s) or s <= 0 for s in self.wrench_scales):
            raise ValueError("wrench_scales must be six positive normalization units, not bounds")
        if not math.isclose(self.duration * self.mpc_hz, round(self.duration * self.mpc_hz)):
            raise ValueError("duration must be an integer number of MPC intervals")

    @property
    def dt(self):
        return 1.0 / self.mpc_hz

    @property
    def variable_friction(self):
        return self.scenario in ("friction", "combined")

    @property
    def liquid(self):
        return self.scenario in ("drain", "combined")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Go2ICRAFullStateT2SConfig(Go2ICRAConfig):
    """Explicit T2S-only input ablation; no unused option added to Nominal.

    The physical input is the unscaled [state(12), dispatched GRF(12)].
    All inherited physical, solver, target and learning settings are unchanged.
    """

    method: str = "t2s"
    t2s_feature_mode: str = "full_state_control24"

    def validate(self):
        super().validate()
        if self.method != "t2s" or self.t2s_feature_mode != "full_state_control24":
            raise ValueError("Full-state input variant is defined only for T2S")


@dataclass(frozen=True)
class Go2ICRATimeReplayT2SConfig(Go2ICRAFullStateT2SConfig):
    """Explicit T2S-only development variant; no baseline/default changes.

    Only the time representation and replay policy are varied relative to the
    fast-3 / latest-8 experiment. Physical inputs remain exactly [x(12), u(12)].
    """

    fast_every: int = 3
    fast_batch: int = 8
    time_feat_dim: int = 32
    t2s_replay_mode: str = "fifo_reservoir"

    def validate(self):
        super().validate()
        if not isinstance(self.time_feat_dim, int) or self.time_feat_dim <= 0 or self.time_feat_dim % 2:
            raise ValueError("time_feat_dim must be a positive even integer")
        if self.t2s_replay_mode not in ("fifo", "fifo_reservoir"):
            raise ValueError("Unknown T2S replay policy")
        if self.t2s_replay_mode == "fifo_reservoir":
            if self.replay_capacity != 100:
                raise ValueError("This variant preserves 50 recent + 50 history entries")
            if self.fast_batch > 50 or self.slow_batch % 2 or self.slow_batch // 2 > 50:
                raise ValueError("Hybrid batches must fit both disjoint pools")


def due_updates(cycle, sample_count, config):
    """Schedules count MPC boundaries, never physics ticks or accepted samples."""
    return (
        cycle > 0 and cycle % config.fast_every == 0 and sample_count >= config.fast_batch,
        cycle > 0 and cycle % config.slow_every == 0 and sample_count >= config.slow_batch,
    )


def releasable(source_cycle, current_cycle, ready_wall, deadline_wall, success):
    """One-period pipeline: no early application, no stale result at a later tick."""
    return bool(success and source_cycle + 1 == current_cycle
                and math.isfinite(ready_wall) and ready_wall <= deadline_wall)
