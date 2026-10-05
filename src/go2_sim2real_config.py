"""Standard-library profile validation for the independent Go2 stress test."""
from dataclasses import dataclass
import math

from src.sim2real_config import load_json_object

PROTOCOL = "go2_sim2real_intra_period_v1"


@dataclass(frozen=True)
class RealismOptions:
    observation_delay_steps: int = 1
    position_std: float = .01
    velocity_std: float = .02
    attitude_std: float = .01
    body_rate_std: float = .02
    noise_correlation_s: float = .10
    actuator_tau_s: float = .005
    actuator_gain_range: float = .03

    def validate(self):
        if type(self.observation_delay_steps) is not int or self.observation_delay_steps < 0:
            raise ValueError("observation.delay_steps must be a nonnegative integer (20 ms per step)")
        for name in ("position_std", "velocity_std", "attitude_std", "body_rate_std",
                     "noise_correlation_s", "actuator_tau_s", "actuator_gain_range"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.noise_correlation_s <= 0:
            raise ValueError("observation.noise_correlation_time_s must be positive")
        if self.actuator_gain_range >= 1:
            raise ValueError("actuator.gain_range must lie in [0, 1)")

    def effective_profile(self):
        return {"schema_version": 1, "system": "go2", "observation": {
            "delay_steps": self.observation_delay_steps,
            "position_noise_std_m": self.position_std,
            "velocity_noise_std_mps": self.velocity_std,
            "attitude_noise_std_rad": self.attitude_std,
            "body_rate_noise_std_radps": self.body_rate_std,
            "noise_correlation_time_s": self.noise_correlation_s,
        }, "actuator": {"time_constant_s": self.actuator_tau_s,
                         "gain_range": self.actuator_gain_range}}


OBSERVATION_FIELDS = {
    "delay_steps": "observation_delay_steps",
    "position_noise_std_m": "position_std",
    "velocity_noise_std_mps": "velocity_std",
    "attitude_noise_std_rad": "attitude_std",
    "body_rate_noise_std_radps": "body_rate_std",
    "noise_correlation_time_s": "noise_correlation_s",
}
ACTUATOR_FIELDS = {"time_constant_s": "actuator_tau_s", "gain_range": "actuator_gain_range"}


def load_profile(path, expected_sha256=None):
    resolved, sha256, data = load_json_object(path, expected_sha256=expected_sha256)
    unknown = set(data) - {"schema_version", "system", "observation", "actuator"}
    if unknown:
        raise ValueError(f"Unknown Go2 profile keys: {sorted(unknown)}")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Go2 sim2real profile requires schema_version 1")
    if data.get("system") != "go2":
        raise ValueError("Go2 sim2real profile requires system 'go2'")
    values = {}
    for section, mapping in (("observation", OBSERVATION_FIELDS), ("actuator", ACTUATOR_FIELDS)):
        entries = data.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"{section} must be a JSON object")
        unknown = set(entries) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
        values.update({mapping[key]: value for key, value in entries.items()})
    options = RealismOptions(**values)
    options.validate()
    return options, {"path": resolved, "sha256": sha256, "effective": options.effective_profile()}


def timing_metadata(options):
    return {"protocol": PROTOCOL, "observation_delay_steps": options.observation_delay_steps,
            "observation_delay_ms": 20 * options.observation_delay_steps,
            "command_release": "measured-latency intra-period release",
            "control_period_ms": 20, "publication_resolution_ms": 1,
            "wbc_period_ms": 2, "deadline_ms": 20,
            "state_prediction": "hold delayed observation; no added next-boundary shift",
            "wbc_proprioception": "ideal current joint/body feedback",
            "legacy_config_timing_role": "latency_replay_pipeline retains learner readiness gating; command release is intra-period",
            "claim": "synthetic stress test; not hardware calibrated or the paper timing protocol"}
