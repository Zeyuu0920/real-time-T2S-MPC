"""Strict, standard-library-only loading of simulation-to-hardware profiles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


QUADROTOR_FIELDS = {
    "observation": {
        "delay_steps": "--measurement-delay-steps",
        "position_noise_std_m": "--position-noise-std",
        "velocity_noise_std_mps": "--velocity-noise-std",
        "attitude_noise_std_rad": "--attitude-noise-std",
        "body_rate_noise_std_radps": "--body-rate-noise-std",
        "noise_correlation_time_s": "--measurement-noise-correlation-time",
    },
    "actuator": {
        "time_constant_s": "--motor-time-constant",
        "gain_range": "--motor-gain-range",
        "noise_std": "--motor-noise-std",
        "noise_correlation_time_s": "--motor-noise-correlation-time",
    },
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def load_json_object(path, expected_sha256=None):
    """Read bytes once, verify their optional hash, and reject ambiguous JSON.

    Returns (absolute path string, SHA-256 hex digest, object). Domain-specific
    schema/type/range checks belong to the profile loader calling this function.
    """
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Sim2real configuration changed after launch (SHA-256 mismatch)")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)
    if not isinstance(data, dict):
        raise ValueError("Sim2real configuration must be a JSON object")
    return str(resolved), digest, data


def _validate_number(section, field, value):
    name = f"{section}.{field}"
    if field == "delay_steps":
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number (not a boolean)")
    if field == "time_constant_s":
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    elif field == "gain_range":
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    elif value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def load_quadrotor_config(path, expected_sha256=None):
    resolved, digest, data = load_json_object(path, expected_sha256)
    unknown = set(data) - {"schema_version", "system", *QUADROTOR_FIELDS}
    if unknown:
        raise ValueError(f"Unknown sim2real fields: {', '.join(sorted(unknown))}")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Sim2real schema_version must be the integer 1")
    if data.get("system") != "quadrotor":
        raise ValueError("Sim2real system must be 'quadrotor'")
    requested = {}
    for section, fields in QUADROTOR_FIELDS.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a JSON object")
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"Unknown {section} fields: {', '.join(sorted(unknown))}")
        requested[section] = {field: _validate_number(section, field, value)
                              for field, value in values.items()}
    return {"path": resolved, "sha256": digest, "requested": requested}


def _flag_index(flags, flag):
    # argparse uses the final occurrence when a custom protocol repeats a flag.
    indices = [index for index, value in enumerate(flags) if value == flag]
    if not indices or indices[-1] + 1 >= len(flags):
        raise ValueError(f"Quadrotor protocol must specify {flag}")
    return indices[-1] + 1


def apply_quadrotor_config(flags, profile=None):
    """Return copied effective argv and auditable metadata; never invent defaults."""
    flags = list(flags)
    actuator_model = flags[_flag_index(flags, "--actuator-model")]
    if profile is not None and actuator_model != "first_order":
        raise ValueError("Quadrotor sim2real profiles require the first_order actuator protocol")
    requested = profile["requested"] if profile else {"observation": {}, "actuator": {}}
    for section, values in requested.items():
        for field, value in values.items():
            flags[_flag_index(flags, QUADROTOR_FIELDS[section][field])] = str(value)
    effective = {}
    for section, fields in QUADROTOR_FIELDS.items():
        effective[section] = {}
        for field, flag in fields.items():
            raw = flags[_flag_index(flags, flag)]
            value = int(raw) if field == "delay_steps" else float(raw)
            effective[section][field] = (value if section == "actuator" and actuator_model == "ideal"
                                         else _validate_number(section, field, value))
    frequency = float(flags[_flag_index(flags, "--control-frequency")])
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Protocol --control-frequency must be finite and positive")
    metadata = {
        "schema_version": 1,
        "system": "quadrotor",
        "input": {"path": profile["path"], "sha256": profile["sha256"]} if profile else None,
        "requested": requested,
        "effective": effective,
        "actuator_model": actuator_model,
        "control_frequency_hz": frequency,
        "measurement_delay_ms": 1000.0 * effective["observation"]["delay_steps"] / frequency,
        "motor_time_constant_scope": "plant_and_mpc_and_aligned_motor_observer",
    }
    if actuator_model == "ideal":
        # The ideal implementation ignores the configured lag/gain/noise flags.
        metadata["ignored_actuator_parameters"] = effective["actuator"]
        effective["actuator"] = None
        metadata["motor_time_constant_scope"] = None
    return flags, metadata
