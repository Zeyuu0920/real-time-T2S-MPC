"""Reference generators for Go2 locomotion tasks."""

from __future__ import annotations

import numpy as np


def path_following_body_command(
    measured_state: np.ndarray,
    *,
    target_speed: float,
    lateral_position_reference: float,
    heading_reference: float,
    lateral_gain: float,
    heading_gain: float,
) -> np.ndarray:
    """Return the baseline-style ``[vx_body, vy_body, yaw_rate]`` command."""

    measured_state = np.asarray(measured_state, dtype=float).reshape(12)
    if target_speed < 0.0:
        raise ValueError("target_speed cannot be negative")
    if lateral_gain < 0.0 or heading_gain < 0.0:
        raise ValueError("path-following gains cannot be negative")

    direction = float(np.sign(target_speed))
    lateral_error = measured_state[1] - float(lateral_position_reference)
    heading_error = np.arctan2(
        np.sin(measured_state[5] - float(heading_reference)),
        np.cos(measured_state[5] - float(heading_reference)),
    )
    return np.array(
        [
            target_speed,
            -direction * lateral_gain * lateral_error,
            -heading_gain * heading_error,
        ],
        dtype=float,
    )


def velocity_height_state_reference(
    measured_state: np.ndarray,
    *,
    target_speed: float,
    target_height: float,
    lateral_position_reference: float,
    heading_reference: float,
    lateral_gain: float,
    heading_gain: float,
    step_seconds: float,
    horizon_steps: int,
) -> np.ndarray:
    """Return the path-following rolling reference at all ``N + 1`` nodes.

    The horizon starts at the measured pose. A global lateral-line error and
    heading error first produce body-frame lateral velocity and yaw-rate
    commands, following the public baseline implementation. Those commands
    are then forward-integrated through the local MPC horizon. Longitudinal
    position remains rolling, while lateral position and yaw are stabilized
    to fixed world-frame references through the outer command loop.
    """

    measured_state = np.asarray(measured_state, dtype=float).reshape(12)
    if target_speed < 0.0:
        raise ValueError("target_speed cannot be negative")
    if target_height <= 0.0 or step_seconds <= 0.0:
        raise ValueError("target_height and step_seconds must be positive")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    body_command = path_following_body_command(
        measured_state,
        target_speed=target_speed,
        lateral_position_reference=lateral_position_reference,
        heading_reference=heading_reference,
        lateral_gain=lateral_gain,
        heading_gain=heading_gain,
    )
    reference = np.zeros((12, horizon_steps + 1), dtype=float)
    reference[2] = target_height
    reference[0:2, 0] = measured_state[0:2]
    reference[5, 0] = measured_state[5]

    for stage in range(horizon_steps + 1):
        yaw = reference[5, stage]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        reference[6, stage] = (
            cosine * body_command[0] - sine * body_command[1]
        )
        reference[7, stage] = (
            sine * body_command[0] + cosine * body_command[1]
        )
        reference[11, stage] = body_command[2]
        if stage == horizon_steps:
            continue
        reference[0:2, stage + 1] = (
            reference[0:2, stage]
            + step_seconds * reference[6:8, stage]
        )
        reference[5, stage + 1] = (
            reference[5, stage] + step_seconds * body_command[2]
        )
    return reference


def straight_line_state_reference(
    initial_state: np.ndarray,
    *,
    current_time: float,
    target_speed: float,
    target_height: float,
    step_seconds: float,
    horizon_steps: int,
) -> np.ndarray:
    """Return the common absolute-time reference at all ``N + 1`` nodes.

    State order is ``[p_world, rpy, v_world, omega_world]``.  Unlike a
    velocity-command reference anchored to the measured state, this reference
    is independent of controller performance and is therefore identical for
    every method at the same experiment time.
    """

    initial_state = np.asarray(initial_state, dtype=float).reshape(12)
    if current_time < 0.0:
        raise ValueError("current_time cannot be negative")
    if target_speed < 0.0:
        raise ValueError("target_speed cannot be negative")
    if target_height <= 0.0 or step_seconds <= 0.0:
        raise ValueError("target_height and step_seconds must be positive")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    absolute_times = current_time + step_seconds * np.arange(horizon_steps + 1)
    reference = np.zeros((12, horizon_steps + 1), dtype=float)
    reference[0] = initial_state[0] + target_speed * absolute_times
    reference[1] = initial_state[1]
    reference[2] = target_height
    reference[5] = initial_state[5]
    reference[6] = target_speed
    return reference
