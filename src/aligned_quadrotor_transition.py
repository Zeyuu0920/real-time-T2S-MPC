"""Causal, timestamp-aligned residual labels for the 3-D quadrotor.

The real-time experiment receives delayed state-estimator packets while motor
commands can change at any 500 Hz plant substep.  A valid online-learning
sample must therefore join data by *source* time, not by Python loop time.
This module records the command actually sent during every physics substep,
maintains the controller-side nominal motor-state observer, and compares the
next measured state with a physics-rate nominal one-step rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.actuated_quadrotor_dynamics import (
    MOTOR_STATE_DIM,
    RIGID_BODY_STATE_DIM,
)
from src.state_measurement import StateEstimatePacket, wrap_angles


@dataclass(frozen=True)
class AlignedResidualTransition:
    """One causal residual sample spanning one complete control interval."""

    source_step: int
    source_time: float
    end_step: int
    end_time: float
    available_step: int
    available_time: float
    start_arrival_step: int
    end_arrival_step: int
    state: np.ndarray
    next_state: np.ndarray
    nominal_next_state: np.ndarray
    equivalent_control: np.ndarray
    command_schedule: np.ndarray
    target: np.ndarray

    @property
    def label_delay_steps(self) -> int:
        """Number of control samples between endpoint sampling and arrival."""
        return int(self.available_step - self.end_step)

    @property
    def label_delay_seconds(self) -> float:
        return float(self.available_time - self.end_time)


class CausalAlignedTransitionBuffer:
    """Join delayed measurements and applied command histories by source step.

    The nominal model is integrated with RK4 at ``physics_dt`` using the exact
    per-substep command schedule.  For a 16-state controller, the last four
    states are a causal motor-amplitude observer driven only by those sent
    commands; hidden plant gain/noise and PyBullet motor truth are never used.
    """

    def __init__(
        self,
        *,
        nominal_derivative,
        state_dimension,
        derivative_indices,
        control_dt,
        physics_dt,
        motor_time_constant=None,
    ):
        self.nominal_derivative = nominal_derivative
        self.state_dimension = int(state_dimension)
        self.derivative_indices = np.asarray(
            derivative_indices, dtype=int
        ).reshape(-1)
        self.control_dt = float(control_dt)
        self.physics_dt = float(physics_dt)
        ratio = self.control_dt / self.physics_dt
        self.physics_steps_per_control = int(round(ratio))
        if self.state_dimension not in (
            RIGID_BODY_STATE_DIM,
            RIGID_BODY_STATE_DIM + MOTOR_STATE_DIM,
        ):
            raise ValueError("state_dimension must be 12 or 16")
        if self.control_dt <= 0.0 or self.physics_dt <= 0.0:
            raise ValueError("control_dt and physics_dt must be positive")
        if not np.isclose(ratio, self.physics_steps_per_control):
            raise ValueError("control_dt must contain an integer number of physics steps")
        if np.any(self.derivative_indices < 0) or np.any(
            self.derivative_indices >= RIGID_BODY_STATE_DIM
        ):
            raise ValueError("residual derivative indices must be physical states")
        self.has_motor_state = self.state_dimension == (
            RIGID_BODY_STATE_DIM + MOTOR_STATE_DIM
        )
        if self.has_motor_state:
            if motor_time_constant is None or float(motor_time_constant) <= 0.0:
                raise ValueError("a 16-state transition requires motor_time_constant")
            self.motor_time_constant = float(motor_time_constant)
            self._motor_decay = np.exp(
                -self.physics_dt / self.motor_time_constant
            )
        else:
            self.motor_time_constant = None
            self._motor_decay = None

        self.measurements: dict[int, StateEstimatePacket] = {}
        self.command_intervals: dict[int, np.ndarray] = {}
        self.motor_states: dict[int, np.ndarray] = {}
        self._emitted_intervals: set[int] = set()

    def reset(self, initial_packet, initial_motor_amplitude=None):
        """Clear all histories and register the source-time-zero estimate."""
        self.measurements.clear()
        self.command_intervals.clear()
        self.motor_states.clear()
        self._emitted_intervals.clear()
        self.add_measurement(initial_packet)
        if self.has_motor_state:
            if initial_motor_amplitude is None:
                raise ValueError("initial motor amplitude is required")
            self.motor_states[0] = np.asarray(
                initial_motor_amplitude, dtype=float
            ).reshape(MOTOR_STATE_DIM).copy()

    def add_measurement(self, packet):
        """Register one arrived packet under its physical source timestamp."""
        if not isinstance(packet, StateEstimatePacket):
            raise TypeError("packet must be a StateEstimatePacket")
        if packet.source_step < 0:
            raise ValueError("measurement source_step cannot be negative")
        expected_time = packet.source_step * self.control_dt
        if not np.isclose(packet.source_time, expected_time):
            raise ValueError("measurement source_time and source_step disagree")
        self.measurements[int(packet.source_step)] = packet

    def record_command_interval(self, source_step, command_schedule):
        """Record commands sent over ``[source_step, source_step + 1)``."""
        source_step = int(source_step)
        schedule = np.asarray(command_schedule, dtype=float)
        expected_shape = (self.physics_steps_per_control, MOTOR_STATE_DIM)
        if schedule.shape != expected_shape:
            raise ValueError(
                f"command_schedule must have shape {expected_shape}, "
                f"got {schedule.shape}"
            )
        if not np.all(np.isfinite(schedule)) or np.any(schedule < 0.0):
            raise ValueError("command_schedule must be finite and nonnegative")
        self.command_intervals[source_step] = schedule.copy()

        if self.has_motor_state:
            if source_step not in self.motor_states:
                raise ValueError(
                    "motor observer intervals must be recorded chronologically"
                )
            motor = self.motor_states[source_step].copy()
            for command in schedule:
                motor = (
                    self._motor_decay * motor
                    + (1.0 - self._motor_decay) * np.sqrt(command)
                )
            self.motor_states[source_step + 1] = motor

    def state_for_packet(self, packet):
        """Return a controller state whose components share one source time."""
        physical = np.asarray(packet.state, dtype=float).reshape(
            RIGID_BODY_STATE_DIM
        )
        if not self.has_motor_state:
            return physical.copy()
        source_step = int(packet.source_step)
        if source_step not in self.motor_states:
            raise ValueError(
                f"no causal motor-state estimate for source step {source_step}"
            )
        return np.concatenate((physical, self.motor_states[source_step]))

    def pop_ready(self):
        """Build every newly available, consecutive one-step transition."""
        ready = []
        for end_step in sorted(self.measurements):
            source_step = end_step - 1
            if source_step < 0 or source_step in self._emitted_intervals:
                continue
            if source_step not in self.measurements:
                continue
            if source_step not in self.command_intervals:
                continue
            if self.has_motor_state and (
                source_step not in self.motor_states
                or end_step not in self.motor_states
            ):
                continue

            start_packet = self.measurements[source_step]
            end_packet = self.measurements[end_step]
            state = self.state_for_packet(start_packet)
            next_state = self.state_for_packet(end_packet)
            schedule = self.command_intervals[source_step]
            nominal_next = self._nominal_rollout(state, schedule)
            target = (
                next_state[self.derivative_indices]
                - nominal_next[self.derivative_indices]
            ) / self.control_dt
            ready.append(
                AlignedResidualTransition(
                    source_step=source_step,
                    source_time=source_step * self.control_dt,
                    end_step=end_step,
                    end_time=end_step * self.control_dt,
                    available_step=end_packet.arrival_step,
                    available_time=end_packet.arrival_time,
                    start_arrival_step=start_packet.arrival_step,
                    end_arrival_step=end_packet.arrival_step,
                    state=state,
                    next_state=next_state,
                    nominal_next_state=nominal_next,
                    equivalent_control=np.mean(schedule, axis=0),
                    command_schedule=schedule.copy(),
                    target=target,
                )
            )
            self._emitted_intervals.add(source_step)
        return ready

    def _nominal_rollout(self, state, command_schedule):
        state = np.asarray(state, dtype=float).reshape(
            self.state_dimension
        ).copy()
        for control in command_schedule:
            state = self._rk4_step(state, control)
            # Avoid an Euler-angle branch jump in the discrete rollout.
            state[6:9] = wrap_angles(state[6:9])
        return state

    def _rk4_step(self, state, control):
        dt = self.physics_dt

        def derivative(value):
            result = self.nominal_derivative(value, control)
            if hasattr(result, "full"):
                result = result.full()
            return np.asarray(result, dtype=float).reshape(self.state_dimension)

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

