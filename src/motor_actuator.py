"""Seeded 500 Hz motor-actuator models for the 3-D quadrotor plant.

The MPC output is a four-element vector of desired rotor thrusts.  The
safe-control-gym plant normally maps those thrusts to RPM instantaneously.
This module inserts the actuator state that is present on the physical
vehicle.  SSI/T2S use the matching nominal lag in their prediction model,
while fixed gain mismatch and stochastic terms stay plant-only:

* first-order rotor-speed dynamics;
* fixed, per-motor thrust-gain mismatch for one flight;
* temporally correlated (Ornstein--Uhlenbeck) relative thrust noise; and
* the existing physical thrust saturation.

The implementation uses the exact discrete-time update of a first-order
system, so its time constant does not depend on the chosen physics step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorActuatorSample:
    """One physics-substep command and the realized four-motor response."""

    command_rpm: np.ndarray
    command_thrust: np.ndarray
    lagged_rpm: np.ndarray
    lagged_thrust: np.ndarray
    motor_gain: np.ndarray
    relative_noise: np.ndarray
    applied_thrust: np.ndarray
    applied_rpm: np.ndarray


class FirstOrderMotorActuator:
    """First-order rotor dynamics with seeded gain mismatch and OU noise.

    ``gain_range`` is the half-width of a uniform, per-flight motor gain:
    ``g_i ~ U(1-gain_range, 1+gain_range)``.  ``noise_std`` is the stationary
    standard deviation of the dimensionless, multiplicative thrust noise.
    """

    name = "first_order"

    def __init__(
        self,
        *,
        dt,
        seed,
        thrust_coefficient,
        thrust_bounds,
        time_constant=0.025,
        gain_range=0.05,
        noise_std=0.01,
        noise_correlation_time=0.05,
    ):
        self.dt = float(dt)
        self.seed = int(seed)
        self.thrust_coefficient = float(thrust_coefficient)
        self.thrust_low = np.broadcast_to(
            np.asarray(thrust_bounds[0], dtype=float), (4,)
        ).copy()
        self.thrust_high = np.broadcast_to(
            np.asarray(thrust_bounds[1], dtype=float), (4,)
        ).copy()
        self.time_constant = float(time_constant)
        self.gain_range = float(gain_range)
        self.noise_std = float(noise_std)
        self.noise_correlation_time = float(noise_correlation_time)

        if self.dt <= 0.0:
            raise ValueError("actuator dt must be positive")
        if self.thrust_coefficient <= 0.0:
            raise ValueError("thrust_coefficient must be positive")
        if np.any(self.thrust_low < 0.0):
            raise ValueError("motor thrust lower bounds cannot be negative")
        if np.any(self.thrust_high <= self.thrust_low):
            raise ValueError("motor thrust upper bounds must exceed lower bounds")
        if self.time_constant <= 0.0:
            raise ValueError("motor time constant must be positive")
        if not 0.0 <= self.gain_range < 1.0:
            raise ValueError("motor gain range must lie in [0, 1)")
        if self.noise_std < 0.0:
            raise ValueError("motor noise standard deviation cannot be negative")
        if self.noise_correlation_time < 0.0:
            raise ValueError("motor noise correlation time cannot be negative")

        self._speed_decay = np.exp(-self.dt / self.time_constant)
        self._noise_decay = (
            np.exp(-self.dt / self.noise_correlation_time)
            if self.noise_correlation_time > 0.0 else 0.0
        )
        self.rng = None
        self.motor_gains = np.ones(4, dtype=float)
        self.relative_noise = np.zeros(4, dtype=float)
        self.rotor_speed = np.zeros(4, dtype=float)
        self._initialized = False

    def reset(self, command_thrust):
        """Reset one flight at a steady commanded rotor speed.

        The aircraft starts in flight, so initializing the rotor-speed state
        at the initial command avoids an artificial motor-off takeoff transient.
        Gain mismatch and OU noise are initialized from their stationary,
        seeded per-flight distributions.
        """
        command_thrust = self._clip_thrust(command_thrust)
        self.rng = np.random.default_rng(self.seed)
        self.motor_gains = self.rng.uniform(
            1.0 - self.gain_range,
            1.0 + self.gain_range,
            size=4,
        )
        self.relative_noise = self.noise_std * self.rng.normal(size=4)
        self.rotor_speed = np.sqrt(
            command_thrust / self.thrust_coefficient
        )
        self._initialized = True

    def step(self, command_rpm):
        """Advance one physics substep and return the realized motor thrust."""
        command_rpm = np.asarray(command_rpm, dtype=float).reshape(4)
        command_thrust = self._clip_thrust(
            self.thrust_coefficient * command_rpm**2
        )
        command_rpm = np.sqrt(
            command_thrust / self.thrust_coefficient
        )
        if not self._initialized:
            self.reset(command_thrust)

        self.rotor_speed = (
            self._speed_decay * self.rotor_speed
            + (1.0 - self._speed_decay) * command_rpm
        )
        innovation_scale = np.sqrt(
            max(0.0, 1.0 - self._noise_decay**2)
        )
        self.relative_noise = (
            self._noise_decay * self.relative_noise
            + innovation_scale * self.noise_std * self.rng.normal(size=4)
        )

        lagged_thrust = self.thrust_coefficient * self.rotor_speed**2
        noisy_gain = self.motor_gains * np.maximum(
            0.0, 1.0 + self.relative_noise
        )
        applied_thrust = self._clip_thrust(noisy_gain * lagged_thrust)
        applied_rpm = np.sqrt(
            applied_thrust / self.thrust_coefficient
        )
        return MotorActuatorSample(
            command_rpm.copy(),
            command_thrust.copy(),
            self.rotor_speed.copy(),
            lagged_thrust.copy(),
            self.motor_gains.copy(),
            self.relative_noise.copy(),
            applied_thrust.copy(),
            applied_rpm.copy(),
        )

    def _clip_thrust(self, thrust):
        return np.clip(
            np.asarray(thrust, dtype=float).reshape(4),
            self.thrust_low,
            self.thrust_high,
        )


class IdealMotorActuator(FirstOrderMotorActuator):
    """Uniform interface for the former instantaneous, deterministic plant."""

    name = "ideal"

    def __init__(self, *, dt, thrust_coefficient, thrust_bounds, seed=0):
        # Bypass FirstOrderMotorActuator's positive time-constant requirement;
        # only its shared bounds/conversion helpers and state fields are used.
        self.dt = float(dt)
        self.seed = int(seed)
        self.thrust_coefficient = float(thrust_coefficient)
        self.thrust_low = np.broadcast_to(
            np.asarray(thrust_bounds[0], dtype=float), (4,)
        ).copy()
        self.thrust_high = np.broadcast_to(
            np.asarray(thrust_bounds[1], dtype=float), (4,)
        ).copy()
        self.time_constant = 0.0
        self.gain_range = 0.0
        self.noise_std = 0.0
        self.noise_correlation_time = 0.0
        self.motor_gains = np.ones(4, dtype=float)
        self.relative_noise = np.zeros(4, dtype=float)
        self.rotor_speed = np.zeros(4, dtype=float)
        self._initialized = False

    def reset(self, command_thrust):
        command_thrust = self._clip_thrust(command_thrust)
        self.rotor_speed = np.sqrt(
            command_thrust / self.thrust_coefficient
        )
        self._initialized = True

    def step(self, command_rpm):
        command_rpm = np.asarray(command_rpm, dtype=float).reshape(4)
        command_thrust = self._clip_thrust(
            self.thrust_coefficient * command_rpm**2
        )
        applied_rpm = np.sqrt(
            command_thrust / self.thrust_coefficient
        )
        self.rotor_speed = applied_rpm.copy()
        return MotorActuatorSample(
            applied_rpm.copy(),
            command_thrust.copy(),
            applied_rpm.copy(),
            command_thrust.copy(),
            self.motor_gains.copy(),
            self.relative_noise.copy(),
            command_thrust.copy(),
            applied_rpm.copy(),
        )


def add_motor_actuator_arguments(parser):
    """Add the reproducible 500 Hz actuator configuration to a CLI parser."""
    parser.add_argument(
        "--actuator-model",
        choices=("first_order", "ideal"),
        default="first_order",
    )
    parser.add_argument("--motor-time-constant", type=float, default=0.025)
    parser.add_argument("--motor-gain-range", type=float, default=0.05)
    parser.add_argument("--motor-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--motor-noise-correlation-time", type=float, default=0.05
    )


def make_motor_actuator(env, args, dt, seed):
    """Build the requested actuator against the environment's physical bounds."""
    common = {
        "dt": dt,
        "seed": seed,
        "thrust_coefficient": env.KF,
        "thrust_bounds": env.physical_action_bounds,
    }
    if args.actuator_model == "ideal":
        return IdealMotorActuator(**common)
    return FirstOrderMotorActuator(
        **common,
        time_constant=args.motor_time_constant,
        gain_range=args.motor_gain_range,
        noise_std=args.motor_noise_std,
        noise_correlation_time=args.motor_noise_correlation_time,
    )
