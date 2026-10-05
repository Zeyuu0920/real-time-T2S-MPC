"""Deterministic delayed/noisy state estimates for 3-D flight experiments.

The PyBullet state remains the simulation truth used for scoring.  Controllers
and online learners receive the output of this estimator surrogate instead.
The error is first-order Gauss--Markov rather than independent sample noise,
which better represents the temporally correlated output of a flight-state
estimator and avoids treating a finite-differenced white-noise sequence as a
physical acceleration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np


POSITION_INDICES = np.array([0, 2, 4], dtype=int)
VELOCITY_INDICES = np.array([1, 3, 5], dtype=int)
ATTITUDE_INDICES = np.array([6, 7, 8], dtype=int)
BODY_RATE_INDICES = np.array([9, 10, 11], dtype=int)


@dataclass(frozen=True)
class StateEstimatePacket:
    """One timestamped state-estimator output available to the controller.

    ``source_*`` identifies when the physical state was sampled.  ``arrival_*``
    identifies when that already-noisy sample becomes available after the
    declared estimator/transport delay.  Keeping both timestamps prevents an
    online learner from pairing a delayed state with a current actuator state
    or control command.
    """

    source_step: int
    source_time: float
    arrival_step: int
    arrival_time: float
    state: np.ndarray


def wrap_angles(values):
    """Wrap radians to ``[-pi, pi)`` without changing the input in place."""
    values = np.asarray(values, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


class DelayedNoisyStateEstimator:
    """Return a delayed 12-state estimate with colored zero-mean error.

    ``standard_deviations`` specifies the stationary one-sigma error for each
    state.  A positive ``correlation_time`` evolves the error as a stationary
    first-order Gauss--Markov process.  ``delay_steps`` represents estimator
    and transport age in complete control samples.
    """

    def __init__(
        self,
        *,
        seed,
        dt,
        standard_deviations,
        correlation_time,
        delay_steps,
    ):
        self.dt = float(dt)
        self.standard_deviations = np.asarray(
            standard_deviations, dtype=float
        ).reshape(12)
        self.correlation_time = float(correlation_time)
        self.delay_steps = int(delay_steps)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if np.any(self.standard_deviations < 0.0):
            raise ValueError("measurement standard deviations cannot be negative")
        if self.correlation_time < 0.0:
            raise ValueError("correlation_time cannot be negative")
        if self.delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")

        self.rng = np.random.default_rng(seed)
        self._error = np.zeros(12, dtype=float)
        self._packet_history = deque(maxlen=self.delay_steps + 1)
        self._initialized = False
        self._source_step = 0
        self._arrival_step = 0

    def reset(self, true_state):
        """Initialize at ``true_state`` and return the first state estimate."""
        return self.reset_packet(true_state).state.copy()

    def reset_packet(self, true_state):
        """Initialize and return a timestamped bootstrap measurement packet."""
        true_state = np.asarray(true_state, dtype=float).reshape(12)
        self._packet_history.clear()
        # Fill the unavailable pre-run history with the initial condition so
        # that the declared delay is present without an artificial zero state.
        self._source_step = 0
        self._arrival_step = 0
        self._error = self.standard_deviations * self.rng.normal(size=12)
        initial = StateEstimatePacket(
            source_step=0,
            source_time=0.0,
            arrival_step=0,
            arrival_time=0.0,
            state=self._measurement_from(true_state),
        )
        for _ in range(self.delay_steps + 1):
            self._packet_history.append(initial)
        self._initialized = True
        return initial

    def observe(self, true_state):
        """Advance one control sample and return the delayed noisy estimate."""
        return self.observe_packet(true_state).state.copy()

    def observe_packet(self, true_state):
        """Advance one sample and return its delayed timestamped packet.

        Noise is attached when a state is sampled, before the packet enters
        the delay queue.  Consequently a delayed packet retains the same
        measurement value and source timestamp while it is in transit.
        """
        if not self._initialized:
            return self.reset_packet(true_state)
        true_state = np.asarray(true_state, dtype=float).reshape(12)
        self._source_step += 1
        self._arrival_step += 1

        if self.correlation_time == 0.0:
            rho = 0.0
        else:
            rho = np.exp(-self.dt / self.correlation_time)
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho * rho))
        self._error = (
            rho * self._error
            + innovation_scale
            * self.standard_deviations
            * self.rng.normal(size=12)
        )
        sampled = StateEstimatePacket(
            source_step=self._source_step,
            source_time=self._source_step * self.dt,
            arrival_step=self._arrival_step,
            arrival_time=self._arrival_step * self.dt,
            state=self._measurement_from(true_state),
        )
        self._packet_history.append(sampled)
        delayed = self._packet_history[0]
        return replace(
            delayed,
            arrival_step=self._arrival_step,
            arrival_time=self._arrival_step * self.dt,
        )

    def _measurement_from(self, true_state):
        estimate = np.asarray(true_state, dtype=float).reshape(12) + self._error
        estimate = estimate.copy()
        estimate[ATTITUDE_INDICES] = wrap_angles(estimate[ATTITUDE_INDICES])
        return estimate


def state_noise_standard_deviations(
    position_std,
    velocity_std,
    attitude_std,
    body_rate_std,
):
    """Build the 12-state one-sigma vector from four physical groups."""
    standard_deviations = np.zeros(12, dtype=float)
    standard_deviations[POSITION_INDICES] = float(position_std)
    standard_deviations[VELOCITY_INDICES] = float(velocity_std)
    standard_deviations[ATTITUDE_INDICES] = float(attitude_std)
    standard_deviations[BODY_RATE_INDICES] = float(body_rate_std)
    return standard_deviations
