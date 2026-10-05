"""Known first-order motor dynamics for the 3-D MPC prediction model.

The physical plant advances rotor speed at 500 Hz.  A controller that treats a
thrust command as an instantaneous body force is therefore missing four
dynamic states.  We represent rotor speed by the numerically better scaled
quantity ``a_i = sqrt(f_i)``.  Since ``f_i = k_f * omega_i**2``, the standard
first-order rotor-speed model is exactly

    da_i/dt = (sqrt(u_i) - a_i) / tau_m,

and the rigid body receives the nominal thrust ``a_i**2``.  Fixed motor-gain
mismatch and actuator noise deliberately remain plant-only unknowns.
"""

from __future__ import annotations

from types import SimpleNamespace

import casadi as cs
import numpy as np


RIGID_BODY_STATE_DIM = 12
MOTOR_STATE_DIM = 4


def motor_amplitude_from_rotor_speed(rotor_speed, thrust_coefficient):
    """Convert RPM state to ``sqrt(newton)`` motor-amplitude state."""
    return np.sqrt(float(thrust_coefficient)) * np.asarray(
        rotor_speed, dtype=float
    ).reshape(MOTOR_STATE_DIM)


def motor_amplitude_from_thrust(thrust):
    """Convert nonnegative motor thrust to the controller actuator state."""
    return np.sqrt(np.asarray(thrust, dtype=float).reshape(MOTOR_STATE_DIM))


def build_nominal_model(env, motor_time_constant=None):
    """Return nominal dynamics fields with optional four-state motor lag.

    ``motor_time_constant=None`` preserves the original 12-state model.  A
    positive time constant returns a 16-state model whose command remains the
    same four desired motor thrusts used by the real vehicle interface.
    """
    control = cs.MX.sym("motor_thrust_command", MOTOR_STATE_DIM)
    physical_start = np.asarray(env.X_GOAL, dtype=float)
    physical_q = np.diag(
        [
            10.0, 1.0, 10.0, 1.0, 15.0, 2.0,
            4.0, 4.0, 1.0, 0.2, 0.2, 0.1,
        ]
    )

    if motor_time_constant is None:
        state = cs.MX.sym("state", RIGID_BODY_STATE_DIM)
        nominal = env.symbolic.fc_func(state, control)
        x_start = physical_start
        cost_q = physical_q
        actual_thrust = control
    else:
        tau_m = float(motor_time_constant)
        if tau_m <= 0.0:
            raise ValueError("motor_time_constant must be positive")
        state = cs.MX.sym(
            "state_with_motor_amplitude",
            RIGID_BODY_STATE_DIM + MOTOR_STATE_DIM,
        )
        physical_state = state[:RIGID_BODY_STATE_DIM]
        motor_amplitude = state[RIGID_BODY_STATE_DIM:]
        actual_thrust = motor_amplitude**2
        rigid_body_dot = env.symbolic.fc_func(physical_state, actual_thrust)
        motor_amplitude_dot = (
            cs.sqrt(control) - motor_amplitude
        ) / tau_m
        nominal = cs.vertcat(rigid_body_dot, motor_amplitude_dot)
        x_start = np.concatenate(
            (physical_start, motor_amplitude_from_thrust(env.U_GOAL))
        )
        # The actuator state is included so MPC predicts lag, but the task is
        # not to keep every rotor at hover.  A tiny regularizer avoids imposing
        # a conflicting motor-state tracking objective.
        cost_q = np.zeros((RIGID_BODY_STATE_DIM + MOTOR_STATE_DIM,) * 2)
        cost_q[:RIGID_BODY_STATE_DIM, :RIGID_BODY_STATE_DIM] = physical_q
        cost_q[RIGID_BODY_STATE_DIM:, RIGID_BODY_STATE_DIM:] = (
            1e-6 * np.eye(MOTOR_STATE_DIM)
        )

    return SimpleNamespace(
        state=state,
        control=control,
        nominal=nominal,
        actual_thrust=actual_thrust,
        x_start=x_start,
        cost_q=cost_q,
        state_dim=int(state.shape[0]),
    )
