"""Parameterized full-3-D quadrotor dynamics for RealTimeL4CasADi."""

from __future__ import annotations

import casadi as cs
import numpy as np

from src.actuated_quadrotor_dynamics import build_nominal_model


RESIDUAL_DERIVATIVE_INDICES = np.array([1, 3, 5, 9, 10, 11])


def _base_model(
    env, residual_model, embedding=None, motor_time_constant=None
):
    """Build nominal rigid-body/actuator dynamics plus a 6-D residual.

    safe-control-gym's full model uses Euler attitude coordinates:
    ``[x, xdot, y, ydot, z, zdot, roll, pitch, yaw, p, q, r]``.
    The four controls are the physical thrusts of the four Crazyflie motors.
    The learned residual corrects three translational and three body-angular
    accelerations, while kinematic identities remain physics-based.
    """
    nominal_fields = build_nominal_model(env, motor_time_constant)
    state = nominal_fields.state
    control = nominal_fields.control
    nominal = nominal_fields.nominal
    network_parts = [state, control]
    if embedding is not None:
        network_parts.append(embedding)
    residual_acceleration = residual_model(cs.vertcat(*network_parts))
    if int(residual_acceleration.shape[0]) != 6:
        raise ValueError("The full 3-D residual model must have six outputs")

    residual = cs.MX.zeros(nominal_fields.state_dim, 1)
    for output_index, derivative_index in enumerate(RESIDUAL_DERIVATIVE_INDICES):
        residual[derivative_index] = residual_acceleration[output_index]

    model = cs.types.SimpleNamespace()
    model.x = state
    model.xdot = cs.MX.sym("state_dot", nominal_fields.state_dim)
    model.u = control
    model.u_min = np.asarray(env.physical_action_bounds[0], dtype=float)
    model.u_max = np.asarray(env.physical_action_bounds[1], dtype=float)
    model.z = cs.vertcat([])
    model.f_expl = nominal + residual
    model.f_nominal = nominal
    model.f_residual = residual
    model.x_start = nominal_fields.x_start
    model.constraints = cs.vertcat([])
    # Position, attitude and altitude receive the strongest stabilization
    # weights; motor commands are penalized around the hover reference.
    model.cost_Q = nominal_fields.cost_q
    model.cost_R = 0.05 * np.eye(4)
    return model


class Quadrotor3DRealTimeDynamics:
    """Full 3-D nominal dynamics plus a stage-parameterized residual."""

    def __init__(
        self, gym_env, residual_model, *, motor_time_constant=None
    ):
        self.gym_env = gym_env
        self.residual_model = residual_model
        self.motor_time_constant = motor_time_constant

    def model(self):
        model = _base_model(
            self.gym_env,
            self.residual_model,
            motor_time_constant=self.motor_time_constant,
        )
        model.p = self.residual_model.get_sym_params()
        model.name = "quadrotor3D_realtime_neural"
        return model


class Quadrotor3DRealTimeT2SDynamics:
    """Full 3-D T2S dynamics with a symbolic sinusoidal time embedding."""

    def __init__(
        self,
        gym_env,
        residual_model,
        *,
        time_feat_dim=16,
        motor_time_constant=None,
    ):
        if time_feat_dim <= 0 or time_feat_dim % 2:
            raise ValueError("time_feat_dim must be a positive even number")
        self.gym_env = gym_env
        self.residual_model = residual_model
        self.time_feat_dim = int(time_feat_dim)
        self.motor_time_constant = motor_time_constant

    def model(self):
        embedding = cs.MX.sym("time_embedding", self.time_feat_dim)
        model = _base_model(
            self.gym_env,
            self.residual_model,
            embedding,
            motor_time_constant=self.motor_time_constant,
        )
        model.p = cs.vertcat(embedding, self.residual_model.get_sym_params())
        model.name = "quadrotor3D_realtime_t2s"
        return model
