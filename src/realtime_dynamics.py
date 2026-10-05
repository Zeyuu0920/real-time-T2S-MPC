"""Parameterized planar-quadrotor dynamics for RealTimeL4CasADi."""

import casadi as cs
import numpy as np


class Quadrotor2DRealTimeDynamics:
    """Nominal dynamics plus a stage-parameterized neural residual."""

    def __init__(self, gym_env, residual_model):
        self.gym_env = gym_env
        self.residual_model = residual_model

    def model(self):
        mass = self.gym_env.MASS
        inertia_yy = self.gym_env.J[1, 1]
        gravity = self.gym_env.GRAVITY_ACC
        length = self.gym_env.L

        x = cs.MX.sym("x")
        x_dot = cs.MX.sym("x_dot")
        z = cs.MX.sym("z")
        z_dot = cs.MX.sym("z_dot")
        theta = cs.MX.sym("theta")
        theta_dot = cs.MX.sym("theta_dot")
        state = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)

        thrust_1 = cs.MX.sym("T1")
        thrust_2 = cs.MX.sym("T2")
        control = cs.vertcat(thrust_1, thrust_2)

        n_motors_per_input = 2.0
        thrust_low = self.gym_env.KF * n_motors_per_input * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MIN_PWM
            + self.gym_env.PWM2RPM_CONST
        ) ** 2
        thrust_high = self.gym_env.KF * n_motors_per_input * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MAX_PWM
            + self.gym_env.PWM2RPM_CONST
        ) ** 2

        nominal = cs.vertcat(
            x_dot,
            cs.sin(theta) * (thrust_1 + thrust_2) / mass,
            z_dot,
            cs.cos(theta) * (thrust_1 + thrust_2) / mass - gravity,
            theta_dot,
            length * (thrust_2 - thrust_1) / inertia_yy / np.sqrt(2),
        )

        network_input = cs.vertcat(state, control)
        residual_acceleration = self.residual_model(network_input)
        residual = cs.vertcat(
            0,
            residual_acceleration[0],
            0,
            residual_acceleration[1],
            0,
            residual_acceleration[2],
        )

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.MX.sym("xdot", 6)
        model.u = control
        model.u_min = thrust_low * np.ones(2)
        model.u_max = thrust_high * np.ones(2)
        model.z = cs.vertcat([])
        model.p = self.residual_model.get_sym_params()
        model.f_expl = nominal + residual
        model.f_nominal = nominal
        model.f_residual = residual
        model.x_start = np.array([0, 0, 0.75, 0, 0, 0])
        model.constraints = cs.vertcat([])
        model.name = "quadrotor2D_realtime_neural"
        return model


class Quadrotor2DRealTimeT2SDynamics:
    """RealTime T2S dynamics with the original sinusoidal time embedding."""

    def __init__(
        self,
        gym_env,
        residual_model,
        *,
        time_feat_dim=32,
        time_scale=1.0,
    ):
        if time_feat_dim <= 0 or time_feat_dim % 2:
            raise ValueError("time_feat_dim must be a positive even number")
        self.gym_env = gym_env
        self.residual_model = residual_model
        self.time_feat_dim = int(time_feat_dim)
        self.time_scale = float(time_scale)

    def model(self):
        mass = self.gym_env.MASS
        inertia_yy = self.gym_env.J[1, 1]
        gravity = self.gym_env.GRAVITY_ACC
        length = self.gym_env.L

        x = cs.MX.sym("x")
        x_dot = cs.MX.sym("x_dot")
        z = cs.MX.sym("z")
        z_dot = cs.MX.sym("z_dot")
        theta = cs.MX.sym("theta")
        theta_dot = cs.MX.sym("theta_dot")
        state = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)

        thrust_1 = cs.MX.sym("T1")
        thrust_2 = cs.MX.sym("T2")
        control = cs.vertcat(thrust_1, thrust_2)

        n_motors_per_input = 2.0
        thrust_low = self.gym_env.KF * n_motors_per_input * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MIN_PWM
            + self.gym_env.PWM2RPM_CONST
        ) ** 2
        thrust_high = self.gym_env.KF * n_motors_per_input * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MAX_PWM
            + self.gym_env.PWM2RPM_CONST
        ) ** 2

        nominal = cs.vertcat(
            x_dot,
            cs.sin(theta) * (thrust_1 + thrust_2) / mass,
            z_dot,
            cs.cos(theta) * (thrust_1 + thrust_2) / mass - gravity,
            theta_dot,
            length * (thrust_2 - thrust_1) / inertia_yy / np.sqrt(2),
        )

        # RealTimeL4CasADi requires a purely symbolic input.  The controller
        # computes the same sinusoidal embedding as the original T2S model at
        # every shooting node and supplies it as a stage parameter.
        embedding = cs.MX.sym("time_embedding", self.time_feat_dim)
        network_input = cs.vertcat(state, control, embedding)
        residual_acceleration = self.residual_model(network_input)
        residual = cs.vertcat(
            0,
            residual_acceleration[0],
            0,
            residual_acceleration[1],
            0,
            residual_acceleration[2],
        )

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.MX.sym("xdot", 6)
        model.u = control
        model.u_min = thrust_low * np.ones(2)
        model.u_max = thrust_high * np.ones(2)
        model.z = cs.vertcat([])
        model.p = cs.vertcat(embedding, self.residual_model.get_sym_params())
        model.f_expl = nominal + residual
        model.f_nominal = nominal
        model.f_residual = residual
        model.x_start = np.array([0, 0, 0.75, 0, 0, 0])
        model.constraints = cs.vertcat([])
        model.name = "quadrotor2D_realtime_t2s"
        return model
