"""SSI-MPC residual model for the shared full-3-D quadrotor pipeline.

This is a clean implementation of the random-Fourier-feature online model in
Zhou and Tzoumas, "Simultaneous System Identification and Model Predictive
Control with No Dynamic Regret" (T-RO, 2025).  The shared plant exposes a
12-state Euler-angle representation, while the SSI feature map uses the
equivalent quaternion, translational velocity, body rate, and four rotor
thrusts used by the authors' released quadrotor implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as cs
import numpy as np

from src.actuated_quadrotor_dynamics import build_nominal_model


SSI_VELOCITY_INDICES = np.array([1, 3, 5], dtype=int)
SSI_SIX_AXIS_DERIVATIVE_INDICES = np.array(
    [1, 3, 5, 9, 10, 11], dtype=int
)
SSI_THRUST_FEATURE_MODES = (
    "lagged_physical",
    "normalized_command",
)


def _ssi_derivative_indices(residual_dimension):
    """Return state-derivative channels learned by an SSI variant."""
    if residual_dimension == 3:
        return SSI_VELOCITY_INDICES
    if residual_dimension == 6:
        return SSI_SIX_AXIS_DERIVATIVE_INDICES
    raise ValueError("SSI residual dimension must be either 3 or 6")


def _euler_to_quaternion_symbolic(roll, pitch, yaw):
    """Return a scalar-first quaternion for ZYX roll-pitch-yaw angles."""
    cr, sr = cs.cos(roll / 2.0), cs.sin(roll / 2.0)
    cp, sp = cs.cos(pitch / 2.0), cs.sin(pitch / 2.0)
    cy, sy = cs.cos(yaw / 2.0), cs.sin(yaw / 2.0)
    return cs.vertcat(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _euler_to_quaternion_numpy(roll, pitch, yaw):
    """Numpy counterpart of :func:`_euler_to_quaternion_symbolic`."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def _symbolic_feature_input(
    state,
    control,
    *,
    thrust_feature_mode="lagged_physical",
    command_thrust_scale=None,
):
    quaternion = _euler_to_quaternion_symbolic(state[6], state[7], state[8])
    velocity = cs.vertcat(state[1], state[3], state[5])
    body_rate = state[9:12]
    if thrust_feature_mode == "lagged_physical":
        # Historical task-matched variant: with actuator-state augmentation,
        # observe the nominal thrust that currently reaches the rigid body.
        thrust = state[12:16] ** 2 if int(state.shape[0]) == 16 else control
    elif thrust_feature_mode == "normalized_command":
        if command_thrust_scale is None:
            raise ValueError(
                "normalized_command requires command_thrust_scale"
            )
        thrust = control / cs.DM(command_thrust_scale)
    else:
        raise ValueError(
            f"unsupported SSI thrust feature mode: {thrust_feature_mode}"
        )
    return cs.vertcat(quaternion, velocity, body_rate, thrust)


def _numpy_feature_input(
    state,
    control,
    *,
    thrust_feature_mode="lagged_physical",
    command_thrust_scale=None,
):
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    if thrust_feature_mode == "lagged_physical":
        thrust = state[12:16] ** 2 if state.size == 16 else control
    elif thrust_feature_mode == "normalized_command":
        if command_thrust_scale is None:
            raise ValueError(
                "normalized_command requires command_thrust_scale"
            )
        thrust = control / np.asarray(command_thrust_scale, dtype=float)
    else:
        raise ValueError(
            f"unsupported SSI thrust feature mode: {thrust_feature_mode}"
        )
    return np.concatenate(
        (
            _euler_to_quaternion_numpy(state[6], state[7], state[8]),
            state[SSI_VELOCITY_INDICES],
            state[9:12],
            thrust,
        )
    )


@dataclass(frozen=True)
class SSIRandomFeatures:
    """Fixed Gaussian-kernel random Fourier feature map."""

    omega: np.ndarray
    phase: np.ndarray
    thrust_feature_mode: str = "lagged_physical"
    command_thrust_scale: np.ndarray | None = None

    def __post_init__(self):
        if self.thrust_feature_mode not in SSI_THRUST_FEATURE_MODES:
            raise ValueError(
                "thrust_feature_mode must be one of "
                f"{SSI_THRUST_FEATURE_MODES}"
            )
        if self.thrust_feature_mode == "normalized_command":
            if self.command_thrust_scale is None:
                raise ValueError(
                    "normalized_command requires command_thrust_scale"
                )
            scale = np.asarray(self.command_thrust_scale, dtype=float).reshape(4)
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError("command_thrust_scale must be positive and finite")
            object.__setattr__(self, "command_thrust_scale", scale.copy())

    @classmethod
    def sample(
        cls,
        seed,
        count=50,
        kernel_std=0.01,
        *,
        thrust_feature_mode="lagged_physical",
        command_thrust_scale=None,
    ):
        if count <= 0:
            raise ValueError("SSI feature count must be positive")
        if kernel_std <= 0:
            raise ValueError("SSI kernel standard deviation must be positive")
        rng = np.random.default_rng(seed)
        # Feature input: quaternion(4), velocity(3), body rate(3), thrust(4).
        omega = rng.normal(0.0, kernel_std, size=(count, 14))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(count,))
        return cls(
            omega=omega,
            phase=phase,
            thrust_feature_mode=thrust_feature_mode,
            command_thrust_scale=command_thrust_scale,
        )

    @property
    def count(self):
        return int(self.omega.shape[0])

    def numpy(self, state, control):
        z = _numpy_feature_input(
            state,
            control,
            thrust_feature_mode=self.thrust_feature_mode,
            command_thrust_scale=self.command_thrust_scale,
        )
        return np.cos(self.omega @ z + self.phase) / np.sqrt(self.count)

    def symbolic(self, state, control):
        z = _symbolic_feature_input(
            state,
            control,
            thrust_feature_mode=self.thrust_feature_mode,
            command_thrust_scale=self.command_thrust_scale,
        )
        return cs.cos(cs.DM(self.omega) @ z + cs.DM(self.phase)) / cs.sqrt(
            self.count
        )


class Quadrotor3DNominalDynamics:
    """Nominal 3-D quadrotor with optional known motor dynamics."""

    def __init__(self, gym_env, *, motor_time_constant=None):
        self.gym_env = gym_env
        self.motor_time_constant = motor_time_constant

    def model(self):
        fields = build_nominal_model(
            self.gym_env, self.motor_time_constant
        )
        model = cs.types.SimpleNamespace()
        model.x = fields.state
        model.xdot = cs.MX.sym("state_dot", fields.state_dim)
        model.u = fields.control
        model.p = cs.MX.sym("empty_parameter", 0)
        model.u_min = np.asarray(
            self.gym_env.physical_action_bounds[0], dtype=float
        )
        model.u_max = np.asarray(
            self.gym_env.physical_action_bounds[1], dtype=float
        )
        model.z = cs.vertcat([])
        model.f_expl = fields.nominal
        model.f_nominal = fields.nominal
        model.f_residual = cs.MX.zeros(fields.state_dim, 1)
        model.x_start = fields.x_start
        model.constraints = cs.vertcat([])
        model.cost_Q = fields.cost_q
        model.cost_R = 0.05 * np.eye(4)
        model.name = "quadrotor3D_nominal_final"
        return model


class Quadrotor3DSSIDynamics:
    """Nominal dynamics plus a three- or six-axis acceleration residual."""

    def __init__(
        self,
        gym_env,
        features: SSIRandomFeatures,
        *,
        motor_time_constant=None,
        residual_dimension=3,
    ):
        self.gym_env = gym_env
        self.features = features
        self.motor_time_constant = motor_time_constant
        self.residual_dimension = int(residual_dimension)
        self.derivative_indices = _ssi_derivative_indices(
            self.residual_dimension
        )

    def model(self):
        base = Quadrotor3DNominalDynamics(
            self.gym_env, motor_time_constant=self.motor_time_constant
        ).model()
        alpha_parameter = cs.MX.sym(
            "ssi_alpha", self.residual_dimension * self.features.count
        )
        # CasADi reshape is column-major.  The matching numpy packing is
        # ``alpha.T.reshape(-1)`` in :class:`SSIOnlineLearner`.
        alpha = cs.reshape(
            alpha_parameter, self.residual_dimension, self.features.count
        )
        residual_acceleration = alpha @ self.features.symbolic(base.x, base.u)
        residual = cs.MX.zeros(int(base.x.shape[0]), 1)
        for output_index, derivative_index in enumerate(
            self.derivative_indices
        ):
            residual[derivative_index] = residual_acceleration[output_index]
        base.p = alpha_parameter
        base.f_expl = base.f_nominal + residual
        base.f_residual = residual
        base.name = "quadrotor3D_ssi_final"
        return base


class SSIOnlineLearner:
    """Per-transition online least-squares update used by SSI-MPC."""

    def __init__(
        self,
        features,
        continuous_dynamics,
        learning_rate=0.25,
        residual_dimension=3,
    ):
        if learning_rate <= 0:
            raise ValueError("SSI learning rate must be positive")
        self.features = features
        self.learning_rate = float(learning_rate)
        self.residual_dimension = int(residual_dimension)
        self.derivative_indices = _ssi_derivative_indices(
            self.residual_dimension
        )
        self.alpha = np.zeros(
            (self.residual_dimension, features.count), dtype=float
        )

        state_dim = int(continuous_dynamics.size1_in(0))
        state = cs.MX.sym("ssi_update_state", state_dim)
        control = cs.MX.sym("ssi_update_control", 4)
        dt = cs.MX.sym("ssi_update_dt")
        parameter = cs.MX.sym(
            "ssi_update_alpha", self.residual_dimension * features.count
        )
        dynamics = continuous_dynamics
        k1 = dynamics(state, control, parameter)
        k2 = dynamics(state + dt * k1 / 2.0, control, parameter)
        k3 = dynamics(state + dt * k2 / 2.0, control, parameter)
        k4 = dynamics(state + dt * k3, control, parameter)
        next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self._predict = cs.Function(
            "ssi_rk4_transition", [state, control, dt, parameter], [next_state]
        )

    @property
    def parameter_vector(self):
        return self.alpha.T.reshape(-1).copy()

    def update(self, previous_state, previous_control, observed_state, dt):
        """Update RFF weights and return the pre-update derivative error."""
        predicted = np.asarray(
            self._predict(
                previous_state,
                previous_control,
                dt,
                self.parameter_vector,
            )
        ).reshape(-1)
        error = (
            predicted[self.derivative_indices]
            - np.asarray(observed_state)[self.derivative_indices]
        ) / dt
        phi = self.features.numpy(previous_state, previous_control)
        self.alpha -= 2.0 * self.learning_rate * np.outer(error, phi)
        return error

    def update_residual_target(self, state, control, residual_target):
        """Update from a shared, timestamp-aligned acceleration target.

        ``residual_target`` is constructed independently of SSI by rolling the
        nominal dynamics through the exact physics-substep command history.
        This lets SSI, STGP and T2S learn from the same six-axis label while
        preserving SSI's original one-step online least-squares update.
        """
        target = np.asarray(residual_target, dtype=float).reshape(-1)
        if target.size < self.residual_dimension:
            raise ValueError(
                "residual_target has fewer channels than the SSI learner"
            )
        phi = self.features.numpy(state, control)
        prediction = self.alpha @ phi
        error = prediction - target[: self.residual_dimension]
        self.alpha -= 2.0 * self.learning_rate * np.outer(error, phi)
        return error
