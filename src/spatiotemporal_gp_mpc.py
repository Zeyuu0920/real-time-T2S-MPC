"""Spatio-temporal GP-MPC baseline for the full 3-D quadrotor.

This module adapts the approximate spatio-temporal GP of Bartels et al.,
"Real-Time Online Learning for Model Predictive Control using a
Spatio-Temporal Gaussian Process Approximation" (2026), to the shared
quadrotor experiment. The latent inducing-point distribution is updated by
the official l4acados Kalman recursion. The controller adapter supplies the
posterior mean, its spatial Jacobian and predictive variance. In the shared
tracking benchmark the posterior mean enters the OCP, while the variance is
propagated and recorded diagnostically without tightening constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import casadi as cs
import gpytorch
import numpy as np
import torch

from src.actuated_quadrotor_dynamics import build_nominal_model
from src.realtime_dynamics_3d import RESIDUAL_DERIVATIVE_INDICES


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_L4ACADOS_SOURCE = _PROJECT_ROOT / "external" / "l4acados" / "src"
if str(_L4ACADOS_SOURCE) not in sys.path:
    sys.path.insert(0, str(_L4ACADOS_SOURCE))

try:
    from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
        BatchIndependentApproximateSpatioTemporalGPModel,
    )
except ImportError as exc:  # pragma: no cover - actionable deployment error.
    raise ImportError(
        "The STGP-MPC baseline requires the official l4acados source at "
        f"{_L4ACADOS_SOURCE}."
    ) from exc


STGP_RESIDUAL_DIM = 6
STGP_SPATIAL_DIM = 13


def pack_affine_residual_parameters(queries, means, jacobians):
    """Pack stage-wise affine GP means for the CasADi dynamics.

    ``jacobians`` follows the l4acados convention ``[output, stage, input]``.
    Each returned row represents ``b + J @ [x, u]`` and exactly recovers the
    GP posterior mean at its corresponding linearization point.
    """
    queries = np.asarray(queries, dtype=float)
    means = np.asarray(means, dtype=float)
    jacobians = np.asarray(jacobians, dtype=float)
    expected_jacobian_shape = (
        STGP_RESIDUAL_DIM,
        queries.shape[0],
        queries.shape[1],
    )
    if means.shape != (queries.shape[0], STGP_RESIDUAL_DIM):
        raise ValueError("STGP mean shape does not match the horizon queries")
    if jacobians.shape != expected_jacobian_shape:
        raise ValueError(
            f"STGP Jacobian must have shape {expected_jacobian_shape}, "
            f"got {jacobians.shape}"
        )
    stage_jacobians = np.moveaxis(jacobians, 1, 0)
    intercepts = means - np.einsum(
        "noi,ni->no", stage_jacobians, queries
    )
    return np.stack(
        [
            np.concatenate(
                (intercepts[stage], stage_jacobians[stage].reshape(-1, order="F"))
            )
            for stage in range(queries.shape[0])
        ]
    )


class Quadrotor3DSTGPDynamics:
    """Nominal dynamics plus a stage-wise affine six-axis GP residual."""

    def __init__(self, gym_env, *, motor_time_constant=None):
        self.gym_env = gym_env
        self.motor_time_constant = motor_time_constant

    def model(self):
        fields = build_nominal_model(self.gym_env, self.motor_time_constant)
        state = fields.state
        control = fields.control
        feature_input = cs.vertcat(state, control)
        input_dimension = int(feature_input.shape[0])
        parameter = cs.MX.sym(
            "stgp_affine_parameter",
            STGP_RESIDUAL_DIM * (1 + input_dimension),
        )
        intercept = parameter[:STGP_RESIDUAL_DIM]
        jacobian = cs.reshape(
            parameter[STGP_RESIDUAL_DIM:],
            STGP_RESIDUAL_DIM,
            input_dimension,
        )
        residual_acceleration = intercept + jacobian @ feature_input
        residual = cs.MX.zeros(fields.state_dim, 1)
        for output_index, derivative_index in enumerate(
            RESIDUAL_DERIVATIVE_INDICES
        ):
            residual[derivative_index] = residual_acceleration[output_index]

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.MX.sym("state_dot", fields.state_dim)
        model.u = control
        model.p = parameter
        model.u_min = np.asarray(
            self.gym_env.physical_action_bounds[0], dtype=float
        )
        model.u_max = np.asarray(
            self.gym_env.physical_action_bounds[1], dtype=float
        )
        model.z = cs.vertcat([])
        model.f_expl = fields.nominal + residual
        model.f_nominal = fields.nominal
        model.f_residual = residual
        model.x_start = fields.x_start
        model.constraints = cs.vertcat([])
        model.cost_Q = fields.cost_q
        model.cost_R = 0.05 * np.eye(4)
        model.name = "quadrotor3D_stgp_mpc"
        return model


class QuadrotorSpatioTemporalGPLearner:
    """Six-output approximate STGP with constant-complexity online updates."""

    def __init__(
        self,
        *,
        state_dimension,
        physical_velocity_limit,
        physical_attitude_limits,
        physical_body_rate_limits,
        physical_action_bounds,
        inducing_point_seed,
        control_dt,
        horizon_dt,
        inducing_point_count=80,
        spatial_lengthscale=1.5,
        temporal_lengthscale=5.0,
        linear_output_variance=1.0,
        angular_output_variance=4.0,
        observation_noise_variance=0.36,
    ):
        if state_dimension not in (12, 16):
            raise ValueError("STGP expects a 12- or 16-state quadrotor model")
        positive = {
            "control_dt": control_dt,
            "horizon_dt": horizon_dt,
            "inducing_point_count": inducing_point_count,
            "spatial_lengthscale": spatial_lengthscale,
            "temporal_lengthscale": temporal_lengthscale,
            "linear_output_variance": linear_output_variance,
            "angular_output_variance": angular_output_variance,
            "observation_noise_variance": observation_noise_variance,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("STGP dimensions, scales, variances and time steps must be positive")

        self.state_dimension = int(state_dimension)
        self.full_input_dimension = self.state_dimension + 4
        self.control_dt = float(control_dt)
        self.horizon_dt = float(horizon_dt)
        self.inducing_point_count = int(inducing_point_count)
        self.spatial_lengthscale = float(spatial_lengthscale)
        self.temporal_lengthscale = float(temporal_lengthscale)
        self.linear_output_variance = float(linear_output_variance)
        self.angular_output_variance = float(angular_output_variance)
        self.observation_noise_variance = float(observation_noise_variance)
        self.inducing_point_seed = int(inducing_point_seed)
        self.dtype = torch.float64

        # Paper-style feature selection retains dynamic variables and omits
        # absolute position. Normalize only from predeclared vehicle limits;
        # no trajectory, evaluation seed, or online observation is used to
        # choose these values. The final four features are the known motor
        # amplitude states when present, or commanded thrust otherwise.
        dynamic_indices = [1, 3, 5, 6, 7, 8, 9, 10, 11]
        velocity_limit = np.broadcast_to(
            np.asarray(physical_velocity_limit, dtype=float), (3,)
        ).copy()
        attitude_limits = np.asarray(
            physical_attitude_limits, dtype=float
        ).reshape(3)
        body_rate_limits = np.asarray(
            physical_body_rate_limits, dtype=float
        ).reshape(3)
        action_lower, action_upper = (
            np.asarray(bound, dtype=float).reshape(4)
            for bound in physical_action_bounds
        )
        if (
            np.any(velocity_limit <= 0.0)
            or np.any(attitude_limits <= 0.0)
            or np.any(body_rate_limits <= 0.0)
            or np.any(action_lower <= 0.0)
            or np.any(action_upper <= action_lower)
        ):
            raise ValueError(
                "physical feature limits and action intervals must be positive"
            )
        if self.state_dimension == 16:
            actuator_indices = [12, 13, 14, 15]
            actuator_lower = np.sqrt(action_lower)
            actuator_upper = np.sqrt(action_upper)
        else:
            actuator_indices = [12, 13, 14, 15]
            actuator_lower = action_lower
            actuator_upper = action_upper
        actuator_center = 0.5 * (actuator_lower + actuator_upper)
        actuator_scale = 0.5 * (actuator_upper - actuator_lower)
        self.feature_indices = torch.tensor(
            dynamic_indices + actuator_indices, dtype=torch.long
        )
        self.feature_center = torch.tensor(
            np.concatenate((np.zeros(9), actuator_center)), dtype=self.dtype
        )
        self.feature_scale = torch.tensor(
            np.concatenate(
                (
                    velocity_limit,
                    attitude_limits,
                    body_rate_limits,
                    actuator_scale,
                )
            ),
            dtype=self.dtype,
        )
        self.physical_velocity_limit = velocity_limit
        self.physical_attitude_limits = attitude_limits
        self.physical_body_rate_limits = body_rate_limits
        self.physical_action_lower = action_lower
        self.physical_action_upper = action_upper

        inducing_points = (
            2.0
            * torch.quasirandom.SobolEngine(
                STGP_SPATIAL_DIM,
                scramble=True,
                seed=self.inducing_point_seed,
            ).draw(self.inducing_point_count).to(dtype=self.dtype)
            - 1.0
        )
        batch_shape = torch.Size([STGP_RESIDUAL_DIM])
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=STGP_RESIDUAL_DIM,
            has_global_noise=False,
        ).to(dtype=self.dtype)
        spatial_covariance = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape,
                active_dims=range(STGP_SPATIAL_DIM),
                ard_num_dims=STGP_SPATIAL_DIM,
            ),
            batch_shape=batch_shape,
        ).to(dtype=self.dtype)
        temporal_covariance = gpytorch.kernels.MaternKernel(
            nu=1.5,
            batch_shape=batch_shape,
            active_dims=[STGP_SPATIAL_DIM],
            ard_num_dims=1,
        ).to(dtype=self.dtype)
        self.gp = BatchIndependentApproximateSpatioTemporalGPModel(
            inducing_points=inducing_points,
            likelihood=likelihood,
            spatial_covariance=spatial_covariance,
            temporal_covariance=temporal_covariance,
            spatial_input_dimension=STGP_SPATIAL_DIM,
            residual_dimension=STGP_RESIDUAL_DIM,
            dt=self.control_dt,
            dtype=self.dtype,
        )
        self.gp.likelihood.initialize(
            task_noises=torch.full(
                (STGP_RESIDUAL_DIM,),
                self.observation_noise_variance,
                dtype=self.dtype,
            )
        )
        self.gp.covar_module.kernels[0].base_kernel.initialize(
            lengthscale=torch.full(
                (STGP_RESIDUAL_DIM, 1, STGP_SPATIAL_DIM),
                self.spatial_lengthscale,
                dtype=self.dtype,
            )
        )
        self.gp.covar_module.kernels[0].initialize(
            outputscale=torch.tensor(
                [self.linear_output_variance] * 3
                + [self.angular_output_variance] * 3,
                dtype=self.dtype,
            )
        )
        self.gp.covar_module.kernels[1].initialize(
            lengthscale=torch.full(
                (STGP_RESIDUAL_DIM, 1, 1),
                self.temporal_lengthscale,
                dtype=self.dtype,
            )
        )
        self.gp.initialize()
        self.gp.eval()
        self.gp.likelihood.eval()
        self._kzz_inverse = torch.cholesky_inverse(
            self.gp.K_ZZ_chol.to_dense()
        )
        self.update_count = 0

    def _prediction_times(self, node_count, current_time):
        return float(current_time) + torch.arange(node_count, dtype=self.dtype) * self.horizon_dt

    def _augmented_prediction_inputs(self, full_inputs, current_time):
        """Return normalized spatial features with absolute query times."""
        raw = torch.as_tensor(full_inputs, dtype=self.dtype)
        raw = torch.atleast_2d(raw)
        spatial = self.spatial_features(raw)
        times = self._prediction_times(raw.shape[0], current_time)
        return torch.cat((spatial, times[:, None]), dim=-1)

    def predictive_mean_and_variance(self, full_inputs, current_time):
        """Evaluate the likelihood-level GP moments used by l4acados zoRO.

        The official residual wrapper passes the GP through its likelihood,
        so ``variance`` includes the learned observation-noise variance in
        addition to latent-function uncertainty.  We follow that convention
        exactly because these diagonal stage variances are added to the zoRO
        disturbance covariance.
        """
        augmented = self._augmented_prediction_inputs(
            full_inputs, current_time
        )
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), (
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False
            )
        ):
            prediction = self.gp.likelihood(self.gp(augmented))
        mean = prediction.mean.detach().cpu().numpy()
        variance = prediction.variance.detach().cpu().numpy()
        return mean, np.maximum(variance, 0.0)

    def official_value_jacobian_and_variance(
        self, full_inputs, current_time
    ):
        """Follow l4acados' GPyTorch residual-evaluation path exactly.

        The authors first evaluate likelihood-level mean/variance and then
        invoke ``torch.autograd.functional.jacobian`` on the sum of horizon
        predictions. Summing preserves each stage derivative because GP
        means at different query nodes are pointwise in their spatial input.
        """
        mean, variance = self.predictive_mean_and_variance(
            full_inputs, current_time
        )
        raw = torch.as_tensor(
            full_inputs, dtype=self.dtype
        ).detach().requires_grad_(True)
        raw = torch.atleast_2d(raw)

        def prediction_sum(raw_input):
            augmented = self._augmented_prediction_inputs(
                raw_input, current_time
            )
            with gpytorch.settings.fast_pred_var(), (
                gpytorch.settings.fast_computations(
                    covar_root_decomposition=False
                )
            ):
                prediction = self.gp.likelihood(self.gp(augmented))
            return prediction.mean.sum(dim=0)

        jacobian = torch.autograd.functional.jacobian(
            prediction_sum, raw
        ).detach().cpu().numpy()
        return mean, jacobian, variance

    def spatial_features(self, full_inputs):
        values = torch.as_tensor(full_inputs, dtype=self.dtype)
        values = torch.atleast_2d(values)
        if values.shape[1] != self.full_input_dimension:
            raise ValueError(
                f"STGP input must have {self.full_input_dimension} columns, "
                f"got {values.shape[1]}"
            )
        return (
            values[:, self.feature_indices] - self.feature_center
        ) / self.feature_scale

    def value_and_jacobian(self, full_inputs, current_time):
        """Return exact posterior mean and analytic spatial Jacobian."""
        raw = torch.as_tensor(full_inputs, dtype=self.dtype)
        raw = torch.atleast_2d(raw)
        spatial = self.spatial_features(raw)
        node_count = raw.shape[0]
        if self.update_count == 0:
            return (
                np.zeros((node_count, STGP_RESIDUAL_DIM), dtype=float),
                np.zeros(
                    (
                        STGP_RESIDUAL_DIM,
                        node_count,
                        self.full_input_dimension,
                    ),
                    dtype=float,
                ),
            )
        times = self._prediction_times(node_count, current_time)
        delta_time = times.unsqueeze(0) - float(self.gp.t_update)
        if torch.any(delta_time < -1e-12):
            raise ValueError("STGP query time precedes its latest online update")
        delta_time = torch.clamp(delta_time, min=0.0)
        # Match the official constant-step implementation, which propagates
        # future temporal states on integer multiples of the online update
        # interval when the MPC shooting interval differs from that interval.
        delta_time = (
            torch.round(delta_time / self.control_dt) * self.control_dt
        )

        spatial_kernel = self.gp.covar_module.kernels[0]
        lengthscale = spatial_kernel.base_kernel.lengthscale[:, 0, :]
        outputscale = spatial_kernel.outputscale
        spatial_delta = (
            spatial[None, :, None, :]
            - self.gp.inducing_points[None, None, :, :]
        )
        kernel_cross = outputscale[:, None, None] * torch.exp(
            -0.5
            * (
                spatial_delta
                / lengthscale[:, None, None, :]
            ).square().sum(dim=-1)
        )

        temporal_lengthscale = self.gp.covar_module.kernels[1].lengthscale[
            :, 0, 0
        ]
        decay_rate = np.sqrt(3.0) / temporal_lengthscale
        scaled_time = decay_rate[:, None] * delta_time
        decay = torch.exp(-scaled_time)
        transition = torch.empty(
            (STGP_RESIDUAL_DIM, node_count, 2, 2), dtype=self.dtype
        )
        transition[..., 0, 0] = decay * (1.0 + scaled_time)
        transition[..., 0, 1] = decay * delta_time
        transition[..., 1, 0] = (
            -decay * decay_rate[:, None].square() * delta_time
        )
        transition[..., 1, 1] = decay * (1.0 - scaled_time)

        latent = self.gp.latent_inducing_mean[..., 0].reshape(
            STGP_RESIDUAL_DIM, self.inducing_point_count, 2
        )
        latent_prediction = torch.einsum(
            "dnab,dmb->dnma", transition, latent
        )
        observation_map = self.gp.state_space_model.H[:, 0, :]
        inducing_prediction = torch.einsum(
            "da,dnma->dnm", observation_map, latent_prediction
        )
        coefficients = torch.einsum(
            "dij,dnj->dni", self._kzz_inverse, inducing_prediction
        )
        mean = torch.sum(kernel_cross * coefficients, dim=-1).T

        kernel_gradient = (
            -kernel_cross[..., None]
            * spatial_delta
            / lengthscale[:, None, None, :].square()
        )
        mean_spatial_jacobian = torch.einsum(
            "dnmk,dnm->dnk", kernel_gradient, coefficients
        )
        jacobian = torch.zeros(
            (
                STGP_RESIDUAL_DIM,
                node_count,
                self.full_input_dimension,
            ),
            dtype=self.dtype,
        )
        jacobian[:, :, self.feature_indices] = (
            mean_spatial_jacobian / self.feature_scale[None, None, :]
        )
        return mean.detach().cpu().numpy(), jacobian.detach().cpu().numpy()

    def update(self, full_input, target, timestamp):
        """Condition the GP on one new transition using the paper's Kalman update."""
        full_input = np.asarray(full_input, dtype=float).reshape(1, -1)
        target = np.asarray(target, dtype=float).reshape(1, STGP_RESIDUAL_DIM)
        timestamp = float(timestamp)
        prediction, _ = self.value_and_jacobian(full_input, timestamp)
        prediction_error = prediction[0] - target[0]
        features = self.spatial_features(full_input)
        target_tensor = torch.as_tensor(target, dtype=self.dtype)
        # Before the first observation the official prior is stationary in
        # time.  A diagnostic query may already have occurred at the current
        # controller time, which is later than a delayed packet's source time.
        # Anchor the still-unconditioned prior at the first *source* timestamp
        # so the causal delayed update is not mistaken for a backwards update.
        if self.update_count == 0:
            self.gp.t_update = timestamp
        time_tensor = torch.tensor([timestamp], dtype=self.dtype)
        self.gp.update(features, target_tensor, time_tensor)
        self.update_count += 1
        return prediction_error


class QuadrotorSTGPResidualAdapter:
    """l4acados residual-model interface for the online quadrotor STGP.

    The controller requests the horizon posterior mean/Jacobian during its
    preparation phase. Posterior variance has no role in this non-cautious
    OCP and is evaluated separately for experiment diagnostics.
    """

    def __init__(self, learner):
        self.learner = learner
        self.current_time = 0.0
        self.current_prediction = np.zeros((0, STGP_RESIDUAL_DIM))
        self.current_variance = np.zeros((0, STGP_RESIDUAL_DIM))
        self.current_inputs = np.zeros((0, learner.full_input_dimension))
        self.current_jacobian = np.zeros(
            (STGP_RESIDUAL_DIM, 0, learner.full_input_dimension)
        )

    def set_query_time(self, current_time):
        self.current_time = float(current_time)

    def value_and_jacobian(self, full_inputs):
        # The closed-form expressions are mathematically identical to the
        # GPyTorch autograd path (covered by a 1e-10 regression test).
        mean, jacobian = self.learner.value_and_jacobian(
            full_inputs, self.current_time
        )
        self.current_inputs = np.asarray(full_inputs, dtype=float).copy()
        self.current_prediction = mean
        self.current_jacobian = jacobian
        return self.current_prediction, self.current_jacobian

    def evaluate_current_variance(self):
        """Record posterior variance at the current MPC query point only."""
        if self.current_inputs.shape[0] == 0:
            self.current_variance = np.zeros((0, STGP_RESIDUAL_DIM))
            return self.current_variance
        _, variance = self.learner.predictive_mean_and_variance(
            self.current_inputs[:1], self.current_time
        )
        self.current_variance = variance
        return self.current_variance


def discrete_acceleration_residual_map(state_dimension, shooting_dt):
    """Map six continuous acceleration residuals into one shooting step.

    l4acados formulates the learned residual as an additive discrete-time
    state increment. Our shared methods learn physical accelerations, so the
    shooting interval converts acceleration units to velocity/body-rate
    increments. This keeps the learned six channels identical across SSI,
    T2S and STGP while respecting l4acados' discrete formulation.
    """
    if int(state_dimension) not in (12, 16):
        raise ValueError("STGP residual map expects 12 or 16 states")
    if float(shooting_dt) <= 0.0:
        raise ValueError("shooting_dt must be positive")
    residual_map = np.zeros(
        (int(state_dimension), STGP_RESIDUAL_DIM), dtype=float
    )
    residual_map[RESIDUAL_DERIVATIVE_INDICES, np.arange(6)] = float(
        shooting_dt
    )
    return residual_map


def configure_stgp_zoro_description(
    ocp,
    *,
    residual_map,
    physical_state_standard_deviations,
    process_noise_variance,
    backoff_scaling_gamma,
):
    """Attach zoRO covariance propagation to an input-constrained OCP.

    The shared tracking benchmark uses ``backoff_scaling_gamma=0`` and has no
    state bounds. The propagated covariance is therefore diagnostic only: it
    cannot alter the motor-thrust constraints or the control law.
    """
    try:
        from acados_template import ZoroDescription
    except ImportError as exc:  # pragma: no cover - deployment error.
        raise ImportError(
            "The stochastic STGP baseline requires acados ZoroDescription"
        ) from exc

    state_dimension = int(ocp.model.x.shape[0])
    physical_std = np.asarray(
        physical_state_standard_deviations, dtype=float
    ).reshape(-1)
    if physical_std.size != 12:
        raise ValueError("physical-state uncertainty must contain 12 values")
    if state_dimension == 16:
        # The controller propagates its known motor-lag states internally;
        # only a tiny numerical covariance is assigned at the current node.
        state_std = np.concatenate((physical_std, np.full(4, 1e-4)))
    elif state_dimension == 12:
        state_std = physical_std
    else:
        raise ValueError("zoRO configuration expects 12 or 16 states")

    process_variance = np.broadcast_to(
        np.asarray(process_noise_variance, dtype=float), (6,)
    ).copy()
    if np.any(process_variance < 0.0):
        raise ValueError("STGP process-noise variance cannot be negative")
    if float(backoff_scaling_gamma) < 0.0:
        raise ValueError("STGP backoff scaling cannot be negative")

    description = ZoroDescription()
    description.backoff_scaling_gamma = float(backoff_scaling_gamma)
    description.P0_mat = np.diag(state_std**2)
    description.W_mat = np.diag(process_variance)
    description.fdbk_K_mat = np.zeros((int(ocp.model.u.shape[0]), state_dimension))
    description.unc_jac_G_mat = np.asarray(residual_map, dtype=float)
    description.input_P0_diag = True
    description.input_P0 = False
    description.input_W_diag = True
    description.input_W_add_diag = True
    description.output_P_matrices = True

    # No state constraints are synthesized here. Deriving these lists from
    # the OCP keeps them empty in the shared tracking benchmark and prevents
    # covariance propagation from silently adding an artificial envelope.
    bound_count = int(np.asarray(ocp.constraints.idxbx).size)
    description.idx_lbx_t = list(range(bound_count))
    description.idx_ubx_t = list(range(bound_count))
    ocp.zoro_description = description
    return ocp
