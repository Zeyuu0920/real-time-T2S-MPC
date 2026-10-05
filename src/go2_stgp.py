"""Official l4acados approximate STGP on the shared Go2 15-D features.

The mean and spatial Jacobian enter the shared deterministic Go2 OCP. This is
explicitly the non-cautious adaptation used by the final quadrotor protocol,
not a claim to reproduce the racing paper's chance-constrained controller.
"""
import numpy as np
import torch
import gpytorch
from src.spatiotemporal_gp_mpc import (
    QuadrotorSpatioTemporalGPLearner,
    BatchIndependentApproximateSpatioTemporalGPModel,
)


class Go2SpatioTemporalGPLearner(QuadrotorSpatioTemporalGPLearner):
    """Reuse the tested posterior/Jacobian/update operations, with Go2 geometry."""

    def __init__(self, *, control_dt=0.02, inducing_point_count=80,
                 spatial_lengthscale=1.5, temporal_lengthscale=5.0,
                 observation_noise_variance=0.36):
        self.full_input_dimension = 15
        self.dtype = torch.float64
        self.control_dt = float(control_dt)
        self.horizon_dt = 0.03
        self.inducing_point_count = int(inducing_point_count)
        self.query_offsets = None
        self.feature_indices = torch.arange(15)
        # Fixed operating-domain scaling, not selected from evaluation data.
        # Last six features already have the author's wrench / 100 scaling.
        self.feature_center = torch.tensor([0.] * 11 + [1.5] + [0.] * 3, dtype=self.dtype)
        self.feature_scale = torch.tensor(
            [.5, .5, 3.14, 1., 1., .5, 2., 2., 2., 1., 1., 2., .2, .2, .2],
            dtype=self.dtype,
        )
        points = (2 * torch.quasirandom.SobolEngine(15, scramble=True, seed=0)
                  .draw(self.inducing_point_count).to(self.dtype) - 1)
        batch = torch.Size([6])
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=6, has_global_noise=False).to(self.dtype)
        spatial = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
            batch_shape=batch, active_dims=range(15), ard_num_dims=15),
            batch_shape=batch).to(self.dtype)
        temporal = gpytorch.kernels.MaternKernel(
            nu=1.5, batch_shape=batch, active_dims=[15], ard_num_dims=1).to(self.dtype)
        self.gp = BatchIndependentApproximateSpatioTemporalGPModel(
            inducing_points=points, likelihood=likelihood,
            spatial_covariance=spatial, temporal_covariance=temporal,
            spatial_input_dimension=15, residual_dimension=6,
            dt=self.control_dt, dtype=self.dtype)
        likelihood.initialize(task_noises=torch.full((6,), observation_noise_variance, dtype=self.dtype))
        spatial.base_kernel.initialize(lengthscale=torch.full((6, 1, 15), spatial_lengthscale, dtype=self.dtype))
        spatial.initialize(outputscale=torch.tensor([1., 1., 1., 4., 4., 4.], dtype=self.dtype))
        temporal.initialize(lengthscale=torch.full((6, 1, 1), temporal_lengthscale, dtype=self.dtype))
        self.gp.initialize()
        self.gp.eval()
        self.gp.likelihood.eval()
        self._kzz_inverse = torch.cholesky_inverse(self.gp.K_ZZ_chol.to_dense())
        self.update_count = 0

    def _prediction_times(self, node_count, current_time):
        if self.query_offsets is None:
            return super()._prediction_times(node_count, current_time)
        if len(self.query_offsets) != node_count:
            raise ValueError("GP prediction offsets do not match query batch")
        return float(current_time) + torch.tensor(self.query_offsets, dtype=self.dtype)

    def horizon_parameters(self, expansion_features, time_now, first_duration):
        """Pack the same first-order interface as T2S; ignore T2S time codes.

        GP time enters its kernel directly. The 16 zero Jacobian columns here
        are only an adapter to the existing symbolic interface, not GP inputs.
        """
        points = np.asarray(expansion_features, dtype=float)
        offsets = np.r_[0., first_duration + .03 * np.arange(len(points) - 1)]
        self.query_offsets = offsets
        try:
            means, jac = self.value_and_jacobian(points[:, :15], time_now)
        finally:
            self.query_offsets = None
        full_jac = np.zeros((len(points), 6, 31))
        full_jac[:, :, :15] = jac.transpose(1, 0, 2)
        return np.concatenate((points, means, full_jac.transpose(0, 2, 1).reshape(len(points), -1)), axis=1)
