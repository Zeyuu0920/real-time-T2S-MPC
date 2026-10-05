"""Frozen, author-style calibration for the existing six-output STGP learner.

No offline observations/posterior are loaded. The artifact defines only feature
normalization, GP priors/hyperparameters and spatial inducing locations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gpytorch
import numpy as np
import torch

from src.spatiotemporal_gp_mpc import (
    BatchIndependentApproximateSpatioTemporalGPModel,
    QuadrotorSpatioTemporalGPLearner,
)

FEATURE_INDICES = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RESIDUAL_NAMES = ["x_ddot", "y_ddot", "z_ddot", "p_dot", "q_dot", "r_dot"]


def load_calibration(path, evaluation_seed=None):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != "quadrotor_stgp_calibration_A_v1":
        raise ValueError("Unsupported STGP calibration schema")
    if data["feature_indices"] != FEATURE_INDICES or data["residual_names"] != RESIDUAL_NAMES:
        raise ValueError("Calibration features/outputs do not match this controller")
    used = set(data["fit_seeds"]) | set(data["validation_seeds"])
    if used & set(range(42, 52)):
        raise ValueError("Evaluation seeds 42–51 must not appear in calibration data")
    if evaluation_seed is not None and evaluation_seed in used:
        raise ValueError("Calibration and evaluation seeds must be disjoint")
    expected = {
        "feature_center": (13,), "feature_scale": (13,),
        "output_mean": (6,), "output_std": (6,),
        "spatial_lengthscales_normalized": (6, 13),
        "temporal_lengthscales_s": (6,), "output_variances_normalized": (6,),
        "noise_variances_normalized": (6,), "inducing_points_normalized": (80, 13),
    }
    for key, shape in expected.items():
        array = np.asarray(data[key], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}")
        if key not in ("feature_center", "output_mean", "inducing_points_normalized") and np.any(array <= 0):
            raise ValueError(f"{key} must be strictly positive")
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


class CalibratedQuadrotorSTGPLearner(QuadrotorSpatioTemporalGPLearner):
    """Same online Kalman recursion and analytic MPC queries; calibrated prior."""

    def __init__(self, *, calibration_path, evaluation_seed=None, **kwargs):
        data, digest = load_calibration(calibration_path, evaluation_seed)
        if kwargs["state_dimension"] != 16 or kwargs.get("inducing_point_count", 80) != 80:
            raise ValueError("This frozen calibration requires 16 states and 80 inducing points")
        if not np.isclose(kwargs["control_dt"], data["control_dt_s"]):
            raise ValueError("Calibration/control sample periods do not match")
        super().__init__(**kwargs)
        self.calibration = data
        self.calibration_path = str(Path(calibration_path).resolve())
        self.calibration_sha256 = digest
        self.feature_center = torch.tensor(data["feature_center"], dtype=self.dtype)
        self.feature_scale = torch.tensor(data["feature_scale"], dtype=self.dtype)
        self.prior_mean = np.asarray(data["output_mean"], dtype=float)
        self.output_std = np.asarray(data["output_std"], dtype=float)
        batch = torch.Size([6])
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=6, has_global_noise=False,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-12),
        ).to(dtype=self.dtype)
        spatial = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch, ard_num_dims=13, active_dims=range(13)),
            batch_shape=batch,
        ).to(dtype=self.dtype)
        temporal = gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=batch, ard_num_dims=1, active_dims=[13]).to(dtype=self.dtype)
        mean = gpytorch.means.ConstantMean(batch_shape=batch).to(dtype=self.dtype)
        mean.initialize(constant=torch.tensor(self.prior_mean, dtype=self.dtype))
        output_variance = np.asarray(data["output_variances_normalized"]) * self.output_std ** 2
        noise_variance = np.asarray(data["noise_variances_normalized"]) * self.output_std ** 2
        likelihood.initialize(task_noises=torch.tensor(noise_variance, dtype=self.dtype))
        spatial.initialize(outputscale=torch.tensor(output_variance, dtype=self.dtype))
        spatial.base_kernel.initialize(lengthscale=torch.tensor(data["spatial_lengthscales_normalized"], dtype=self.dtype)[:, None, :])
        temporal.initialize(lengthscale=torch.tensor(data["temporal_lengthscales_s"], dtype=self.dtype)[:, None, None])
        self.gp = BatchIndependentApproximateSpatioTemporalGPModel(
            inducing_points=torch.tensor(data["inducing_points_normalized"], dtype=self.dtype),
            likelihood=likelihood, mean=mean, spatial_covariance=spatial,
            temporal_covariance=temporal, spatial_input_dimension=13,
            residual_dimension=6, dt=self.control_dt, dtype=self.dtype,
        )
        self.gp.initialize()
        self.gp.eval()
        self.gp.likelihood.eval()
        for parameter in self.gp.parameters():
            parameter.requires_grad_(False)
        self._kzz_inverse = torch.cholesky_inverse(self.gp.K_ZZ_chol.to_dense())
        self.update_count = 0

    def value_and_jacobian(self, full_inputs, current_time):
        # The inherited analytic expression computes the centered latent GP.
        # Constant prior mean has zero spatial derivative and must be restored.
        mean, jacobian = super().value_and_jacobian(full_inputs, current_time)
        return mean + self.prior_mean[None, :], jacobian

    def metadata(self):
        data = self.calibration
        return {
            "stgp_version": "A_offline_calibrated_online_posterior",
            "stgp_calibration_path": self.calibration_path,
            "stgp_calibration_sha256": self.calibration_sha256,
            "stgp_calibration_fit_seeds": ";".join(map(str, data["fit_seeds"])),
            "stgp_calibration_validation_seeds": ";".join(map(str, data["validation_seeds"])),
            "stgp_calibration_sample_count": data["training_sample_count"],
            "stgp_inducing_domain": "offline_optimized_normalized_calibration_features",
            "stgp_feature_normalization": "fit_only_empirical_mean_and_sample_std",
            "stgp_hyperparameter_source": "author_style_independent_offline_marginal_likelihood",
            "stgp_prior_mean": ";".join(map(str, self.prior_mean)),
            "stgp_inducing_point_seed": data["optimization_seed"],
            "stgp_spatial_lengthscale": json.dumps(data["spatial_lengthscales_normalized"]),
            "stgp_temporal_lengthscale_s": json.dumps(data["temporal_lengthscales_s"]),
            "stgp_linear_output_variance": json.dumps((np.asarray(data["output_variances_normalized"]) * self.output_std ** 2)[:3].tolist()),
            "stgp_angular_output_variance": json.dumps((np.asarray(data["output_variances_normalized"]) * self.output_std ** 2)[3:].tolist()),
            "stgp_observation_noise_variance": json.dumps((np.asarray(data["noise_variances_normalized"]) * self.output_std ** 2).tolist()),
            "stgp_initial_online_observations": 0,
        }
