"""Three-dimensional Dryden wind and Crazyflie rotor-drag coupling.

The planar experiments deliberately suppress lateral airflow.  This module
keeps all three world-frame components for the full 3-D quadrotor.  The
longitudinal channel uses the standard first-order Dryden shaping filter and
the lateral/vertical channels use the standard second-order filters.  Wind is
converted to force with the rotor-speed-dependent drag coefficients already
stored in safe-control-gym's Crazyflie URDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as pb


@dataclass(frozen=True)
class DistributedWindSample:
    """Wind velocity and the six-axis wrench applied at one physics step.

    ``force`` is expressed in the world frame because it is compared with the
    translational plant motion.  ``torque_body`` and ``rotor_forces_body`` are
    expressed in the body frame, matching PyBullet's rotor-link convention and
    the body-rate states p, q and r.
    """

    velocity: np.ndarray
    relative_air_velocity: np.ndarray
    force: np.ndarray
    torque_body: np.ndarray
    rotor_forces_body: np.ndarray
    rotor_wind_velocities: np.ndarray
    local_mean: np.ndarray
    local_sigma: np.ndarray


class DrydenWind3D:
    """Generate a non-stationary three-axis Dryden wind realization."""

    def __init__(
        self,
        dt,
        seed,
        mean_velocity=(1.5, 0.4, 0.2),
        sigma=(0.35, 0.25, 0.15),
        length_scale=(2.0, 1.5, 1.0),
        advection_speed=1.5,
        mean_velocity_end=(3.0, 0.8, 0.4),
        sigma_end=(0.7, 0.5, 0.3),
        parameter_ramp_duration=20.0,
    ):
        self.dt = float(dt)
        self.mean_velocity = np.asarray(mean_velocity, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.advection_speed = float(advection_speed)
        self.mean_velocity_end = np.asarray(mean_velocity_end, dtype=float)
        self.sigma_end = np.asarray(sigma_end, dtype=float)
        self.parameter_ramp_duration = float(parameter_ramp_duration)

        for name, value in (
            ("mean_velocity", self.mean_velocity),
            ("sigma", self.sigma),
            ("length_scale", self.length_scale),
            ("mean_velocity_end", self.mean_velocity_end),
            ("sigma_end", self.sigma_end),
        ):
            if value.shape != (3,):
                raise ValueError(f"{name} must contain x, y and z components")
        if self.dt <= 0 or self.advection_speed <= 0:
            raise ValueError("dt and advection_speed must be positive")
        if np.any(self.sigma < 0) or np.any(self.sigma_end < 0):
            raise ValueError("turbulence intensities must be non-negative")
        if np.any(self.length_scale <= 0):
            raise ValueError("length scales must be positive")
        if self.parameter_ramp_duration <= 0:
            raise ValueError("parameter_ramp_duration must be positive")
        max_rate = self.advection_speed / np.min(self.length_scale)
        if self.dt * max_rate >= 0.2:
            raise ValueError(
                "Dryden integration step is too large for the selected "
                "length scale and advection speed"
            )

        self.rng = np.random.default_rng(seed)
        self._longitudinal_state = 0.0
        self._lateral_state = np.zeros(2, dtype=float)
        self._vertical_state = np.zeros(2, dtype=float)
        self.time = 0.0
        self.local_mean = self.mean_velocity.copy()
        self.local_sigma = self.sigma.copy()

    def _smooth_phase(self):
        phase = np.clip(self.time / self.parameter_ramp_duration, 0.0, 1.0)
        return phase * phase * (3.0 - 2.0 * phase)

    def _second_order_step(self, state, rate):
        state_0, state_1 = state
        noise_increment = np.sqrt(self.dt) * self.rng.normal()
        state_0_new = state_0 + state_1 * self.dt
        state_1_new = (
            state_1
            + (-rate**2 * state_0 - 2.0 * rate * state_1) * self.dt
            + noise_increment
        )
        state[:] = (state_0_new, state_1_new)
        gain = np.sqrt(3.0 * rate)
        return gain * (state_1_new + rate * state_0_new / np.sqrt(3.0))

    def step(self):
        phase = self._smooth_phase()
        self.local_mean = self.mean_velocity + phase * (
            self.mean_velocity_end - self.mean_velocity
        )
        self.local_sigma = self.sigma + phase * (
            self.sigma_end - self.sigma
        )

        rate_x = self.advection_speed / self.length_scale[0]
        self._longitudinal_state += (
            -rate_x * self._longitudinal_state * self.dt
            + np.sqrt(2.0 * rate_x * self.dt) * self.rng.normal()
        )
        rate_y = self.advection_speed / self.length_scale[1]
        rate_z = self.advection_speed / self.length_scale[2]
        lateral = self._second_order_step(self._lateral_state, rate_y)
        vertical = self._second_order_step(self._vertical_state, rate_z)
        normalized = np.array(
            [self._longitudinal_state, lateral, vertical], dtype=float
        )
        sample = self.local_mean + self.local_sigma * normalized
        self.time += self.dt
        return sample


class RotorDragWind3D:
    """Map a spatially nonuniform wind to forces at all four rotors.

    The Dryden filters define the wind at the vehicle centre.  A fixed
    world-frame linear gradient then evaluates a local wind velocity at each
    rotor.  Applying the four drag forces at the URDF rotor positions produces
    both the net force and the aerodynamic moment ``sum(r_i x F_i)``.

    A zero gradient recovers a spatially uniform wind.  The default gradient
    represents an off-centre/finite-width fan or a nearby flow obstruction,
    for which the two sides of a small quadrotor need not see identical flow.
    """

    def __init__(
        self,
        env,
        dt,
        seed,
        spatial_gradient=(4.0, -4.0, 0.0),
        minimum_local_wind_scale=0.25,
        **dryden_kwargs,
    ):
        self.env = env
        self.wind = DrydenWind3D(dt=dt, seed=seed, **dryden_kwargs)
        self.spatial_gradient = np.asarray(spatial_gradient, dtype=float)
        self.minimum_local_wind_scale = float(minimum_local_wind_scale)
        if self.spatial_gradient.shape != (3,):
            raise ValueError("spatial_gradient must contain x, y and z components")
        if self.minimum_local_wind_scale <= 0:
            raise ValueError("minimum_local_wind_scale must be positive")

        # Exact rotor locations from safe-control-gym's cf2x.urdf.  L is the
        # arm length from the centre to a rotor; the X-layout coordinates are
        # L/sqrt(2) in the body x and y directions.
        offset = float(self.env.L) / np.sqrt(2.0)
        self.rotor_positions_body = np.array(
            [
                [offset, offset, 0.0],
                [-offset, offset, 0.0],
                [-offset, -offset, 0.0],
                [offset, -offset, 0.0],
            ],
            dtype=float,
        )

    def _motor_rpm(self, action):
        thrust = np.asarray(action, dtype=float).reshape(-1)
        if thrust.shape != (4,):
            raise ValueError("The 3-D quadrotor requires four motor thrusts")
        low, high = self.env.physical_action_bounds
        thrust = np.clip(thrust, low, high)
        return np.sqrt(thrust / self.env.KF)

    def step(
        self,
        vehicle_velocity,
        attitude_quaternion,
        action,
        angular_velocity=(0.0, 0.0, 0.0),
    ):
        wind_velocity = self.wind.step()
        vehicle_velocity = np.asarray(vehicle_velocity, dtype=float)
        angular_velocity = np.asarray(angular_velocity, dtype=float)
        rotation_body_to_world = np.asarray(
            pb.getMatrixFromQuaternion(attitude_quaternion), dtype=float
        ).reshape(3, 3)

        rotor_positions_world = (
            rotation_body_to_world @ self.rotor_positions_body.T
        ).T
        local_scales = np.maximum(
            self.minimum_local_wind_scale,
            1.0 + rotor_positions_world @ self.spatial_gradient,
        )
        rotor_wind_world = local_scales[:, None] * wind_velocity[None, :]

        vehicle_velocity_body = rotation_body_to_world.T @ vehicle_velocity
        angular_velocity_body = rotation_body_to_world.T @ angular_velocity
        rpm = self._motor_rpm(action)
        rotor_speeds = 2.0 * np.pi * rpm / 60.0
        rotor_forces_body = np.empty((4, 3), dtype=float)
        for index, rotor_position in enumerate(self.rotor_positions_body):
            local_vehicle_velocity = (
                vehicle_velocity_body
                + np.cross(angular_velocity_body, rotor_position)
            )
            local_wind_body = (
                rotation_body_to_world.T @ rotor_wind_world[index]
            )
            relative_body = local_vehicle_velocity - local_wind_body
            drag_gain = np.asarray(self.env.DRAG_COEFF) * rotor_speeds[index]
            rotor_forces_body[index] = -drag_gain * relative_body

        force_body = np.sum(rotor_forces_body, axis=0)
        force_world = rotation_body_to_world @ force_body
        torque_body = np.sum(
            np.cross(self.rotor_positions_body, rotor_forces_body), axis=0
        )
        relative_world = vehicle_velocity - wind_velocity
        return DistributedWindSample(
            wind_velocity.copy(),
            relative_world.copy(),
            force_world.copy(),
            torque_body.copy(),
            rotor_forces_body.copy(),
            rotor_wind_world.copy(),
            self.wind.local_mean.copy(),
            self.wind.local_sigma.copy(),
        )


def add_physical_wind_3d_arguments(parser):
    """Add reproducible non-stationary 3-D wind parameters."""
    for axis, mean, sigma, length, mean_end, sigma_end in (
        ("x", 1.5, 0.35, 2.0, 3.0, 0.7),
        ("y", 0.4, 0.25, 1.5, 0.8, 0.5),
        ("z", 0.2, 0.15, 1.0, 0.4, 0.3),
    ):
        parser.add_argument(f"--mean-wind-{axis}", type=float, default=mean)
        parser.add_argument(
            f"--turbulence-sigma-{axis}", type=float, default=sigma
        )
        parser.add_argument(f"--length-scale-{axis}", type=float, default=length)
        parser.add_argument(
            f"--mean-wind-{axis}-end", type=float, default=mean_end
        )
        parser.add_argument(
            f"--turbulence-sigma-{axis}-end", type=float, default=sigma_end
        )
    parser.add_argument("--advection-speed", type=float, default=1.5)
    parser.add_argument("--wind-ramp-duration", type=float, default=20.0)
    parser.add_argument("--wind-gradient-x", type=float, default=4.0)
    parser.add_argument("--wind-gradient-y", type=float, default=-4.0)
    parser.add_argument("--wind-gradient-z", type=float, default=0.0)


def make_physical_wind_3d(env, args, dt, seed):
    return RotorDragWind3D(
        env,
        dt=dt,
        seed=seed,
        mean_velocity=tuple(
            getattr(args, f"mean_wind_{axis}") for axis in "xyz"
        ),
        sigma=tuple(
            getattr(args, f"turbulence_sigma_{axis}") for axis in "xyz"
        ),
        length_scale=tuple(
            getattr(args, f"length_scale_{axis}") for axis in "xyz"
        ),
        advection_speed=args.advection_speed,
        mean_velocity_end=tuple(
            getattr(args, f"mean_wind_{axis}_end") for axis in "xyz"
        ),
        sigma_end=tuple(
            getattr(args, f"turbulence_sigma_{axis}_end") for axis in "xyz"
        ),
        parameter_ramp_duration=args.wind_ramp_duration,
        spatial_gradient=tuple(
            getattr(args, f"wind_gradient_{axis}") for axis in "xyz"
        ),
    )
