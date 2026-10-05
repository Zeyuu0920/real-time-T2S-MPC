"""Quadrotor environment with per-physics-substep plant hooks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pybullet as pb

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor


class MultirateWindQuadrotor(Quadrotor):
    """Update actuator dynamics and disturbances at every PyBullet substep.

    ``Quadrotor.step`` normally computes one disturbance force and holds it for
    all ``PYB_STEPS_PER_CTRL`` physics steps.  This subclass leaves the normal
    behavior untouched until a callback or actuator model is installed.  A
    physics-rate command schedule may release a newly completed MPC action
    partway through the control interval; actuator state and disturbance are
    evaluated immediately before every individual physics step.
    """

    def __init__(self, *args, **kwargs):
        self._substep_disturbance_callback: Callable[[int], object] | None = None
        self._substep_actuator_model = None
        self.last_actuator_samples = []
        self.current_applied_thrust = None
        super().__init__(*args, **kwargs)

    def set_substep_disturbance_callback(
        self,
        callback: Callable[[int], object] | None,
    ) -> None:
        self._substep_disturbance_callback = callback

    def set_substep_actuator_model(self, model, initial_thrust) -> None:
        """Install and initialize a seeded physics-rate motor model."""
        self._substep_actuator_model = model
        model.reset(initial_thrust)
        self.last_actuator_samples = []
        self.current_applied_thrust = np.asarray(
            initial_thrust, dtype=float
        ).reshape(4).copy()

    def step(self, action, command_schedule=None):
        callback = self._substep_disturbance_callback
        actuator = self._substep_actuator_model
        if callback is None and actuator is None:
            return super().step(action)

        aggregate_steps = self.PYB_STEPS_PER_CTRL
        if command_schedule is None:
            command_rpm = super().before_step(action)
            command_rpm_schedule = np.repeat(
                np.asarray(command_rpm, dtype=float)[None, :],
                aggregate_steps,
                axis=0,
            )
        else:
            commands = np.asarray(command_schedule, dtype=float)
            if commands.shape != (aggregate_steps, 4):
                raise ValueError(
                    "command_schedule must have shape "
                    f"({aggregate_steps}, 4), got {commands.shape}"
                )
            # Run the usual reset check and preserve the last raw command for
            # environment diagnostics.  Convert each scheduled physical
            # command through the same clipping/motor mapping as ``step``.
            super().before_step(commands[-1])
            command_rpm_schedule = np.stack(
                [self._preprocess_control(command) for command in commands]
            )
        self.last_actuator_samples = []
        try:
            # Reuse BaseAviary's tested one-step force application and physics
            # path, but supply a new force before every individual substep.
            self.PYB_STEPS_PER_CTRL = 1
            for substep in range(aggregate_steps):
                command_rpm = command_rpm_schedule[substep]
                if actuator is None:
                    applied_rpm = command_rpm
                    self.current_applied_thrust = (
                        self.KF * np.asarray(command_rpm, dtype=float) ** 2
                    )
                else:
                    actuator_sample = actuator.step(command_rpm)
                    self.last_actuator_samples.append(actuator_sample)
                    applied_rpm = actuator_sample.applied_rpm
                    self.current_applied_thrust = (
                        actuator_sample.applied_thrust.copy()
                    )

                disturbance = callback(substep) if callback is not None else None
                if disturbance is not None and hasattr(
                    disturbance, "rotor_forces_body"
                ):
                    rotor_forces = np.asarray(
                        disturbance.rotor_forces_body, dtype=float
                    )
                    if rotor_forces.shape != (4, 3):
                        raise ValueError(
                            "rotor_forces_body must have shape (4, 3)"
                        )
                    # External forces accumulate until the following
                    # stepSimulation call inside _advance_simulation.  Applying
                    # each force to its URDF rotor link lets PyBullet generate
                    # the associated moment about the vehicle centre.
                    for rotor, force_body in enumerate(rotor_forces):
                        pb.applyExternalForce(
                            self.DRONE_ID,
                            linkIndex=rotor,
                            forceObj=force_body,
                            posObj=[0.0, 0.0, 0.0],
                            flags=pb.LINK_FRAME,
                            physicsClientId=self.PYB_CLIENT,
                        )
                    super()._advance_simulation(applied_rpm, None)
                else:
                    force = (
                        None if disturbance is None
                        else np.asarray(disturbance, dtype=float).reshape(3)
                    )
                    super()._advance_simulation(applied_rpm, force)
        finally:
            self.PYB_STEPS_PER_CTRL = aggregate_steps

        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return super().after_step(obs, rew, done, info)
