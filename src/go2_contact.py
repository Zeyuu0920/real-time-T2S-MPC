"""Actual Go2 foot-contact wrench extraction for residual supervision.

MuJoCo reports each solved contact wrench in the contact frame.  This module
maps those wrenches to the world frame, keeps the force acting on each named
Go2 foot, and forms the exact net moment about the dry robot COM.  The liquid
payload bodies are deliberately excluded from that COM because the Acados
nominal model uses the dry Go2 mass and treats the payload coupling as the
unknown dynamics to be learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np


FOOT_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class FootContactWrench:
    """One simulation-step contact wrench, expressed in the world frame."""

    foot_forces_world: np.ndarray
    net_force_world: np.ndarray
    net_moment_world: np.ndarray
    contact_count: int


@dataclass(frozen=True)
class IntervalContactWrench:
    """Wrench averaged over one leg-control interval."""

    foot_forces_world: np.ndarray
    net_force_world: np.ndarray
    net_moment_world: np.ndarray
    simulation_samples: int
    mean_contact_count: float


class Go2FootContactReader:
    """Extract actual four-foot contact wrenches from a MuJoCo solve."""

    def __init__(self, model: mj.MjModel) -> None:
        self.model = model
        foot_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_NAMES
        ]
        if min(foot_ids) < 0:
            missing = [
                name for name, geom_id in zip(FOOT_NAMES, foot_ids) if geom_id < 0
            ]
            raise ValueError(f"missing Go2 foot collision geoms: {missing}")
        self._foot_index_by_geom = {
            int(geom_id): index for index, geom_id in enumerate(foot_ids)
        }

        # The equivalent liquid plant adds bodies whose names all start with
        # ``liquid_``.  Excluding them recovers the COM of the same dry robot
        # represented by PinGo2Model and the centroidal MPC.
        robot_body_ids = []
        for body_id in range(1, model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
            is_hidden_payload = name.startswith("liquid_") or name.startswith(
                "payload_"
            )
            if not is_hidden_payload and model.body_mass[body_id] > 0.0:
                robot_body_ids.append(body_id)
        if not robot_body_ids:
            raise ValueError("MuJoCo model contains no massive dry-robot bodies")
        self._robot_body_ids = np.asarray(robot_body_ids, dtype=int)
        self.robot_mass = float(np.sum(model.body_mass[self._robot_body_ids]))

    def dry_robot_com_world(self, data: mj.MjData) -> np.ndarray:
        """Return the mass-weighted dry-Go2 COM in world coordinates."""

        masses = self.model.body_mass[self._robot_body_ids]
        return np.sum(
            masses[:, None] * data.xipos[self._robot_body_ids], axis=0
        ) / self.robot_mass

    def read(
        self,
        data: mj.MjData,
        *,
        com_world: np.ndarray | None = None,
    ) -> FootContactWrench:
        """Return all current foot-ground forces and their centroidal wrench.

        ``mj_contactForce`` returns force:torque in the contact frame.  The
        rows of ``contact.frame`` are the contact axes in world coordinates,
        hence its transpose maps the wrench to the world frame.  MuJoCo's
        returned wrench acts on ``geom2``; the equal-and-opposite wrench acts
        on ``geom1``.
        """

        foot_forces = np.zeros((4, 3), dtype=float)
        net_moment = np.zeros(3, dtype=float)
        dry_com = (
            self.dry_robot_com_world(data)
            if com_world is None
            else np.asarray(com_world, dtype=float).reshape(3)
        )
        foot_contact_count = 0

        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom2 in self._foot_index_by_geom:
                foot_index = self._foot_index_by_geom[geom2]
                sign = 1.0
            elif geom1 in self._foot_index_by_geom:
                foot_index = self._foot_index_by_geom[geom1]
                sign = -1.0
            else:
                continue

            local_wrench = np.zeros(6, dtype=float)
            mj.mj_contactForce(self.model, data, contact_id, local_wrench)
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force_world = sign * (contact_to_world @ local_wrench[:3])
            torque_world = sign * (contact_to_world @ local_wrench[3:])
            foot_forces[foot_index] += force_world
            lever_world = np.asarray(contact.pos, dtype=float) - dry_com
            net_moment += np.cross(lever_world, force_world) + torque_world
            foot_contact_count += 1

        return FootContactWrench(
            foot_forces_world=foot_forces,
            net_force_world=np.sum(foot_forces, axis=0),
            net_moment_world=net_moment,
            contact_count=foot_contact_count,
        )


class ContactWrenchIntervalAccumulator:
    """Average 1 kHz solved contact wrenches over a 200 Hz control period."""

    def __init__(self) -> None:
        self._foot_force_sum = np.zeros((4, 3), dtype=float)
        self._net_force_sum = np.zeros(3, dtype=float)
        self._net_moment_sum = np.zeros(3, dtype=float)
        self._contact_count_sum = 0
        self._samples = 0

    def add(self, sample: FootContactWrench) -> None:
        self._foot_force_sum += np.asarray(sample.foot_forces_world, dtype=float)
        self._net_force_sum += np.asarray(sample.net_force_world, dtype=float)
        self._net_moment_sum += np.asarray(sample.net_moment_world, dtype=float)
        self._contact_count_sum += int(sample.contact_count)
        self._samples += 1

    def average_and_reset(self) -> IntervalContactWrench:
        samples = self._samples
        denominator = max(1, samples)
        result = IntervalContactWrench(
            foot_forces_world=self._foot_force_sum / denominator,
            net_force_world=self._net_force_sum / denominator,
            net_moment_world=self._net_moment_sum / denominator,
            simulation_samples=samples,
            mean_contact_count=self._contact_count_sum / denominator,
        )
        self._foot_force_sum.fill(0.0)
        self._net_force_sum.fill(0.0)
        self._net_moment_sum.fill(0.0)
        self._contact_count_sum = 0
        self._samples = 0
        return result
