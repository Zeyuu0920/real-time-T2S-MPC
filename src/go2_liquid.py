"""Equivalent-mechanical liquid payload for the full-order Go2 MuJoCo plant.

The liquid model lives only in the MuJoCo plant.  The Acados centroidal model
continues to use the nominal dry-robot mass and never receives the two slosh
coordinates.  A first horizontal slosh mode is represented by one modal mass
with two orthogonal slide joints; the remaining liquid mass moves rigidly with
the tank.  This is the standard first-mode mass--spring approximation for a
partially filled rectangular tank.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np
import pinocchio as pin

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import XML_PATH, MuJoCo_GO2_Model
from src.go2_friction import (
    SmoothRandomFrictionConfig, SmoothRandomFrictionField,
    MuJoCoRandomFrictionGo2Model, hide_upstream_world_geometries,
    add_smooth_random_friction_ground,
)


GRAVITY = 9.81
ROBOT_NQ = 19
ROBOT_NV = 18


@dataclass(frozen=True)
class LiquidProperties:
    """Physical first-mode parameters at one fill depth."""

    fill_depth: float
    liquid_mass: float
    modal_mass: float
    rigid_mass: float
    angular_frequency: float
    frequency_hz: float
    stiffness: float
    damping: float
    equivalent_pendulum_length: float


@dataclass(frozen=True)
class LiquidTankConfig:
    """Geometry and protocol for a square/rectangular carried liquid tank."""

    length: float = 0.20
    width: float = 0.20
    height: float = 0.12
    initial_fill_depth: float = 0.06
    final_fill_depth: float = 0.03
    liquid_density: float = 1000.0
    container_mass: float = 0.60
    damping_ratio: float = 0.05
    mount_position: tuple[float, float, float] = (0.0, 0.0, 0.14)
    maximum_modal_travel: float = 0.07
    drain_start: float = 5.0
    drain_end: float = 45.0

    def validate(self) -> None:
        if min(self.length, self.width, self.height) <= 0.0:
            raise ValueError("tank dimensions must be positive")
        if not 0.0 < self.initial_fill_depth < self.height:
            raise ValueError("initial_fill_depth must lie inside the tank")
        if not 0.0 < self.final_fill_depth <= self.initial_fill_depth:
            raise ValueError(
                "final_fill_depth must be positive and no larger than the initial depth"
            )
        if self.liquid_density <= 0.0 or self.container_mass <= 0.0:
            raise ValueError("liquid density and container mass must be positive")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio cannot be negative")
        if self.maximum_modal_travel <= 0.0:
            raise ValueError("maximum_modal_travel must be positive")
        if self.drain_end <= self.drain_start:
            raise ValueError("drain_end must be greater than drain_start")

    def properties(self, fill_depth: float) -> LiquidProperties:
        """Return the first longitudinal mode for the requested liquid depth.

        ``length`` is used in both horizontal directions in the square-tank
        benchmark.  A non-square tank would require different x/y modal
        parameters and is intentionally not claimed by this implementation.
        """

        self.validate()
        fill_depth = float(fill_depth)
        if not 0.0 < fill_depth < self.height:
            raise ValueError("fill_depth must lie inside the tank")
        if not np.isclose(self.length, self.width):
            raise ValueError(
                "the two-axis shared-modal-mass model requires a square tank"
            )

        liquid_mass = self.liquid_density * self.length * self.width * fill_depth
        depth_ratio = fill_depth / self.length
        tanh_term = np.tanh(np.pi * depth_ratio)
        angular_frequency = np.sqrt(
            GRAVITY * np.pi / self.length * tanh_term
        )
        modal_fraction = 8.0 / np.pi**3 * tanh_term / depth_ratio
        # Numerical roundoff aside, the analytical first-mode fraction lies
        # in (0, 1).  Clipping protects MuJoCo from a malformed geometry.
        modal_mass = liquid_mass * float(np.clip(modal_fraction, 0.0, 1.0))
        rigid_mass = liquid_mass - modal_mass
        stiffness = modal_mass * angular_frequency**2
        damping = (
            2.0 * self.damping_ratio * modal_mass * angular_frequency
        )
        return LiquidProperties(
            fill_depth=fill_depth,
            liquid_mass=liquid_mass,
            modal_mass=modal_mass,
            rigid_mass=rigid_mass,
            angular_frequency=angular_frequency,
            frequency_hz=angular_frequency / (2.0 * np.pi),
            stiffness=stiffness,
            damping=damping,
            equivalent_pendulum_length=GRAVITY / angular_frequency**2,
        )

    def fill_depth_at(self, time_s: float, protocol: str) -> float:
        """Return fixed or smoothly decreasing fill depth at ``time_s``."""

        if protocol == "liquid_fixed":
            return self.initial_fill_depth
        if protocol != "liquid_drain":
            raise ValueError("liquid protocol must be liquid_fixed or liquid_drain")
        progress = np.clip(
            (float(time_s) - self.drain_start)
            / (self.drain_end - self.drain_start),
            0.0,
            1.0,
        )
        # Smooth endpoints avoid impulsive mass/stiffness derivatives.
        progress = progress * progress * (3.0 - 2.0 * progress)
        return float(
            self.initial_fill_depth
            + progress * (self.final_fill_depth - self.initial_fill_depth)
        )


@dataclass(frozen=True)
class LiquidWrenchSample:
    """Payload-on-robot mount wrench expressed in the world frame."""

    force: np.ndarray
    torque: np.ndarray
    liquid_mass: float
    fill_depth: float
    frequency_hz: float
    modal_position: np.ndarray
    modal_velocity: np.ndarray


def _sphere_inertia(mass: float, radius: float) -> np.ndarray:
    value = 0.4 * mass * radius**2
    return np.full(3, value, dtype=float)


def _box_inertia(mass: float, half_sizes: np.ndarray) -> np.ndarray:
    x, y, z = 2.0 * np.asarray(half_sizes, dtype=float)
    return mass / 12.0 * np.array(
        [y * y + z * z, x * x + z * z, x * x + y * y], dtype=float
    )


class MuJoCoLiquidGo2Model(MuJoCo_GO2_Model):
    """Go2 model augmented with an unobserved, dynamically coupled tank."""

    def __init__(
        self,
        config: LiquidTankConfig,
        *,
        protocol: str = "liquid_fixed",
        friction_config: SmoothRandomFrictionConfig | None = None,
    ) -> None:
        config.validate()
        if protocol not in {"liquid_fixed", "liquid_drain"}:
            raise ValueError("unsupported liquid protocol")
        self.liquid_config = config
        self.liquid_protocol = protocol
        initial = config.properties(config.initial_fill_depth)

        spec = mj.MjSpec.from_file(str(XML_PATH))
        base = spec.body("base_link")
        tank = base.add_body(
            name="liquid_tank",
            pos=list(config.mount_position),
        )
        tank.add_geom(
            name="liquid_tank_shell",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[config.length / 2.0, config.width / 2.0, config.height / 2.0],
            mass=config.container_mass,
            contype=0,
            conaffinity=0,
            group=2,
            rgba=[0.72, 0.82, 0.92, 0.28],
        )
        tank.add_site(
            name="liquid_tank_mount",
            pos=[0.0, 0.0, -config.height / 2.0],
            size=[0.012, 0.0, 0.0],
            rgba=[0.2, 0.2, 0.2, 1.0],
        )

        liquid_center_z = -config.height / 2.0 + initial.fill_depth / 2.0
        rigid = tank.add_body(
            name="liquid_rigid_mass",
            pos=[0.0, 0.0, liquid_center_z],
        )
        rigid.add_geom(
            name="liquid_rigid_visual",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[
                0.92 * config.length / 2.0,
                0.92 * config.width / 2.0,
                0.92 * initial.fill_depth / 2.0,
            ],
            mass=max(initial.rigid_mass, 1e-6),
            contype=0,
            conaffinity=0,
            group=2,
            rgba=[0.05, 0.42, 0.95, 0.32],
        )

        modal = tank.add_body(
            name="liquid_modal_mass",
            pos=[0.0, 0.0, liquid_center_z],
        )
        joint_arguments = dict(
            type=mj.mjtJoint.mjJNT_SLIDE,
            stiffness=initial.stiffness,
            damping=initial.damping,
            limited=True,
            range=[-config.maximum_modal_travel, config.maximum_modal_travel],
            margin=0.005,
        )
        modal.add_joint(
            name="liquid_slosh_x",
            axis=[1.0, 0.0, 0.0],
            **joint_arguments,
        )
        modal.add_joint(
            name="liquid_slosh_y",
            axis=[0.0, 1.0, 0.0],
            **joint_arguments,
        )
        self._modal_radius = 0.025
        modal.add_geom(
            name="liquid_modal_visual",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[self._modal_radius, 0.0, 0.0],
            mass=max(initial.modal_mass, 1e-6),
            contype=0,
            conaffinity=0,
            group=2,
            rgba=[0.0, 0.25, 1.0, 0.92],
        )

        spec.add_sensor(
            name="liquid_mount_force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="liquid_tank_mount",
        )
        spec.add_sensor(
            name="liquid_mount_torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="liquid_tank_mount",
        )

        # The upstream home key contains only the 19 robot coordinates.  The
        # two zero slosh coordinates are appended in traversal order.
        for key in spec.keys:
            key.qpos = np.concatenate((np.asarray(key.qpos), np.zeros(2)))

        self.friction_config = friction_config
        if friction_config is not None:
            self.friction_field = SmoothRandomFrictionField.generate(friction_config)
            self.disabled_upstream_geometries = hide_upstream_world_geometries(spec)
            self.friction_tile_names = add_smooth_random_friction_ground(
                spec, self.friction_field
            )
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.base_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.tank_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "liquid_tank"
        )
        self.rigid_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "liquid_rigid_mass"
        )
        self.rigid_geom_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_GEOM, "liquid_rigid_visual"
        )
        self.modal_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "liquid_modal_mass"
        )
        self.mount_sid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, "liquid_tank_mount"
        )
        self.slosh_x_jid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "liquid_slosh_x"
        )
        self.slosh_y_jid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "liquid_slosh_y"
        )
        self.mount_force_sensor_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SENSOR, "liquid_mount_force"
        )
        self.mount_torque_sensor_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SENSOR, "liquid_mount_torque"
        )
        self._current_properties = initial
        self._last_fill_depth = initial.fill_depth
        if friction_config is not None:
            self._friction_contact_geom_ids = {
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "friction_contact_plane")
            }
            self._foot_geom_ids = {
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
                for name in ("FL", "FR", "RL", "RR")
            }

    # Share the contact law itself, not just its visualization, with the rigid
    # payload terrain. No friction or liquid state is exposed to Pinocchio.
    friction_at = MuJoCoRandomFrictionGo2Model.friction_at
    apply_spatial_contact_friction = MuJoCoRandomFrictionGo2Model.apply_spatial_contact_friction
    active_foot_contact_sliding_friction = MuJoCoRandomFrictionGo2Model.active_foot_contact_sliding_friction

    def update_with_q_pin(self, q_pin) -> None:
        q_pin = np.asarray(q_pin, dtype=float).reshape(ROBOT_NQ)
        px, py, pz, qx, qy, qz, qw = q_pin[:7]
        self.data.qpos[:ROBOT_NQ] = np.concatenate(
            ([px, py, pz, qw, qx, qy, qz], q_pin[7:])
        )
        self.data.qpos[ROBOT_NQ:] = 0.0
        mj.mj_forward(self.model, self.data)

    def update_pin_with_mujoco(self, go2: PinGo2Model) -> None:
        # Liquid coordinates are deliberately hidden from Pinocchio and MPC.
        mujoco_q = np.asarray(self.data.qpos[:ROBOT_NQ], dtype=float)
        mujoco_dq = np.asarray(self.data.qvel[:ROBOT_NV], dtype=float)
        qw, qx, qy, qz = mujoco_q[3:7]
        rotation = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        velocity_body = rotation.T @ mujoco_dq[0:3]
        q_pin = np.concatenate(
            (mujoco_q[0:3], [qx, qy, qz, qw], mujoco_q[7:ROBOT_NQ])
        )
        dq_pin = np.concatenate(
            (velocity_body, mujoco_dq[3:6], mujoco_dq[6:ROBOT_NV])
        )
        go2.update_model(q_pin, dq_pin)

    @property
    def liquid_properties(self) -> LiquidProperties:
        return self._current_properties

    def update_liquid_parameters(self, time_s: float) -> LiquidProperties:
        """Update quasi-static drain parameters before the next physics step."""

        fill_depth = self.liquid_config.fill_depth_at(
            time_s, self.liquid_protocol
        )
        if abs(fill_depth - self._last_fill_depth) < 1e-12:
            return self._current_properties
        properties = self.liquid_config.properties(fill_depth)

        rigid_half_sizes = np.array(
            [
                0.92 * self.liquid_config.length / 2.0,
                0.92 * self.liquid_config.width / 2.0,
                0.92 * fill_depth / 2.0,
            ]
        )
        self.model.body_mass[self.rigid_bid] = max(properties.rigid_mass, 1e-6)
        self.model.body_inertia[self.rigid_bid] = _box_inertia(
            max(properties.rigid_mass, 1e-6), rigid_half_sizes
        )
        self.model.body_mass[self.modal_bid] = max(properties.modal_mass, 1e-6)
        self.model.body_inertia[self.modal_bid] = _sphere_inertia(
            max(properties.modal_mass, 1e-6), self._modal_radius
        )
        liquid_center_z = (
            -self.liquid_config.height / 2.0 + properties.fill_depth / 2.0
        )
        self.model.body_pos[self.rigid_bid, 2] = liquid_center_z
        self.model.body_pos[self.modal_bid, 2] = liquid_center_z
        self.model.geom_size[self.rigid_geom_id, 2] = rigid_half_sizes[2]

        for joint_id in (self.slosh_x_jid, self.slosh_y_jid):
            dof_id = self.model.jnt_dofadr[joint_id]
            self.model.jnt_stiffness[joint_id] = properties.stiffness
            self.model.dof_damping[dof_id] = properties.damping

        # Recompute subtree mass and other constants after changing inertial
        # parameters.  mj_setConst temporarily installs qpos0 in ``data``, so
        # preserve the live trajectory explicitly; otherwise a drain update
        # would teleport the robot to the XML home keyframe.
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        act = self.data.act.copy()
        ctrl = self.data.ctrl.copy()
        simulation_time = float(self.data.time)
        mj.mj_setConst(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act[:] = act
        self.data.ctrl[:] = ctrl
        self.data.time = simulation_time
        mj.mj_forward(self.model, self.data)
        self._current_properties = properties
        self._last_fill_depth = fill_depth
        return properties

    def mount_wrench_world(self) -> LiquidWrenchSample:
        """Return the wrench exerted by the entire tank payload on the robot."""

        force_adr = self.model.sensor_adr[self.mount_force_sensor_id]
        torque_adr = self.model.sensor_adr[self.mount_torque_sensor_id]
        force_local = self.data.sensordata[force_adr : force_adr + 3]
        torque_local = self.data.sensordata[torque_adr : torque_adr + 3]
        rotation_world_from_site = self.data.site_xmat[self.mount_sid].reshape(3, 3)
        # Force/torque sensors report the parent-on-child wrench.  Negating it
        # yields the disturbance applied by the tank to the Go2 base.
        force_world = -(rotation_world_from_site @ force_local)
        torque_world = -(rotation_world_from_site @ torque_local)
        qpos_x = self.model.jnt_qposadr[self.slosh_x_jid]
        qpos_y = self.model.jnt_qposadr[self.slosh_y_jid]
        dof_x = self.model.jnt_dofadr[self.slosh_x_jid]
        dof_y = self.model.jnt_dofadr[self.slosh_y_jid]
        return LiquidWrenchSample(
            force=np.asarray(force_world, dtype=float).copy(),
            torque=np.asarray(torque_world, dtype=float).copy(),
            liquid_mass=self._current_properties.liquid_mass,
            fill_depth=self._current_properties.fill_depth,
            frequency_hz=self._current_properties.frequency_hz,
            modal_position=np.array(
                [self.data.qpos[qpos_x], self.data.qpos[qpos_y]], dtype=float
            ),
            modal_velocity=np.array(
                [self.data.qvel[dof_x], self.data.qvel[dof_y]], dtype=float
            ),
        )
