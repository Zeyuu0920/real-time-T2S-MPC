"""Random-strength alternating red/blue friction terrain for Go2."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from convex_mpc.mujoco_model import XML_PATH, MuJoCo_GO2_Model


def friction_rgba(
    sliding: float, config: "SmoothRandomFrictionConfig"
) -> np.ndarray:
    """Map the paper-matched sliding-friction range from blue to red."""

    normalized = (float(sliding) - config.sliding_min) / (
        config.sliding_max - config.sliding_min
    )
    hue = 0.63 * (1.0 - float(np.clip(normalized, 0.0, 1.0)))
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
    return np.array([red, green, blue, 1.0], dtype=float)


@dataclass(frozen=True)
class SmoothRandomFrictionConfig:
    """A discrete alternating terrain with randomized red/blue strengths.

    The class name is retained for compatibility with existing Route-B
    scripts.  Physics are intentionally discontinuous at region boundaries,
    matching the switched red/blue layout in Zhou et al. while randomizing
    each region near the two paper-reported friction endpoints.
    """

    seed: int = 42
    x_min: float = -3.0
    x_max: float = 45.0
    tile_length: float = 0.02
    half_width: float = 3.0
    half_thickness: float = 0.025
    sliding_min: float = 0.05
    sliding_max: float = 0.50
    red_sliding_min: float = 0.45
    red_sliding_max: float = 0.50
    blue_sliding_min: float = 0.05
    blue_sliding_max: float = 0.10
    red_length_min: float = 3.80
    red_length_max: float = 4.20
    blue_length_min: float = 0.04
    blue_length_max: float = 0.06
    warmup_end_x: float = 0.50

    def validate(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if min(
            self.tile_length,
            self.half_width,
            self.half_thickness,
            self.red_length_min,
            self.blue_length_min,
        ) <= 0.0:
            raise ValueError("friction terrain dimensions must be positive")
        if not 0.0 < self.sliding_min < self.sliding_max:
            raise ValueError("friction bounds must satisfy 0 < min < max")
        if not (
            self.sliding_min
            <= self.blue_sliding_min
            <= self.blue_sliding_max
            < self.red_sliding_min
            <= self.red_sliding_max
            <= self.sliding_max
        ):
            raise ValueError(
                "blue/red friction ranges must be ordered inside global bounds"
            )
        if self.red_length_max < self.red_length_min:
            raise ValueError("red friction length bounds are invalid")
        if self.blue_length_max < self.blue_length_min:
            raise ValueError("blue friction length bounds are invalid")
        if not self.x_min <= self.warmup_end_x < self.x_max:
            raise ValueError("warmup_end_x must lie inside the terrain")


@dataclass(frozen=True)
class RigidPayloadConfig:
    """Unknown rigid payload mounted on the Go2 trunk in the MuJoCo plant.

    The default box dimensions reproduce the inertia reported for the 4 kg
    MuJoCo payload in Zhou et al. (arXiv:2510.15626) to rounding precision:
    ``[0.00234, 0.00304, 0.00414] kg m^2``.  The centroidal controller retains
    the dry-Go2 mass and inertia.
    """

    mass: float = 4.0
    mount_position: tuple[float, float, float] = (0.0, 0.0, 0.13)
    half_size: tuple[float, float, float] = (
        0.0426028168,
        0.03591656999,
        0.02156385865,
    )

    def validate(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("rigid payload mass must be positive")
        if min(self.half_size) <= 0.0:
            raise ValueError("rigid payload half sizes must be positive")


def add_rigid_payload_to_spec(
    spec: mj.MjSpec,
    config: RigidPayloadConfig,
) -> None:
    """Attach a collision-free rigid payload to ``base_link``."""

    config.validate()
    base = spec.body("base_link")
    payload = base.add_body(
        name="payload_fixed",
        pos=list(config.mount_position),
    )
    payload.add_geom(
        name="payload_fixed_visual",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=list(config.half_size),
        mass=config.mass,
        contype=0,
        conaffinity=0,
        group=2,
        rgba=[0.92, 0.64, 0.12, 0.92],
    )


@dataclass(frozen=True)
class SmoothRandomFrictionField:
    """Tile-sampled piecewise-constant alternating friction field."""

    config: SmoothRandomFrictionConfig
    centers: np.ndarray
    sliding: np.ndarray
    region_index: np.ndarray
    tile_length: float

    @classmethod
    def generate(
        cls, config: SmoothRandomFrictionConfig
    ) -> "SmoothRandomFrictionField":
        config.validate()
        count = int(np.ceil((config.x_max - config.x_min) / config.tile_length))
        tile_length = config.tile_length
        centers = config.x_min + (np.arange(count) + 0.5) * tile_length

        rng = np.random.default_rng(config.seed)
        sliding = np.full(count, config.sliding_max, dtype=float)
        region_index = np.full(count, -1, dtype=int)
        cursor = max(config.x_min, config.warmup_end_x)
        patch_index = 0

        while cursor < config.x_max:
            is_red = patch_index % 2 == 0
            if is_red:
                patch_value = rng.uniform(
                    config.red_sliding_min, config.red_sliding_max
                )
            else:
                patch_value = rng.uniform(
                    config.blue_sliding_min, config.blue_sliding_max
                )
            if is_red:
                patch_length = rng.uniform(
                    config.red_length_min, config.red_length_max
                )
            else:
                patch_length = rng.uniform(
                    config.blue_length_min, config.blue_length_max
                )
            patch_end = cursor + patch_length
            patch_mask = (centers >= cursor) & (centers < patch_end)
            sliding[patch_mask] = patch_value
            region_index[patch_mask] = patch_index
            cursor = patch_end
            patch_index += 1
        return cls(
            config=config,
            centers=centers,
            sliding=sliding,
            region_index=region_index,
            tile_length=tile_length,
        )

    def tile_index(self, x_position: float) -> int:
        index = int(
            np.floor((float(x_position) - self.config.x_min) / self.tile_length)
        )
        return int(np.clip(index, 0, self.sliding.size - 1))

    def sliding_at(self, x_position: float, y_position: float = 0.0) -> float:
        del y_position
        return float(self.sliding[self.tile_index(x_position)])

    def vector_at(
        self, x_position: float, y_position: float = 0.0
    ) -> np.ndarray:
        sliding = self.sliding_at(x_position, y_position)
        # Preserve the paper's two endpoint vectors exactly:
        # [0.05, 0.05, 0.001] and [0.5, 0.5, 0.01].
        return np.array([sliding, sliding, 0.02 * sliding], dtype=float)

    def rgba_at_index(self, index: int) -> np.ndarray:
        return friction_rgba(self.sliding[index], self.config)


def hide_upstream_world_geometries(spec: mj.MjSpec) -> int:
    """Disable the imported plane and demo obstacles in physics and rendering."""

    disabled = 0
    for geometry in spec.worldbody.geoms:
        geometry.contype = 0
        geometry.conaffinity = 0
        geometry.group = 5
        geometry.rgba = [0.0, 0.0, 0.0, 0.0]
        geometry.pos[2] = -100.0
        disabled += 1
    return disabled


def add_smooth_random_friction_ground(
    spec: mj.MjSpec,
    field: SmoothRandomFrictionField,
    *,
    name_prefix: str = "",
) -> list[str]:
    """Add one seamless contact plane and one textured field visualization."""

    contact_name = f"{name_prefix}friction_contact_plane"
    reference_friction = field.vector_at(0.0)
    spec.worldbody.add_geom(
        name=contact_name,
        type=mj.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 0.1],
        friction=reference_friction.tolist(),
        priority=2,
        condim=6,
        contype=1,
        conaffinity=1,
        group=5,
        rgba=[0.0, 0.0, 0.0, 0.0],
    )
    return [
        add_friction_texture_ground(
            spec,
            field,
            name_prefix=name_prefix,
        )
    ]


def add_friction_texture_ground(
    spec: mj.MjSpec,
    field: SmoothRandomFrictionField,
    *,
    name_prefix: str = "",
) -> str:
    """Render the complete spatial field with one textured, non-contact box."""

    texture_name = f"{name_prefix}friction_gradient_texture"
    material_name = f"{name_prefix}friction_gradient_material"
    ground_name = f"{name_prefix}friction_gradient_ground"
    rgb = np.stack(
        [field.rgba_at_index(index)[:3] for index in range(field.sliding.size)]
    )
    texture_data = np.repeat(
        np.rint(255.0 * rgb)[None, :, :].astype(np.uint8),
        repeats=4,
        axis=0,
    )
    texture = spec.add_texture(
        name=texture_name,
        type=mj.mjtTexture.mjTEXTURE_2D,
        width=field.sliding.size,
        height=texture_data.shape[0],
        nchannel=3,
    )
    texture.data = texture_data.tobytes()
    material = spec.add_material(
        name=material_name,
        texrepeat=[1.0, 1.0],
        texuniform=False,
        reflectance=0.05,
        specular=0.05,
        shininess=0.1,
    )
    material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = texture_name
    route_length = field.tile_length * field.sliding.size
    spec.worldbody.add_geom(
        name=ground_name,
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[
            field.config.x_min + 0.5 * route_length,
            0.0,
            -field.config.half_thickness,
        ],
        size=[
            0.5 * route_length,
            field.config.half_width,
            field.config.half_thickness,
        ],
        material=material_name,
        contype=0,
        conaffinity=0,
        rgba=[1.0, 1.0, 1.0, 1.0],
    )
    return ground_name


class MuJoCoRandomFrictionGo2Model(MuJoCo_GO2_Model):
    """Full-order Go2 plant on an unobserved spatial random-friction field."""

    def __init__(
        self,
        config: SmoothRandomFrictionConfig,
        *,
        rigid_payload: RigidPayloadConfig | None = None,
    ) -> None:
        self.friction_config = config
        self.friction_field = SmoothRandomFrictionField.generate(config)
        self.rigid_payload_config = rigid_payload
        spec = mj.MjSpec.from_file(str(XML_PATH))
        if rigid_payload is not None:
            add_rigid_payload_to_spec(spec, rigid_payload)
        self.disabled_upstream_geometries = hide_upstream_world_geometries(spec)
        self.friction_tile_names = add_smooth_random_friction_ground(
            spec, self.friction_field
        )
        self.friction_contact_geom_names = ["friction_contact_plane"]
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.base_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "base_link"
        )
        self._friction_contact_geom_ids = {
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in self.friction_contact_geom_names
        }
        self._foot_geom_ids = {
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in ("FL", "FR", "RL", "RR")
        }

    def friction_at(
        self, x_position: float, y_position: float = 0.0
    ) -> np.ndarray:
        """Return plant friction at a world-frame ground position."""

        return self.friction_field.vector_at(x_position, y_position)

    def active_foot_contact_sliding_friction(self) -> np.ndarray:
        """Return effective MuJoCo sliding friction for active foot contacts."""

        values = []
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            if (
                pair & self._foot_geom_ids
                and pair & self._friction_contact_geom_ids
            ):
                values.append(float(contact.friction[0]))
        return np.asarray(values, dtype=float)

    def apply_spatial_contact_friction(self) -> None:
        """Apply the hidden region value at every current foot contact point.

        A single plane provides seamless collision geometry while a texture
        renders the material regions.  Overriding ``mjContact.friction`` after
        ``mj_step1`` makes the red/blue boundary affect the actual contact law
        without introducing collision seams.
        """

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (
                pair & self._foot_geom_ids
                and pair & self._friction_contact_geom_ids
            ):
                continue
            sliding, torsional, rolling = self.friction_at(
                contact.pos[0], contact.pos[1]
            )
            contact.friction[:] = [
                sliding,
                sliding,
                torsional,
                rolling,
                rolling,
            ]

    def apply_continuous_contact_friction(self) -> None:
        """Backward-compatible alias for the spatial contact update."""

        self.apply_spatial_contact_friction()
