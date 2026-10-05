"""Portable adapters for the frozen Go2 cone-0.5 experiment.

Sliding nodes come from released fields. The original generator supplies
terrain geometry and torsional/rolling friction.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import math
import numpy as np
from src.go2_friction import (
    SmoothRandomFrictionConfig, SmoothRandomFrictionField, friction_rgba,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/go2/final.json"
LEGACY_GENERATE = SmoothRandomFrictionField.generate


@dataclass(frozen=True)
class FrozenFrictionConfig(SmoothRandomFrictionConfig):
    sliding_field_mode: str = "approved_gaussian_copula_linear_v1"
    sliding_node_spacing: float = .2
    sliding_gaussian_correlation_length: float = 1.
    sliding_underlying_mean: float = .35
    sliding_underlying_std: float = .10
    sliding_correlation_jitter: float = 1e-10
    non_sliding_policy: str = "legacy_alternating_spatial_values_unchanged"
    legacy_region_parameters_apply_to: str = "torsional_and_rolling_only"
    effective_sliding_min: float = .50
    effective_sliding_max: float = .80
    effective_sliding_bounds_override: bool = True


class FrozenFrictionField:
    def __init__(self, config):
        self.config = config
        seed = config.seed - 300000
        record = json.loads(CONFIG_PATH.read_text())["fields"][str(seed)]
        baseline = record["world_config"]
        normalized = asdict(config)
        for key in ("effective_sliding_min", "effective_sliding_max", "sliding_field_mode"):
            normalized[key] = baseline[key]
        if normalized != baseline:
            raise ValueError("Friction settings differ from the frozen field beyond the requested sliding bounds")
        lower, upper = config.effective_sliding_min, config.effective_sliding_max
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper):
            raise ValueError("Sliding bounds must be finite and satisfy 0 <= min <= max")
        legacy_names = SmoothRandomFrictionConfig.__dataclass_fields__
        self.legacy = LEGACY_GENERATE(SmoothRandomFrictionConfig(
            **{k: v for k, v in asdict(config).items() if k in legacy_names}))
        with np.load(ROOT / record["path"], allow_pickle=False) as data:
            self.nodes = data["x_nodes"].copy()
            self.latent = data["latent"].copy()
            self.node_sliding = data["sliding_nodes"].copy()
            recorded_x = data["x"].copy()
            recorded_vectors = data["friction"].copy()
        self.centers = self.legacy.centers
        self.tile_length = self.legacy.tile_length
        self.region_index = self.legacy.region_index
        self.sliding = np.interp(self.centers, self.nodes, self.node_sliding)
        # Check all three coefficients before compiling a plant; physics
        # interpolation remains on the original 0.2 m grid.
        np.testing.assert_array_equal(
            np.array([self.vector_at(x) for x in recorded_x]), recorded_vectors)
        original_bounds = (baseline["effective_sliding_min"], baseline["effective_sliding_max"])
        if (lower, upper) != original_bounds:
            # Preserve the seed's spatial pattern. This is an affine map of
            # the saved realization, not a new truncated-Gaussian sample.
            low0, high0 = original_bounds
            self.node_sliding = lower + (self.node_sliding - low0) / (high0 - low0) * (upper - lower)
            self.sliding = np.interp(self.centers, self.nodes, self.node_sliding)

    def sliding_at(self, x_position, y_position=0.):
        return float(np.interp(x_position, self.nodes, self.node_sliding))

    def vector_at(self, x_position, y_position=0.):
        vector = self.legacy.vector_at(x_position, y_position)
        vector[0] = self.sliding_at(x_position, y_position)
        return vector

    def rgba_at_index(self, index):
        return friction_rgba(self.sliding[index],
                             SimpleNamespace(sliding_min=min(.05, self.config.effective_sliding_min),
                                             sliding_max=max(.80, self.config.effective_sliding_max)))

    def tile_index(self, x_position):
        return self.legacy.tile_index(x_position)


def frozen_world_config(paper_config, friction_min=.5, friction_max=.8):
    record = json.loads(CONFIG_PATH.read_text())["fields"][str(paper_config.seed)]
    config = FrozenFrictionConfig(**record["world_config"])
    if (friction_min, friction_max) != (.5, .8):
        config = replace(config, effective_sliding_min=friction_min,
                         effective_sliding_max=friction_max,
                         sliding_field_mode="affine_rescaled_frozen_sliding_v1")
    return config


def cone050_controller_process(connection, config, cpu_core=None, training_core=None):
    """Spawn-safe adapter: change the MPC cone before model construction."""
    import src.go2_icra_controller as controller
    import src.go2_paper_acados as acados
    original = acados.PaperAcadosMPC

    def build(**kwargs):
        kwargs["name_suffix"] += "_public_cone050_v1"
        return original(**kwargs)

    with patch.object(acados, "PAPER_FRICTION_COEFFICIENT", .5), \
         patch.object(controller, "PaperAcadosMPC", side_effect=build):
        controller.controller_process(connection, config, cpu_core, training_core)


@contextmanager
def frozen_protocol(*, liquid_mass_scale=1., friction_min=.5, friction_max=.8,
                    ground_friction=.8):
    """Keep the paper world by default; overrides affect the hidden plant."""
    import src.go2_icra_experiment as experiment
    if not all(math.isfinite(v) for v in (liquid_mass_scale, friction_min, friction_max, ground_friction)):
        raise ValueError("Disturbance values must be finite")
    if liquid_mass_scale <= 0 or not math.isfinite(1000. * liquid_mass_scale):
        raise ValueError("Liquid mass scale must yield a finite positive density")
    if not 0 <= friction_min <= friction_max or ground_friction < 0:
        raise ValueError("Friction must be nonnegative with min <= max")
    original_world = experiment.make_world
    original_tank_config = experiment.LiquidTankConfig

    def make_tank(**kwargs):
        tank = original_tank_config(**kwargs)
        return replace(tank, liquid_density=tank.liquid_density * liquid_mass_scale)

    def make_world(config):
        if config.liquid and liquid_mass_scale != 1.:
            with patch.object(experiment, "LiquidTankConfig", side_effect=make_tank):
                simulation = original_world(config)
        else:
            simulation = original_world(config)
        if not config.variable_friction and ground_friction != .8:
            # Toe priority is one: changing only the floor does not change
            # effective contact friction. Set sliding on both participants.
            for name in ("floor", "FL", "FR", "RL", "RR"):
                geom = experiment.mj.mj_name2id(simulation.model, experiment.mj.mjtObj.mjOBJ_GEOM, name)
                if geom < 0:
                    raise RuntimeError(f"Missing MuJoCo contact geom: {name}")
                simulation.model.geom_friction[geom, 0] = ground_friction
        return simulation

    def world_config(paper_config):
        return frozen_world_config(paper_config, friction_min, friction_max)

    def generate(config):
        if isinstance(config, FrozenFrictionConfig):
            return FrozenFrictionField(config)
        return LEGACY_GENERATE(config)

    with patch.object(experiment, "_alternating_friction_config", world_config), \
         patch.object(SmoothRandomFrictionField, "generate", side_effect=generate), \
         patch.object(experiment, "controller_process", cone050_controller_process), \
         patch.object(experiment, "make_world", make_world):
        yield
