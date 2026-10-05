"""CLI validation and bounded physical checks; no control trials are run."""
from dataclasses import asdict
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_go2 import apply_disturbance_overrides, selected_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_configs_are_unchanged_for_every_paper_case():
    for seed in range(42, 52):
        for scene in ("drain", "friction", "combined"):
            for method in ("nominal", "ssi", "stgp", "t2s"):
                frozen, _ = selected_config(method, scene, seed)
                actual, settings, description = apply_disturbance_overrides(frozen)
                assert asdict(actual) == asdict(frozen)
                assert settings == dict(liquid_mass_scale=1., friction_min=.5,
                                        friction_max=.8, ground_friction=.8)
                assert description["requested_overrides"] == {}


@pytest.mark.parametrize("scene,options", [
    ("friction", {"payload_mass": 0}),
    ("friction", {"payload_mass": -1}),
    ("friction", {"payload_mass": float("inf")}),
    ("combined", {"payload_mass": 6}),
    ("friction", {"liquid_mass_scale": 2}),
    ("drain", {"liquid_mass_scale": 0}),
    ("combined", {"liquid_mass_scale": float("nan")}),
    ("combined", {"liquid_mass_scale": 1e308}),
    ("drain", {"friction_min": .2}),
    ("combined", {"friction_min": -.1}),
    ("combined", {"friction_min": .9}),
    ("combined", {"friction_max": float("inf")}),
    ("friction", {"ground_friction": .2}),
    ("drain", {"ground_friction": -.1}),
])
def test_invalid_or_inapplicable_options_fail_before_loading_simulator(scene, options):
    config, _ = selected_config("t2s", scene, 42)
    with pytest.raises(ValueError):
        apply_disturbance_overrides(config, **options)


def test_dry_run_reports_actual_liquid_mass_and_custom_ground_without_dependencies():
    # -S disables site-packages, ensuring this path uses the standard library.
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_go2.py"),
        "--scenario", "combined", "--liquid-mass-scale", "1.5",
        "--friction-min", "0.2", "--friction-max", "0.4", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["disturbances"]["payload"]["initial_total_mass_kg"] == pytest.approx(5.7)
    assert data["disturbances"]["payload"]["final_total_mass_kg"] == pytest.approx(2.7)
    assert data["disturbances"]["ground"]["sliding_min"] == .2
    assert data["disturbances"]["ground"]["sliding_max"] == .4
    assert data["mpc_friction_coefficient"] == .5


def test_cli_rejects_ignored_payload_option_with_actionable_error():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_go2.py"),
        "--scenario", "combined", "--payload-mass", "6", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "--liquid-mass-scale" in result.stderr


@pytest.fixture
def physics():
    # These checks need the documented native dependencies. They compile
    # small MuJoCo worlds but never run the MPC or integrate a trajectory.
    pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("convex_mpc")
    pytest.importorskip("l4casadi")
    pytest.importorskip("acados_template")
    experiment = pytest.importorskip("src.go2_icra_experiment")
    import src.go2_public_protocol as adapter
    import numpy as np
    return experiment, adapter, np


def test_frozen_and_rescaled_fields_preserve_torsion_rolling_and_seed_pattern(physics):
    _, adapter, np = physics
    manifest = json.loads((ROOT / "configs/go2/final.json").read_text())
    for seed in (42, 51):
        record = manifest["fields"][str(seed)]
        original = adapter.FrozenFrictionField(adapter.FrozenFrictionConfig(**record["world_config"]))
        from types import SimpleNamespace
        custom = adapter.FrozenFrictionField(adapter.frozen_world_config(SimpleNamespace(seed=seed), .1, .3))
        with np.load(ROOT / record["path"]) as data:
            frozen = np.array([original.vector_at(x) for x in data["x"]])
            actual = np.array([custom.vector_at(x) for x in data["x"]])
            np.testing.assert_array_equal(frozen, data["friction"])
        np.testing.assert_array_equal(actual[:, 1:], frozen[:, 1:])
        np.testing.assert_allclose(actual[:, 0], .1 + (frozen[:, 0] - .5) / .3 * .2,
                                   rtol=0, atol=2e-16)
        np.testing.assert_array_equal(custom.latent, original.latent)
    zero = adapter.FrozenFrictionField(adapter.frozen_world_config(SimpleNamespace(seed=42), 0., 0.))
    np.testing.assert_array_equal(zero.node_sliding, np.zeros_like(zero.node_sliding))


def test_rigid_payload_option_changes_actual_plant_mass(physics):
    experiment, adapter, np = physics
    base, _ = selected_config("nominal", "friction", 42)
    changed, settings, _ = apply_disturbance_overrides(base, payload_mass=6.)
    with adapter.frozen_protocol():
        original = experiment.make_world(base)
    with adapter.frozen_protocol(**settings):
        actual = experiment.make_world(changed)
    assert actual.rigid_payload_config.mass == 6.
    assert np.sum(actual.model.body_mass) - np.sum(original.model.body_mass) == pytest.approx(2.)


def test_liquid_scale_and_fixed_ground_change_real_physics(physics):
    experiment, adapter, np = physics
    base, _ = selected_config("nominal", "drain", 42)
    changed, settings, _ = apply_disturbance_overrides(base, liquid_mass_scale=2., ground_friction=.25)
    with adapter.frozen_protocol():
        original = experiment.make_world(base)
    with adapter.frozen_protocol(**settings):
        actual = experiment.make_world(changed)
    # Evaluate the production trajectory field expression against the real
    # worlds, without starting its controller process or integrating a trial.
    tree = ast.parse(inspect.getsource(experiment.run_icra_experiment))
    expressions = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.keyword) and node.arg == "friction_at_body"]
    assert len(expressions) == 1
    recorded_friction = compile(ast.Expression(expressions[0]), "<trajectory friction field>", "eval")
    assert eval(recorded_friction, {"simulation": original, "config": base}) == .8
    assert eval(recorded_friction, {"simulation": actual, "config": changed}) == .25
    assert actual.liquid_config.liquid_density == 2000.
    assert actual.liquid_config.container_mass == .6
    assert actual.liquid_properties.liquid_mass == pytest.approx(6.8)
    assert np.sum(actual.model.body_mass) - np.sum(original.model.body_mass) == pytest.approx(3.4)
    for name in ("floor", "FL", "FR", "RL", "RR"):
        geom = experiment.mj.mj_name2id(actual.model, experiment.mj.mjtObj.mjOBJ_GEOM, name)
        assert actual.model.geom_friction[geom, 0] == .25
        np.testing.assert_array_equal(actual.model.geom_friction[geom, 1:], original.model.geom_friction[geom, 1:])
    # Geometry-to-contact propagation is checked with forward dynamics only.
    actual.data.qpos[2] = .2
    experiment.mj.mj_forward(actual.model, actual.data)
    toe_ids = {experiment.mj.mj_name2id(actual.model, experiment.mj.mjtObj.mjOBJ_GEOM, n)
               for n in ("FL", "FR", "RL", "RR")}
    floor = experiment.mj.mj_name2id(actual.model, experiment.mj.mjtObj.mjOBJ_GEOM, "floor")
    contacts = [c for c in actual.data.contact if floor in (c.geom1, c.geom2)
                and ({int(c.geom1), int(c.geom2)} & toe_ids)]
    assert contacts
    assert all(c.friction[0] == .25 and c.friction[1] == .25 for c in contacts)
    original.update_liquid_parameters(45.)
    actual.update_liquid_parameters(45.)
    assert actual.liquid_properties.liquid_mass == pytest.approx(2.8)
    assert np.sum(actual.model.body_mass) - np.sum(original.model.body_mass) == pytest.approx(1.4)


def test_variable_ground_override_reaches_active_contact_coefficients(physics):
    experiment, adapter, np = physics
    base, _ = selected_config("nominal", "friction", 42)
    config, settings, _ = apply_disturbance_overrides(base, friction_min=.2, friction_max=.2)
    with adapter.frozen_protocol(**settings):
        world = experiment.make_world(config)
    assert world.friction_config.effective_sliding_min == .2
    assert world.friction_config.effective_sliding_max == .2
    world.data.qpos[2] = .2
    experiment.mj.mj_forward(world.model, world.data)
    world.apply_spatial_contact_friction()
    coefficients = world.active_foot_contact_sliding_friction()
    assert coefficients.size > 0
    np.testing.assert_array_equal(coefficients, np.full_like(coefficients, .2))
