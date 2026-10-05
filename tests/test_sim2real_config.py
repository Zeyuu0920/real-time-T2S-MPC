"""Profile validation, real worker handoff, and physical parameter consumption."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_quadrotor
from src.deadline_control import ComputationAwareControlRelease
from src.motor_actuator import add_motor_actuator_arguments, make_motor_actuator, FirstOrderMotorActuator
from src.sim2real_config import apply_quadrotor_config, load_quadrotor_config
from src.state_measurement import DelayedNoisyStateEstimator, state_noise_standard_deviations

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_quadrotor.py"
EXAMPLE = ROOT / "configs/sim2real/quadrotor_example.json"
CONFIG = json.loads((ROOT / "configs/quadrotor.json").read_text())


def invoke(*args):
    return subprocess.run([sys.executable, "-S", str(RUNNER), *map(str, args), "--dry-run"],
                          capture_output=True, text=True, check=False)


def run_list(*args):
    result = invoke(*args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["runs"]


def write_profile(tmp_path, **sections):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(schema_version=1, system="quadrotor", **sections)))
    return path


def test_all_120_default_paper_argument_vectors_are_unchanged():
    runs = run_list("--method", "all", "--scenario", "all", "--seeds", *range(42, 52))
    assert len(runs) == 120
    argv = []
    for run in runs:
        flags = run["args"][:]
        flags[flags.index("--output-dir") + 1] = "<output>"
        argv.append(flags)
        assert run["sim2real"]["input"] is None
        assert run["sim2real"]["measurement_delay_ms"] == 20
    # Captured from all selected runs before adding the profile interface.
    assert hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest() == (
        "3388edc320045c550cf3c4f0c826f0558811fb9e583c90508a4f5e51fdc607b7"
    )


def test_partial_profile_inherits_actual_protocol_values_and_records_input(tmp_path):
    path = write_profile(tmp_path, actuator={"time_constant_s": 0.04})
    profile = load_quadrotor_config(path)
    flags = CONFIG["common_args"][:]
    flags[flags.index("--position-noise-std") + 1] = "0.123"
    original = flags[:]
    updated, metadata = apply_quadrotor_config(flags, profile)
    assert flags == original
    assert metadata["input"] == {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert metadata["requested"] == {"observation": {}, "actuator": {"time_constant_s": 0.04}}
    assert metadata["effective"]["observation"]["position_noise_std_m"] == 0.123
    assert metadata["effective"]["actuator"]["time_constant_s"] == 0.04
    assert updated[updated.index("--motor-time-constant") + 1] == "0.04"
    changed = {index for index, (before, after) in enumerate(zip(flags, updated)) if before != after}
    assert changed == {flags.index("--motor-time-constant") + 1}


def test_custom_protocol_ideal_parameters_are_not_reported_as_effective(tmp_path):
    flags = CONFIG["common_args"][:]
    flags[flags.index("--actuator-model") + 1] = "ideal"
    flags[flags.index("--motor-time-constant") + 1] = "0"
    updated, metadata = apply_quadrotor_config(flags)
    assert updated == flags
    assert metadata["effective"]["actuator"] is None
    assert metadata["motor_time_constant_scope"] is None
    assert metadata["ignored_actuator_parameters"]["time_constant_s"] == 0
    with pytest.raises(ValueError, match="first_order"):
        apply_quadrotor_config(flags, load_quadrotor_config(write_profile(tmp_path)))


@pytest.mark.parametrize("raw", [
    '[]', '{"schema_version":1,"schema_version":1,"system":"quadrotor"}',
    '{"schema_version":1,"system":"quadrotor","actuator":{"noise_std":0,"noise_std":1}}',
    '{"schema_version":1,"system":"quadrotor","actuator":{"noise_std":NaN}}',
    '{"schema_version":1,"system":"quadrotor","actuator":{"noise_std":Infinity}}',
    '{"schema_version":1,"system":"quadrotor","actuator":{"noise_std":-Infinity}}',
    '{"schema_version":1,"system":"quadrotor","actuator":{"noise_std":1e999}}',
])
def test_ambiguous_or_nonfinite_json_is_rejected_before_runtime(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    result = invoke("--sim2real-config", path)
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize("patch", [
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 2},
    {"system": "go2"}, {"unexpected": 0}, {"observation": None}, {"actuator": []},
    {"observation": {"delay_steps": True}}, {"observation": {"delay_steps": 1.5}},
    {"observation": {"delay_steps": -1}}, {"observation": {"position_noise_std_m": "0.1"}},
    {"observation": {"position_noise_std_m": False}}, {"observation": {"body_rate_noise_std_radps": -1}},
    {"actuator": {"time_constant_s": 0}}, {"actuator": {"gain_range": 1}},
    {"actuator": {"noise_std": None}}, {"actuator": {"noise_std": float("inf")}},
    {"actuator": {"noise_std": 10**400}}, {"actuator": {"model": "ideal"}},
    {"observation": {"measurement_delay_steps": 1}},
])
def test_schema_type_range_and_unknown_fields_are_rejected(tmp_path, patch):
    data = dict(schema_version=1, system="quadrotor")
    data.update(patch)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_quadrotor_config(path)


def test_zero_noise_delay_and_correlation_are_valid(tmp_path):
    path = write_profile(tmp_path, observation={"delay_steps": 0, "noise_correlation_time_s": 0,
                                               "position_noise_std_m": 0},
                         actuator={"gain_range": 0, "noise_std": 0, "noise_correlation_time_s": 0})
    metadata = run_list("--sim2real-config", path)[0]["sim2real"]
    assert metadata["measurement_delay_ms"] == 0
    assert metadata["effective"]["actuator"]["time_constant_s"] == 0.025


def test_empty_partial_profile_preserves_arguments(tmp_path):
    path = write_profile(tmp_path)
    assert run_list("--sim2real-config", path)[0]["args"] == run_list()[0]["args"]


def test_dry_run_is_read_only_with_stdlib_only_and_custom_seed(tmp_path):
    path = write_profile(tmp_path, observation={"delay_steps": 3})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    run = run_list("--sim2real-config", path, "--seed", 2026, "--output-dir", tmp_path / "results")[0]
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert not (tmp_path / "results").exists()
    assert run["seed"] == 2026
    assert run["sim2real"]["measurement_delay_ms"] == 60


def test_first_dry_run_creates_no_bytecode_or_output_in_fresh_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    for relative in ("scripts/run_quadrotor.py", "src/sim2real_config.py"):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    before = sorted(str(path.relative_to(checkout)) for path in checkout.rglob("*"))
    result = subprocess.run([sys.executable, "-S", str(checkout / "scripts/run_quadrotor.py"),
        "--config", str(ROOT / "configs/quadrotor.json"), "--sim2real-config", str(EXAMPLE), "--dry-run"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert sorted(str(path.relative_to(checkout)) for path in checkout.rglob("*")) == before


def test_real_worker_dry_run_all_methods_and_reject_changed_input(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    path.write_bytes(EXAMPLE.read_bytes())
    commands = []
    real_run = subprocess.run
    monkeypatch.setattr(run_quadrotor, "validate_runtime", lambda *_: None)
    monkeypatch.setattr(run_quadrotor.subprocess, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--method", "all", "--sim2real-config", str(path),
                                     "--output-dir", str(tmp_path / "runs"), "--no-affinity"])
    run_quadrotor.main()
    assert len(commands) == 4
    expected = load_quadrotor_config(path)
    for command in commands:
        assert command[command.index("--sim2real-config") + 1] == str(path.resolve())
        assert command[command.index("--_sim2real-sha256") + 1] == expected["sha256"]
        result = real_run([command[0], "-S", *command[1:], "--dry-run"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        run = json.loads(result.stdout)["runs"][0]
        assert run["controller"] == command[command.index("--controller") + 1]
        assert run["sim2real"]["requested"] == expected["requested"]
        assert run["sim2real"]["effective"] == expected["requested"]
    assert not (tmp_path / "runs").exists()
    path.write_text(path.read_text() + "\n")
    result = real_run([commands[0][0], "-S", *commands[0][1:], "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "SHA-256 mismatch" in result.stderr
    assert not result.stdout


def _function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("method", ["t2s", "nominal", "ssi", "stgp"])
def test_actual_backend_parser_and_component_constructors_consume_profile(method, monkeypatch):
    run = run_list("--method", method, "--sim2real-config", EXAMPLE, "--no-affinity")[0]
    path = ROOT / (run["module"].replace(".", "/") + ".py")
    namespace = dict(argparse=argparse, Path=Path, os=os, np=np, CONTROL_FREQUENCY=50,
                     PHYSICS_FREQUENCY=500, PHYSICS_DT=0.002, REPLAY_MAX=100,
                     SSI_THRUST_FEATURE_MODES=("lagged_physical", "normalized_command"),
                     ComputationAwareControlRelease=ComputationAwareControlRelease,
                     add_motor_actuator_arguments=add_motor_actuator_arguments)
    namespace["add_physical_wind_3d_arguments"] = _function(
        ROOT / "src/physical_wind_3d.py", "add_physical_wind_3d_arguments", {})
    parse_args = _function(path, "parse_args", namespace)
    monkeypatch.setattr(sys, "argv", [str(path), *run["args"]])
    args = parse_args(method)
    # Execute the production construction statements, avoiding PyBullet/acados
    # imports and the control loop while retaining real estimator/motor classes.
    namespace.update(args=args, seed=42, dt=1 / args.control_frequency, controller=method,
                     RIGID_BODY_STATE_DIM=12, state_noise_standard_deviations=state_noise_standard_deviations,
                     DelayedNoisyStateEstimator=DelayedNoisyStateEstimator, make_motor_actuator=make_motor_actuator,
                     env=SimpleNamespace(KF=1.0, physical_action_bounds=(np.zeros(4), np.full(4, 4.0))))
    run_once = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_once")
    names = {"measurement_std", "estimator_delay_steps", "state_estimator", "actuator",
             "controller_motor_time_constant", "controller_state_dim"}
    statements = [node for node in run_once.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    assert len(statements) == len(names)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    estimator = namespace["state_estimator"]
    actuator = namespace["actuator"]
    assert isinstance(estimator, DelayedNoisyStateEstimator)
    assert isinstance(actuator, FirstOrderMotorActuator)
    assert estimator.delay_steps == 2
    assert estimator.correlation_time == 0.15
    np.testing.assert_allclose(estimator.standard_deviations,
                               [0.02, 0.04, 0.02, 0.04, 0.02, 0.04, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04])
    # A delayed packet retains the exact noisy sample produced two steps earlier.
    undelayed = DelayedNoisyStateEstimator(seed=200042, dt=estimator.dt,
        standard_deviations=estimator.standard_deviations, correlation_time=0.15, delay_steps=0)
    estimator.reset_packet(np.zeros(12))
    undelayed.reset_packet(np.zeros(12))
    packets = [undelayed.observe_packet(np.full(12, step * 0.1)) for step in range(1, 5)]
    delayed = [estimator.observe_packet(np.full(12, step * 0.1)) for step in range(1, 5)]
    np.testing.assert_array_equal(delayed[2].state, packets[0].state)
    assert delayed[2].arrival_step - delayed[2].source_step == 2
    assert actuator.time_constant == namespace["controller_motor_time_constant"] == 0.04
    assert namespace["controller_state_dim"] == 16
    assert (actuator.gain_range, actuator.noise_std, actuator.noise_correlation_time) == (0.08, 0.02, 0.08)
    actuator.reset(np.full(4, 0.25))
    sample = actuator.step(np.ones(4))
    np.testing.assert_allclose(sample.lagged_rpm, 1 - 0.5 * np.exp(-0.002 / 0.04))
    assert np.all((sample.motor_gain >= 0.92) & (sample.motor_gain <= 1.08))
    assert np.any(sample.relative_noise != 0)


def test_worker_persists_same_profile_metadata_before_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--_worker", "--sim2real-config", str(EXAMPLE),
                                     "--output-dir", str(tmp_path), "--no-affinity"])
    args = run_quadrotor.parse_args()
    runs = list(run_quadrotor.selected_runs(args, CONFIG))
    calls = []
    monkeypatch.setattr(run_quadrotor.importlib, "import_module",
                        lambda _: SimpleNamespace(main=lambda method: calls.append((method, sys.argv[:]))))
    run_quadrotor.worker(args, CONFIG, runs)
    saved = json.loads((Path(runs[0]["output_dir"]) / "run_protocol.json").read_text())
    assert saved["sim2real"] == runs[0]["sim2real"]
    assert calls == [("t2s", [str(RUNNER), *runs[0]["args"]])]
