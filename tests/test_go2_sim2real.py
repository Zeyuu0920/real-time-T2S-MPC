"""Bounded checks for the independent Go2 sensor/torque interface; no MPC trials."""
import ast
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_go2_sim2real as runner
from src.go2_sim2real_config import RealismOptions, load_profile, timing_metadata

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/sim2real/go2_example.json"


def profile_file(tmp_path, **sections):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema_version": 1, "system": "go2", **sections}))
    return path


def invoke(*args):
    return subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_go2_sim2real.py"), *args],
                          capture_output=True, text=True)


def test_example_and_sparse_profile_preserve_original_research_defaults(tmp_path):
    example, metadata = load_profile(EXAMPLE)
    sparse, _ = load_profile(profile_file(tmp_path))
    assert example == sparse == RealismOptions()
    assert metadata["effective"]["actuator"] == {"time_constant_s": .005, "gain_range": .03}
    assert timing_metadata(example)["observation_delay_ms"] == 20


@pytest.mark.parametrize("section,key,value", [
    ("observation", "delay_steps", -1), ("observation", "delay_steps", True),
    ("observation", "delay_steps", 1.0), ("observation", "delay_steps", "1"),
    ("observation", "position_noise_std_m", -1), ("observation", "position_noise_std_m", True),
    ("observation", "velocity_noise_std_mps", "0.2"),
    ("observation", "noise_correlation_time_s", 0),
    ("actuator", "time_constant_s", -1), ("actuator", "gain_range", 1),
    ("actuator", "gain_range", -0.1), ("actuator", "noise_std", .01),
])
def test_profile_rejects_invalid_or_unimplemented_parameters(tmp_path, section, key, value):
    with pytest.raises(ValueError):
        load_profile(profile_file(tmp_path, **{section: {key: value}}))


@pytest.mark.parametrize("text", [
    '{"schema_version":1,"system":"go2","observation":{"position_noise_std_m":NaN}}',
    '{"schema_version":1,"system":"go2","observation":{"position_noise_std_m":1e999}}',
    '{"schema_version":1,"system":"go2","actuator":{"gain_range":0,"gain_range":0.1}}',
    '{"schema_version":1,"system":"quadrotor"}',
    '{"schema_version":true,"system":"go2"}',
    '{"schema_version":1,"system":"go2","actuator":null}',
    '{"schema_version":1,"system":"go2","timing":"ideal"}',
])
def test_strict_json_and_schema_validation(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_profile(path)


def test_profile_hash_prevents_changed_file_after_environment_reexec(tmp_path):
    path = profile_file(tmp_path)
    _, profile = load_profile(path)
    path.write_text(json.dumps({"schema_version": 1, "system": "go2", "actuator": {"gain_range": .2}}))
    with pytest.raises(ValueError):
        load_profile(path, expected_sha256=profile["sha256"])


def test_stdlib_dry_run_has_effective_profile_timing_and_environment_disturbances(tmp_path):
    path = profile_file(tmp_path, observation={"delay_steps": 3, "position_noise_std_m": .02},
                        actuator={"time_constant_s": 0, "gain_range": 0})
    output = tmp_path / "output"
    result = invoke("--sim2real-config", str(path), "--method", "nominal", "--scenario", "drain",
                    "--duration", ".04", "--ground-friction", ".25", "--output-dir", str(output), "--dry-run")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["protocol"] == "go2_sim2real_intra_period_v1"
    assert data["timing"]["observation_delay_ms"] == 60
    assert data["timing"]["publication_resolution_ms"] == 1
    assert data["config"]["duration"] == .04
    assert data["sim2real_profile"]["effective"]["actuator"]["time_constant_s"] == 0
    assert data["disturbances"]["ground"]["sliding"] == .25
    assert not output.exists()


@pytest.mark.parametrize("duration", ["0", "-1", ".03", "nan", "inf"])
def test_duration_validation_precedes_runtime(duration):
    result = invoke("--sim2real-config", str(EXAMPLE), "--duration", duration, "--dry-run")
    assert result.returncode == 2
    assert "0.02 s" in result.stderr


def test_mode_requires_explicit_profile_and_never_overwrites_existing_output(tmp_path, monkeypatch):
    assert invoke("--dry-run").returncode == 2
    assert invoke("--help").returncode == 0
    output = tmp_path / "combined_t2s_seed42_sim2real_delay20ms_60s"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("previous attempt")
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts/run_go2_sim2real.py"),
        "--sim2real-config", str(EXAMPLE), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Preserving existing output"):
        runner.main()
    assert sentinel.read_text() == "previous attempt"
    assert list(output.iterdir()) == [sentinel]


def test_controller_target_is_spawn_pickleable():
    assert pickle.loads(pickle.dumps(runner.sim2real_controller_process)) is runner.sim2real_controller_process


@pytest.fixture
def interfaces():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from src import go2_realism_io as io
    return np, io


def test_boundary_sensor_coherence_determinism_and_joint_encoder_preservation(interfaces):
    np, io = interfaces
    a, b = io.BoundarySensor(RealismOptions(), 42, .02), io.BoundarySensor(RealismOptions(), 42, .02)
    q = np.zeros(19); q[6] = 1.; q[7:] = np.arange(12) / 10
    dq = np.zeros(18)
    for _ in range(20):
        qa, va = a.observe(q, dq); qb, vb = b.observe(q, dq)
        np.testing.assert_array_equal(qa, qb); np.testing.assert_array_equal(va, vb)
        np.testing.assert_array_equal(qa[7:], q[7:])
        np.testing.assert_array_equal(va[6:], dq[6:])
        assert np.linalg.norm(qa[3:7]) == pytest.approx(1.)


def test_existing_joint_response_tau_gain_and_torque_limits(interfaces):
    np, io = interfaces
    motor = io.JointResponse(RealismOptions(actuator_gain_range=0), 42, .001, np.zeros(12))
    for _ in range(5):
        actual = motor.step(np.ones(12))
    np.testing.assert_allclose(actual, 1 - np.exp(-1))
    immediate = io.JointResponse(RealismOptions(actuator_tau_s=0, actuator_gain_range=0), 42, .001, np.zeros(12))
    np.testing.assert_array_equal(immediate.step(np.full(12, 1e6)), immediate.limit)
    a = io.JointResponse(RealismOptions(), 42, .001, np.zeros(12))
    b = io.JointResponse(RealismOptions(), 42, .001, np.zeros(12))
    np.testing.assert_array_equal(a.gain, b.gain)
    assert np.all((a.gain >= .97) & (a.gain <= 1.03))


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_configurable_delay_has_no_future_or_duplicate_learning_transitions(interfaces, delay):
    np, io = interfaces
    options = RealismOptions(observation_delay_steps=delay)
    options.validate()
    sensor = io.BoundarySensor(options, 42, .02)
    # Execute the driver's actual source-selection expression rather than a
    # duplicate test implementation; exercise real sensor and label adapters.
    tree = ast.parse((ROOT / "src/go2_realism_experiment.py").read_text())
    expression = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "source" for t in n.targets))
    source_code = compile(ast.Expression(expression), "<Go2 delayed packet selection>", "eval")
    packets, schedules, eligible, samples, delivered = [], [], [], [], []
    last_source = -1
    for cycle in range(9):
        q = np.zeros(19); q[0] = .1 * cycle; q[6] = 1.
        observed_q, observed_dq = sensor.observe(q, np.zeros(18))
        observed = np.r_[observed_q[:3], np.zeros(3), observed_dq[:6]]
        packets.append(dict(time=cycle * .02, state=observed, feet=np.full(12, cycle)))
        source = eval(source_code, {"cycle": cycle, "options": options})
        delivered.append(source)
        assert 0 <= source <= cycle
        if source > last_source:
            sample = io.aligned_transition(packets, schedules, eligible, source)
            if sample is not None:
                np.testing.assert_array_equal(sample[0], packets[source - 1]["state"])
                np.testing.assert_array_equal(sample[1], packets[source]["state"])
                np.testing.assert_allclose(sample[2], schedules[source - 1].mean(axis=0))
                assert sample[-1] < packets[source]["time"] <= packets[cycle]["time"]
                samples.append(sample)
        last_source = source
        schedules.append(np.r_[np.full((6, 12), cycle), np.full((14, 12), cycle + 1)])
        eligible.append(cycle != 2)
    assert delivered == [max(0, i - delay) for i in range(9)]
    timestamps = [s[-1] for s in samples]
    assert timestamps == [i * .02 for i in range(8 - delay) if i != 2]
    assert len(timestamps) == len(set(timestamps))


def test_intra_period_release_keeps_original_command_schedule(interfaces):
    np, _ = interfaces
    from src.deadline_control import ComputationAwareControlRelease
    release = ComputationAwareControlRelease(np.zeros(12))
    decision = release.release(np.ones(12), compute_ms=5.2, deadline_ms=20.,
                               substep_ms=1., substep_count=20, candidate_step=0)
    assert decision.held_substeps == 6
    np.testing.assert_array_equal(decision.command_schedule[:6], np.zeros((6, 12)))
    np.testing.assert_array_equal(decision.command_schedule[6:], np.ones((14, 12)))
    late = release.release(np.full(12, 2.), compute_ms=20.1, deadline_ms=20.,
                           substep_ms=1., substep_count=20, candidate_step=1)
    assert not late.candidate_accepted
    np.testing.assert_array_equal(late.command_schedule, np.ones((20, 12)))


def test_interruption_after_full_interval_retains_coherent_terminal_boundary(interfaces):
    """Execute the actual capture/finalization helpers without loading MuJoCo."""
    np, io = interfaces
    tree = ast.parse((ROOT / "src/go2_realism_experiment.py").read_text())
    definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name in ("capture_boundary", "retain_completed_endpoint")]
    truth = np.zeros(12); truth[2] = .3
    q = np.zeros(19); q[2] = .3; q[6] = 1.
    dq = np.zeros(18)
    go2 = SimpleNamespace(state=truth, current_config=SimpleNamespace(
        get_q=lambda: q.copy(), get_dq=lambda: dq.copy()))
    sensor_model = SimpleNamespace()
    sensor_model.update_model = lambda q, dq: setattr(
        sensor_model, "state", np.r_[q[:3], np.zeros(3), dq[:6]])
    namespace = dict(
        config=SimpleNamespace(dt=.02, speed=.5, height=.3, variable_friction=False,
                               liquid=False, payload_mass=4.),
        options=RealismOptions(), go2=go2, sensor_model=sensor_model,
        sensor=io.BoundarySensor(RealismOptions(), 42, .02),
        simulation=SimpleNamespace(update_pin_with_mujoco=lambda _: None,
            model=SimpleNamespace(geom=lambda _: SimpleNamespace(friction=[.8, .02, .01]))),
        world_state_to_paper_state=lambda state: state.copy(),
        _paper_world_state=lambda model: model.state,
        _negative_world_foot_levers=lambda _: np.zeros(12),
        NAMES=('x','y','z','roll','pitch','yaw','vx','vy','vz','omega_x','omega_y','omega_z'),
        initial_x=0., initial_y=0., packets=[], q_history=[], dq_history=[],
        noise_history=[], state_history=[], source_indices=[], eligible=[])
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(tree), "exec"), namespace)
    capture = namespace["capture_boundary"]
    retain = namespace["retain_completed_endpoint"]
    rows = [capture(0, 0.)[0]]
    command_history = []
    try:
        # A completed physical interval followed immediately by STOP/interrupt.
        truth[0] = q[0] = .012
        truth[6] = dq[0] = .6
        command_history.append(np.ones((20, 12)))
        namespace["eligible"].append(True)
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        retain(rows, len(command_history), capture)
    assert [row["time"] for row in rows] == [0., .02]
    assert rows[-1]["x"] == .012
    assert rows[-1]["vx_error"] == pytest.approx(.1)
    assert rows[-1]["source_time"] == 0.
    assert rows[-1]["observation_age_ms"] == 20.
    assert rows[-1]["fresh_control"] is False
    assert namespace["source_indices"] == [0, 0]
    for name in ("packets", "q_history", "dq_history", "noise_history", "state_history"):
        assert len(namespace[name]) == len(command_history) + 1
    np.testing.assert_array_equal(namespace["state_history"][-1], truth)
    # The actual metrics helper must report the integrated endpoint, not t=0.
    summary_tree = ast.parse((ROOT / "src/go2_icra_experiment.py").read_text())
    summary_function = next(n for n in summary_tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == "summarize")
    summary_namespace = {"np": np, "asdict": asdict}
    exec(compile(ast.Module(body=[summary_function], type_ignores=[]), "<Go2 metrics>", "exec"),
         summary_namespace)
    config, _ = runner.selected_config("nominal", "drain", 42)
    metrics = summary_namespace["summarize"](config, rows, [], "error", .02, 0)
    assert metrics["simulated_seconds"] == .02
    assert metrics["full_duration"] is False
    # Already recorded terminal boundaries and runs without an interval are untouched.
    retain(rows, len(command_history), lambda *_: pytest.fail("duplicate observation"))
    retain([], 0, lambda *_: pytest.fail("no physical interval"))


def test_candidate_published_at_19ms_has_no_wbc_effect_inside_interval():
    tree = ast.parse((ROOT / "src/go2_realism_experiment.py").read_text())
    expression = next(k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                      for k in n.keywords if k.arg == "intervals_without_new_candidate_effect")
    # 18 ms can reach the last WBC update; 19/20 ms wait for the next interval.
    jobs = [dict(cycle=0, applied=True, wbc_consumption_time=t) for t in (.018, .020, .020)]
    jobs.append(dict(cycle=0, applied=False, wbc_consumption_time=None))
    count = eval(compile(ast.Expression(expression), "<WBC-effect counter>", "eval"),
                 {"jobs": jobs, "config": SimpleNamespace(dt=.02)})
    assert count == 3
