"""Batch bookkeeping tests use synthetic artifacts only; no simulator is imported."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_paper as batch
import summarize_results as summary


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_trial(system="go2"):
    # A tiny synthetic horizon keeps tests independent of numerical dependencies.
    trial = dict(id=f"{system}/combined/t2s/seed42", system=system,
                 scenario="combined", method="t2s", seed=42,
                 expected_duration_s=2.0, control_hz=2.0, planned_control_periods=4)
    if system == "go2":
        raw = dict(method="t2s", scenario="combined", seed=42, duration=2.0, mpc_hz=2.0)
        trial["config_sha256"] = batch.digest(raw)
        trial["expected_release"] = {"protocol": "synthetic", "mpc_friction_coefficient": .5, "fresh_run": True}
        trial["field_sha256"] = "synthetic-field-hash"
    else:
        raw = None
        trial["expected_args"] = ["--duration", "2", "--seed", "42"]
    return trial, raw


def make_attempt(output, trial, raw=None, *, number=1, status="completed", returncode=0, state="finished"):
    attempt = output / "trials" / trial["id"] / f"attempt-{number:03d}"
    attempt.mkdir(parents=True)
    summary.write_json(attempt / "attempt.json", dict(trial_id=trial["id"], returncode=returncode, state=state))
    (attempt / "runner.log").write_text("synthetic fixture, not an experiment\n")
    if trial["system"] == "go2":
        folder = attempt / "artifacts" / "go2_trial"
        folder.mkdir(parents=True)
        times = [0, .5, 1, 1.5, 2] if status == "completed" else [0, .5, 1]
        csv_file(folder / "trajectory.csv", [{"time": t} for t in times])
        metrics = dict(config=raw, status=status, full_duration=status == "completed",
                       simulated_seconds=times[-1], launched_jobs=3, deadline_misses=1,
                       skipped_busy_cycles=1, solver_failures=0, timing_ms={"mean": 10, "p95": 22})
        metrics["tracking" if status == "completed" else "prefix_diagnostics"] = dict(
            vx_rmse=.3, height_rmse=.04, y_rmse=.02, roll_pitch_rmse=.05, pace_x_rmse=.2)
        summary.write_json(folder / "metrics.json", metrics)
        summary.write_json(folder / "release_protocol.json", dict(
            **trial["expected_release"], field={"sha256": trial["field_sha256"]},
            disturbances={"requested_overrides": {}}))
    else:
        folder = attempt / "artifacts" / "t2s/combined/seed42"
        folder.mkdir(parents=True)
        protocol = dict(controller="t2s", scenario="combined", seed=42,
                        args=trial["expected_args"] + ["--result-tag", "paper", "--output-dir", str(folder)])
        summary.write_json(folder / "run_protocol.json", protocol)
        csv_file(folder / "quad.csv", [{"time": (i + 1) / 2, "position_error_xyz": x, "attitude_error": x / 10}
                                       for i, x in enumerate([0, 0, 0, 4])])
        csv_file(folder / "quad_timing.csv", [{"step": i, "control_critical_ms": ms} for i, ms in enumerate([499, 500, 501, 1000])])
        csv_file(folder / "quad_summary.csv", [dict(controller="t2s", seed=42, steps=4,
                 control_frequency_hz=2, control_deadline_misses=2, control_release_deadline_misses=99,
                 solver_failures=0, control_critical_mean_ms=625, control_critical_p95_ms=1000)])
    return attempt


def test_plan_contains_all_240_fixed_trials():
    plan = batch.build_plan()
    assert len(plan["trials"]) == 240
    assert len({t["id"] for t in plan["trials"]}) == 240
    for system in ("quadrotor", "go2"):
        trials = [t for t in plan["trials"] if t["system"] == system]
        assert len(trials) == 120
        assert {t["seed"] for t in trials} == set(range(42, 52))
        assert {t["method"] for t in trials} == set(batch.METHODS)
        assert {t["expected_duration_s"] for t in trials} == ({20} if system == "quadrotor" else {60})
        assert len(batch.build_plan(system)["trials"]) == 120
        for trial in trials:
            command = batch.command_for(trial, Path("attempt"))
            assert not any(flag in command for flag in ("--wind-scale", "--turbulence-scale", "--payload-mass", "--duration"))


def test_shell_dry_run_is_dependency_free_and_does_not_create_output(tmp_path):
    output = tmp_path / "new/results"
    result = subprocess.run(["bash", str(ROOT / "scripts/reproduce_paper.sh"), "--dry-run", "--output-dir", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["trial_count"] == 240
    assert len(plan["commands"]) == 240
    assert not output.parent.exists()
    assert not list(tmp_path.iterdir())


def test_mean_error_is_not_replaced_by_rmse_or_release_deadlines(tmp_path):
    trial, _ = fixture_trial("quadrotor")
    make_attempt(tmp_path, trial)
    row, _ = summary.inspect_trial(trial, tmp_path)
    assert row["status"] == "completed"
    assert row["mean_position_error_m"] == 1.0
    assert row["position_rmse_m"] == 2.0
    assert row["mean_attitude_error_rad"] == .1
    assert row["deadline_misses"] == 2  # > period, excluding equality and release misses.


def test_valid_early_termination_never_replaced_by_later_success(tmp_path):
    trial, raw = fixture_trial()
    make_attempt(tmp_path, trial, raw, status="fell")
    make_attempt(tmp_path, trial, raw, number=2)
    row, attempts = summary.inspect_trial(trial, tmp_path)
    assert row["status"] == "early_terminated"
    assert row["attempt"] == "attempt-001"
    assert "vx_rmse" not in row
    assert row["prefix_vx_rmse"] == .3
    assert len(attempts) == 2
    group = summary.aggregate([row])[0]
    assert group["completed"] == 0
    assert group["vx_rmse_n"] == 0
    assert group["vx_rmse_mean"] is None


@pytest.mark.parametrize("returncode,state,expected", [(0, "finished", "completed"), (7, "finished", "failed"), (None, "running", "interrupted")])
def test_process_outcome_cannot_be_hidden_by_complete_artifacts(tmp_path, returncode, state, expected):
    trial, raw = fixture_trial()
    make_attempt(tmp_path, trial, raw, returncode=returncode, state=state)
    row, _ = summary.inspect_trial(trial, tmp_path)
    assert row["status"] == expected
    if expected != "completed":
        assert "vx_rmse" not in row
        assert row["prefix_vx_rmse"] == .3


def test_missing_invalid_and_truncated_results_are_visible(tmp_path):
    trial, raw = fixture_trial()
    assert summary.inspect_trial(trial, tmp_path)[0]["status"] == "missing"
    attempt = make_attempt(tmp_path, trial, raw)
    path = attempt / "artifacts/go2_trial/metrics.json"
    path.write_text("broken JSON")
    assert summary.inspect_trial(trial, tmp_path)[0]["status"] == "invalid"
    path.unlink()
    assert summary.inspect_trial(trial, tmp_path)[0]["status"] == "missing"
    make_attempt(tmp_path, trial, raw, number=2)
    folder = tmp_path / "trials" / trial["id"] / "attempt-002/artifacts/go2_trial"
    csv_file(folder / "trajectory.csv", [{"time": 0}, {"time": .5}])
    metrics = json.loads((folder / "metrics.json").read_text())
    metrics["simulated_seconds"] = .5
    summary.write_json(folder / "metrics.json", metrics)
    assert summary.inspect_trial(trial, tmp_path)[0]["status"] == "invalid"


def test_custom_config_cannot_enter_paper_summary(tmp_path):
    trial, raw = fixture_trial()
    raw["duration"] = 8.0
    make_attempt(tmp_path, trial, raw)
    row, _ = summary.inspect_trial(trial, tmp_path)
    assert row["status"] == "invalid"
    assert "fixed paper protocol" in row["reason"]


def test_aggregate_preserves_planned_denominator_sample_std_and_attempts(tmp_path):
    trial, raw = fixture_trial()
    make_attempt(tmp_path, trial, raw, returncode=4)
    make_attempt(tmp_path, trial, raw, number=2)
    one, attempts = summary.inspect_trial(trial, tmp_path)
    two = dict(one, seed=43, vx_rmse=.5)
    missing = dict(one, seed=44, status="missing", attempt_count=0, unsuccessful_attempts=0)
    group = summary.aggregate([one, two, missing])[0]
    assert group["planned"] == 3
    assert group["completed"] == 2
    assert group["completion_rate"] == 2 / 3
    assert group["vx_rmse_mean"] == .4
    assert group["vx_rmse_std"] == pytest.approx(2 ** .5 / 10)
    assert group["vx_rmse_n"] == 2
    assert attempts[0]["status"] == "failed"
    report = summary.write_summaries(tmp_path, {"plan": {"trials": [trial]}})
    assert report["completed"] == 1
    assert len(summary.csv_rows(tmp_path / "attempts.csv")) == 2
    assert json.loads((tmp_path / "summary.json").read_text())["planned"] == 1


def test_resume_and_existing_output_guards(tmp_path, monkeypatch):
    trial, raw = fixture_trial()
    plan = dict(trials=[trial], marker="original")
    env = {"test": True}
    monkeypatch.setattr(batch, "build_plan", lambda system: plan)
    monkeypatch.setattr(batch, "environment", lambda: env)
    calls = []
    def failed_attempt(trial, output):
        calls.append(trial["id"])
        make_attempt(output, trial, raw, returncode=3)
        return False
    monkeypatch.setattr(batch, "run_attempt", failed_attempt)
    output = tmp_path / "suite"
    assert batch.main(["--output-dir", str(output)]) == 1
    assert batch.main(["--output-dir", str(output)]) == 2
    assert len(calls) == 1
    old_log = output / "trials" / trial["id"] / "attempt-001/runner.log"
    old_contents = old_log.read_bytes()
    def successful_retry(trial, output):
        calls.append(trial["id"])
        make_attempt(output, trial, raw, number=2)
        return False
    monkeypatch.setattr(batch, "run_attempt", successful_retry)
    assert batch.main(["--output-dir", str(output), "--resume"]) == 0
    assert old_log.read_bytes() == old_contents
    monkeypatch.setattr(batch, "run_attempt", lambda *args: pytest.fail("completed trial was rerun"))
    assert batch.main(["--output-dir", str(output), "--resume"]) == 0
    plan["marker"] = "changed"
    assert batch.main(["--output-dir", str(output), "--resume"]) == 2


def test_resume_preserves_valid_fall(tmp_path, monkeypatch):
    trial, raw = fixture_trial()
    plan = {"trials": [trial]}
    output = tmp_path / "suite"
    batch.prepare_output(output, plan, False, {})
    make_attempt(output, trial, raw, status="fell")
    monkeypatch.setattr(batch, "build_plan", lambda system: plan)
    monkeypatch.setattr(batch, "environment", lambda: {})
    monkeypatch.setattr(batch, "run_attempt", lambda *args: pytest.fail("fall was silently retried"))
    assert batch.main(["--output-dir", str(output), "--resume"]) == 1


def test_custom_disturbance_with_unchanged_go2_config_is_rejected(tmp_path):
    trial, raw = fixture_trial()
    attempt = make_attempt(tmp_path, trial, raw)
    path = attempt / "artifacts/go2_trial/release_protocol.json"
    release = json.loads(path.read_text())
    release["disturbances"]["requested_overrides"] = {"friction_min": .2}
    summary.write_json(path, release)
    row, _ = summary.inspect_trial(trial, tmp_path)
    assert row["status"] == "invalid"
    assert "Custom Go2 disturbance" in row["reason"]


@pytest.mark.parametrize("raw_status,expected", [("stopped_by_request", "interrupted"), ("wall_timeout", "failed")])
def test_resume_retries_stop_and_watchdog_without_overwriting_history(tmp_path, monkeypatch, raw_status, expected):
    trial, raw = fixture_trial()
    plan = {"trials": [trial]}
    output = tmp_path / "suite"
    batch.prepare_output(output, plan, False, {})
    old_attempt = make_attempt(output, trial, raw, status=raw_status)
    before = {p.relative_to(old_attempt): p.read_bytes() for p in old_attempt.rglob("*") if p.is_file()}
    assert summary.inspect_trial(trial, output)[0]["status"] == expected
    monkeypatch.setattr(batch, "build_plan", lambda system: plan)
    monkeypatch.setattr(batch, "environment", lambda: {})
    retried = []
    def successful_retry(planned, destination):
        retried.append(planned["id"])
        make_attempt(destination, planned, raw, number=2)
        return False
    monkeypatch.setattr(batch, "run_attempt", successful_retry)
    assert batch.main(["--output-dir", str(output), "--resume"]) == 0
    assert retried == [trial["id"]]
    assert before == {p.relative_to(old_attempt): p.read_bytes() for p in old_attempt.rglob("*") if p.is_file()}
    row, attempts = summary.inspect_trial(trial, output)
    assert row["status"] == "completed"
    assert row["attempt"] == "attempt-002"
    assert row["unsuccessful_attempts"] == 1
    assert [a["status"] for a in attempts] == [expected, "completed"]
    assert [r["status"] for r in summary.csv_rows(output / "attempts.csv")] == [expected, "completed"]
