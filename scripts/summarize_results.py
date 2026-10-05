#!/usr/bin/env python3
"""Summarize actual artifacts from a reproduce_paper suite, including all failures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STATES = ("completed", "early_terminated", "failed", "missing", "interrupted", "invalid")
TRACKING_METRICS = ("mean_position_error_m", "mean_attitude_error_rad", "position_rmse_m", "attitude_rmse_rad",
           "vx_rmse", "height_rmse", "y_rmse", "roll_pitch_rmse", "pace_x_rmse")
METRICS = TRACKING_METRICS + ("controller_mean_ms", "controller_p95_ms")
COUNTERS = ("observed_control_periods", "launched_tasks", "deadline_misses", "busy_periods", "solver_failures")


def write_json(path, value):
    """Replace only the suite's own metadata or derived summaries atomically."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def required_number(value, name):
    result = number(value)
    if result is None:
        raise ValueError(f"Missing/nonfinite {name}")
    return result


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def only_file(directory, pattern):
    paths = list(directory.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing artifact: {pattern}")
    if len(paths) != 1:
        raise ValueError(f"Ambiguous artifacts: {pattern} ({len(paths)} files)")
    return paths[0]


def trajectory(path):
    rows = csv_rows(path)
    times = [required_number(r.get("time"), "trajectory time") for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Trajectory timestamps are not strictly increasing")
    return rows, times


def rmse(rows, key):
    values = [required_number(r.get(key), key) for r in rows]
    return math.sqrt(statistics.fmean(v * v for v in values)) if values else None


def validate_grid(times, count, hz, starts_at_zero):
    if len(times) != count:
        return False
    offset = 0 if starts_at_zero else 1
    return all(abs(t - (i + offset) / hz) <= 1e-6 for i, t in enumerate(times))


def quadrotor_artifacts(trial, artifacts):
    path = only_file(artifacts, "*_summary.csv")
    records = csv_rows(path)
    if len(records) != 1:
        raise ValueError("Quadrotor summary must contain exactly one trial")
    summary = records[0]
    if summary.get("controller") != trial["method"] or int(summary["seed"]) != trial["seed"]:
        raise ValueError("Quadrotor summary identity differs from the planned trial")
    protocol = json.loads((path.parent / "run_protocol.json").read_text())
    if (protocol["controller"], protocol["scenario"], protocol["seed"]) != (trial["method"], trial["scenario"], trial["seed"]):
        raise ValueError("Quadrotor protocol identity differs from the planned trial")
    actual_args, skip = [], False
    for arg in protocol["args"]:
        if skip:
            skip = False
        elif arg in ("--output-dir", "--result-tag"):
            skip = True
        else:
            actual_args.append(arg)
    if actual_args != trial["expected_args"]:
        raise ValueError("Quadrotor arguments differ from the fixed paper protocol")
    stem = path.name.removesuffix("_summary.csv")
    rows, times = trajectory(path.with_name(stem + ".csv"))
    timing = csv_rows(path.with_name(stem + "_timing.csv"))
    if len(timing) != len(rows):
        raise ValueError("Quadrotor timing and trajectory lengths disagree")
    if required_number(summary.get("control_frequency_hz"), "control frequency") != trial["control_hz"]:
        raise ValueError("Quadrotor control frequency differs from protocol")
    periods = trial["planned_control_periods"]
    if len(times) > periods or not validate_grid(times, len(times), trial["control_hz"], False):
        raise ValueError("Quadrotor trajectory has an invalid time grid")
    complete = len(times) == periods
    if required_number(summary.get("steps"), "steps") != len(rows):
        raise ValueError("Quadrotor step count disagrees with trajectory")
    values = dict(position_rmse_m=rmse(rows, "position_error_xyz"),
                  attitude_rmse_rad=rmse(rows, "attitude_error"),
                  mean_position_error_m=statistics.fmean(required_number(r["position_error_xyz"], "position error") for r in rows) if rows else None,
                  mean_attitude_error_rad=statistics.fmean(required_number(r["attitude_error"], "attitude error") for r in rows) if rows else None)
    result = dict(status="completed" if complete else "early_terminated",
                  reason="full fixed-grid trajectory" if complete else "trajectory does not cover the full paper horizon",
                  simulated_seconds=times[-1] if times else 0,
                  artifact=str(path.relative_to(artifacts)),
                  observed_control_periods=len(rows), launched_tasks=len(timing),
                  deadline_misses=sum(required_number(r.get("control_critical_ms"), "control_critical_ms") > 1000.0 / trial["control_hz"] for r in timing),
                  solver_failures=required_number(summary.get("solver_failures"), "solver failures"),
                  controller_mean_ms=number(summary.get("control_critical_mean_ms")),
                  controller_p95_ms=number(summary.get("control_critical_p95_ms")))
    result.update(values if complete else {"prefix_" + k: v for k, v in values.items()})
    return result


def go2_artifacts(trial, artifacts):
    path = only_file(artifacts, "metrics.json")
    metrics = json.loads(path.read_text())
    actual = hashlib.sha256(json.dumps(metrics["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != trial["config_sha256"]:
        raise ValueError("Go2 configuration differs from the fixed paper protocol")
    release = json.loads((path.parent / "release_protocol.json").read_text())
    if any(release.get(key) != value for key, value in trial["expected_release"].items()):
        raise ValueError("Go2 release metadata differs from the paper protocol")
    if release.get("disturbances", {}).get("requested_overrides") != {}:
        raise ValueError("Custom Go2 disturbance overrides cannot enter the paper summary")
    if release.get("field", {}).get("sha256") != trial["field_sha256"]:
        raise ValueError("Go2 friction-field metadata differs from the paper protocol")
    rows, times = trajectory(path.parent / "trajectory.csv")
    seconds = required_number(metrics.get("simulated_seconds"), "simulated duration")
    if times and abs(times[-1] - seconds) > 1e-6:
        raise ValueError("Go2 metrics duration disagrees with trajectory")
    raw_status = metrics.get("status")
    if raw_status not in ("completed", "fell", "stopped_by_request", "wall_timeout", "error"):
        raise ValueError("Unknown Go2 termination status")
    if len(times) > trial["planned_control_periods"] + 1 or not validate_grid(times, len(times), trial["control_hz"], True):
        raise ValueError("Go2 trajectory has an invalid time grid")
    full_grid = validate_grid(times, trial["planned_control_periods"] + 1, trial["control_hz"], True)
    complete = raw_status == "completed" and metrics.get("full_duration") is True and full_grid
    # Requested stops and host watchdog timeouts are recoverable execution outcomes.
    # Only a fall is a scientific early termination that resume must preserve.
    status = "completed" if complete else {
        "completed": "invalid", "fell": "early_terminated",
        "stopped_by_request": "interrupted", "wall_timeout": "failed", "error": "failed",
    }[raw_status]
    result = dict(status=status, raw_status=raw_status, reason=str(raw_status), simulated_seconds=seconds,
                  artifact=str(path.relative_to(artifacts)), observed_control_periods=max(0, len(rows) - 1),
                  launched_tasks=number(metrics.get("launched_jobs")),
                  deadline_misses=number(metrics.get("deadline_misses")),
                  busy_periods=number(metrics.get("skipped_busy_cycles")),
                  solver_failures=number(metrics.get("solver_failures")))
    if raw_status == "completed" and not complete:
        result.update(status="invalid", reason="reported completion lacks full-duration trajectory evidence")
    tracking = metrics.get("tracking" if complete else "prefix_diagnostics") or {}
    for key in ("vx_rmse", "height_rmse", "y_rmse", "roll_pitch_rmse", "pace_x_rmse"):
        value = required_number(tracking.get(key), key) if complete else number(tracking.get(key))
        result[key if complete else "prefix_" + key] = value
    timing = metrics.get("timing_ms") or {}
    result.update(controller_mean_ms=number(timing.get("mean")), controller_p95_ms=number(timing.get("p95")))
    return result


def inspect_attempt(trial, attempt):
    row = {k: trial[k] for k in ("id", "system", "scenario", "method", "seed", "expected_duration_s", "planned_control_periods")}
    row.update(attempt=attempt.name, status="missing", reason="attempt metadata missing", log=str(attempt / "runner.log"))
    metadata = attempt / "attempt.json"
    if not metadata.is_file():
        return row
    try:
        record = json.loads(metadata.read_text())
        if record.get("trial_id") != trial["id"]:
            raise ValueError("Attempt metadata belongs to another trial")
        row.update(returncode=record.get("returncode"), started_at=record.get("started_at"), finished_at=record.get("finished_at"))
        reader = quadrotor_artifacts if trial["system"] == "quadrotor" else go2_artifacts
        try:
            row.update(reader(trial, attempt / "artifacts"))
        except FileNotFoundError as exc:
            row.update(status="missing", reason=str(exc))
        except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
            row.update(status="invalid", reason=str(exc))
        row["artifact_status"] = row["status"]
        if record.get("state") != "finished":
            row.update(status="interrupted", reason="attempt did not finish; artifacts cannot establish process success")
        elif record.get("returncode") != 0:
            row.update(status="failed", reason=f"runner exit {record.get('returncode')}: {record.get('error', row['reason'])}")
        # Prefixes remain diagnostics even if a later plotting/save error follows a full trajectory.
        if row["status"] != "completed":
            for key in TRACKING_METRICS:
                if key in row:
                    row["prefix_" + key] = row.pop(key)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        row.update(status="invalid", reason=str(exc))
    return row


def inspect_trial(trial, output):
    directory = output / "trials" / trial["id"]
    attempts = [inspect_attempt(trial, p) for p in sorted(directory.glob("attempt-*")) if p.is_dir()]
    if attempts:
        # A valid scientific termination is terminal, including a fall. Never pick a later lucky retry.
        row = next((r for r in attempts if r["status"] in ("completed", "early_terminated")), attempts[-1]).copy()
    else:
        row = {k: trial[k] for k in ("id", "system", "scenario", "method", "seed", "expected_duration_s", "planned_control_periods")}
        row.update(status="missing", reason="not started")
    row.update(attempt_count=len(attempts), unsuccessful_attempts=sum(r["status"] != "completed" for r in attempts))
    return row, attempts


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["scenario"], row["method"])].append(row)
    groups = []
    for (system, scenario, method), trials in sorted(grouped.items()):
        counts = Counter(r["status"] for r in trials)
        group = dict(system=system, scenario=scenario, method=method, planned=len(trials),
                     **{status: counts[status] for status in STATES})
        group.update(completion_rate=counts["completed"] / len(trials),
                     attempted=sum(r["attempt_count"] > 0 for r in trials),
                     total_attempts=sum(r["attempt_count"] for r in trials),
                     unsuccessful_attempts=sum(r["unsuccessful_attempts"] for r in trials),
                     planned_control_periods=sum(r["planned_control_periods"] for r in trials))
        for key in COUNTERS:
            values = [r[key] for r in trials if number(r.get(key)) is not None]
            group[key + "_observed_total"] = sum(values) if values else None
            group[key + "_trials_observed"] = len(values)
        for key in METRICS:
            values = [r[key] for r in trials if r["status"] == "completed" and number(r.get(key)) is not None]
            group[key + "_n"] = len(values)
            group[key + "_mean"] = statistics.fmean(values) if values else None
            group[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    return groups


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["id", "system", "scenario", "method", "seed", "attempt", "status", "reason"]
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", newline="", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_summaries(output, manifest):
    rows, attempts = [], []
    for trial in manifest["plan"]["trials"]:
        row, history = inspect_trial(trial, output)
        rows.append(row)
        attempts.extend(history)
    groups = aggregate(rows)
    report = dict(planned=len(rows), completed=sum(r["status"] == "completed" for r in rows),
                  status_counts=dict(Counter(r["status"] for r in rows)),
                  quadrotor_paper_metrics="Per-trial mean Euclidean position error and mean SO(3) angle; across-seed sample std. RMSE columns are supplementary. Deadline misses count control_critical_ms > the control period (20 ms).",
                  metric_policy="Full-trajectory metrics use completed trials only; n and all planned outcomes are retained. Std is sample std across seeds. Controller p95 columns average trial-level p95 values, not pooled percentiles.",
                  resume_policy="Keep the first valid completed or scientific early-terminated outcome. Requested stops and host watchdog timeouts can retry; earlier attempts remain in attempts.csv.",
                  trials=rows, groups=groups)
    write_csv(output / "trials.csv", rows)
    write_csv(output / "attempts.csv", attempts)
    write_csv(output / "summary.csv", groups)
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper")
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        manifest = json.loads((output / "manifest.json").read_text())
        report = write_summaries(output, manifest)
        print(f"Completed {report['completed']}/{report['planned']}; summaries written to {output}")
        return 0 if report["completed"] == report["planned"] else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"summarize_results: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
