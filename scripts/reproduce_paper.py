#!/usr/bin/env python3
"""Run the fixed 240-trial paper simulation suite serially (no hardware runs)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import inspect_trial, write_json, write_summaries

METHODS = ("nominal", "ssi", "stgp", "t2s")
SEEDS = tuple(range(42, 52))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def build_plan(system="all", root=ROOT):
    """Only checked-in paper configurations are accepted; no parameter overrides."""
    trials, files = [], set()
    systems = ("quadrotor", "go2") if system == "all" else (system,)
    for name in systems:
        path = root / ("configs/quadrotor.json" if name == "quadrotor" else "configs/go2/final.json")
        config = json.loads(path.read_text())
        files.add(path)
        files.add(root / f"scripts/run_{name}.py")
        if config.get("schema_version") != 1:
            raise ValueError(f"Unsupported configuration schema: {path}")
        scenarios = ("mwi", "tii", "combined") if name == "quadrotor" else ("drain", "friction", "combined")
        if name == "quadrotor":
            if tuple(config["seeds"]) != SEEDS or set(config["methods"]) != set(METHODS) or set(config["scenarios"]) != set(scenarios):
                raise ValueError("Quadrotor configuration does not contain the complete paper matrix")
            common = config["common_args"]
            duration = float(common[common.index("--duration") + 1])
            hz = float(common[common.index("--control-frequency") + 1])
        else:
            expected = {f"seed{s}/{c}/{m}" for s in SEEDS for c in scenarios for m in METHODS}
            if set(config["configs"]) != expected:
                raise ValueError("Go2 configuration does not contain the complete paper matrix")
            for seed in SEEDS:
                field = config["fields"][str(seed)]
                path = root / field["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != field["sha256"]:
                    raise ValueError(f"Frozen friction-map checksum mismatch: {path}")
                files.add(path)
        for scenario in scenarios:
            for method in METHODS:
                for seed in SEEDS:
                    trial = dict(id=f"{name}/{scenario}/{method}/seed{seed}", system=name,
                                 scenario=scenario, method=method, seed=seed)
                    if name == "go2":
                        raw = config["configs"][f"seed{seed}/{scenario}/{method}"]
                        if (raw["method"], raw["scenario"], raw["seed"]) != (method, scenario, seed):
                            raise ValueError("Go2 configuration identity mismatch")
                        duration, hz = float(raw["duration"]), float(raw["mpc_hz"])
                        trial["config_sha256"] = digest(raw)
                        trial["expected_release"] = dict(
                            protocol=config["source_protocol"],
                            config_sha256=hashlib.sha256((root / "configs/go2/final.json").read_bytes()).hexdigest(),
                            adapter_sha256=hashlib.sha256((root / "src/go2_public_protocol.py").read_bytes()).hexdigest(),
                            mpc_friction_coefficient=config["mpc_friction_coefficient"], fresh_run=True)
                        trial["field_sha256"] = config["fields"][str(seed)]["sha256"]
                    else:
                        flags = common + config["scenarios"][scenario]["args"] + config["methods"][method]["args"]
                        flags = flags + ["--seed", str(seed), "--control-cpu-core", str(config["cpu_affinity"]["control"])]
                        if method == "t2s":
                            flags += ["--trainer-cpu-core", str(config["cpu_affinity"]["trainer"])]
                        trial["expected_args"] = flags
                    trial.update(expected_duration_s=duration, control_hz=hz,
                                 planned_control_periods=round(duration * hz))
                    trials.append(trial)
    # Include simulation source and dependency declarations to prevent mixed-source resume.
    files.update((root / "src").glob("*.py"))
    files.update((root / "requirements").glob("*.txt"))
    files.add(root / "docs/dependencies.json")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    return dict(schema_version=1, system=system, protocol_files=hashes, trials=trials,
                scope="fixed paper simulation suite; no hardware experiments")


def environment():
    packages = {}
    for name in ("numpy", "scipy", "torch", "casadi", "l4casadi", "gpytorch", "pybullet", "mujoco", "pin"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    native = {}
    for name in ("acados", "l4acados", "safe-control-gym", "go2-convex-mpc"):
        override = {"acados": "ACADOS_SOURCE_DIR", "go2-convex-mpc": "GO2_CONVEX_MPC_DIR"}.get(name)
        path = Path(os.environ.get(override, ROOT / "external" / name) if override else ROOT / "external" / name).expanduser().resolve()
        native[name] = {"path": str(path), "commit": None, "dirty": None}
        if path.is_dir():
            try:
                commands = (["rev-parse", "HEAD"], ["status", "--porcelain", "--untracked-files=no"])
                for key, command in zip(("commit", "dirty"), commands):
                    result = subprocess.run(["git", "--no-optional-locks", "-C", str(path), *command], capture_output=True, text=True, timeout=5)
                    native[name][key] = result.stdout.strip() if result.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                pass
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), hostname=socket.gethostname(),
                available_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                packages=packages, native_dependencies=native)


def command_for(trial, attempt, root=ROOT):
    command = [sys.executable, "-u", str(root / f"scripts/run_{trial['system']}.py"),
               "--method", trial["method"], "--scenario", trial["scenario"],
               "--seed", str(trial["seed"]), "--output-dir", str(attempt / "artifacts")]
    if trial["system"] == "quadrotor":
        command += ["--config", str(root / "configs/quadrotor.json"), "--result-tag", "paper"]
    return command


def prepare_output(output, plan, resume, current_environment):
    manifest_path = output / "manifest.json"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not resume or not manifest_path.is_file():
            raise ValueError(f"Preserving existing output: {output}; use a new directory or --resume for this suite")
        stored = json.loads(manifest_path.read_text())
        if stored.get("plan") != plan:
            raise ValueError("Resume refused: suite selection, configuration, or simulation source changed")
        if stored.get("environment") != current_environment:
            raise ValueError("Resume refused: interpreter, packages, host, or available CPU set changed; use a new directory")
        return stored
    if resume:
        raise ValueError("--resume requires an existing suite manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(created_at=stamp(), plan=plan, environment=current_environment,
                    timing="Host-dependent measured controller-latency replay; runs are serial.")
    write_json(manifest_path, manifest)
    return manifest


def run_attempt(trial, output):
    directory = output / "trials" / trial["id"]
    attempts = sorted(directory.glob("attempt-*"))
    number = max((int(p.name.split("-")[-1]) for p in attempts), default=0) + 1
    attempt = directory / f"attempt-{number:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    command = command_for(trial, attempt)
    record = dict(trial_id=trial["id"], state="running", started_at=stamp(), command=command)
    write_json(attempt / "attempt.json", record)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(attempt / "matplotlib")
    env["PYTHONUNBUFFERED"] = "1"
    interrupted = False
    with (attempt / "runner.log").open("x") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                os.killpg(process.pid, signal.SIGINT)
                try:
                    record["returncode"] = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        record["returncode"] = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        record["returncode"] = process.wait()
        except OSError as exc:
            record.update(returncode=-1, error=str(exc))
    record.update(state="interrupted" if interrupted else "finished", finished_at=stamp())
    write_json(attempt / "attempt.json", record)
    return interrupted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("all", "quadrotor", "go2"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper")
    parser.add_argument("--dry-run", action="store_true", help="Print the full fixed plan; do not import simulators or write files")
    parser.add_argument("--resume", action="store_true", help="Continue the same suite; preserve completed and valid early-terminated trials, retain earlier attempts")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.system)
        output = args.output_dir.expanduser().resolve()
        if args.dry_run:
            print(json.dumps(dict(trial_count=len(plan["trials"]), output_dir=str(output),
                                  resume=args.resume, plan=plan,
                                  commands=[command_for(t, output / "trials" / t["id"] / "attempt-001") for t in plan["trials"]]), indent=2))
            return 0
        manifest = prepare_output(output, plan, args.resume, environment())
        # A lock prevents simultaneous resume into the same suite output.
        import fcntl
        with (output / ".suite.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Another suite process is using {output}")
            for index, trial in enumerate(plan["trials"], 1):
                previous, _ = inspect_trial(trial, output)
                if args.resume and previous["status"] in ("completed", "early_terminated"):
                    print(f"[{index}/{len(plan['trials'])}] preserve {trial['id']}: {previous['status']}", flush=True)
                    continue
                print(f"[{index}/{len(plan['trials'])}] run {trial['id']}", flush=True)
                interrupted = run_attempt(trial, output)
                current, _ = inspect_trial(trial, output)
                print(f"  {current['status']}: {current.get('reason', '')}", flush=True)
                if interrupted:
                    write_summaries(output, manifest)
                    return 130
            report = write_summaries(output, manifest)
        print(f"Completed {report['completed']}/{report['planned']} trials. Summaries: {output}")
        return 0 if report["completed"] == report["planned"] else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"reproduce_paper: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
