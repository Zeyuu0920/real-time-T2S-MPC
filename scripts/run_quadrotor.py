#!/usr/bin/env python3
"""Run the selected quadrotor protocol; --help and --dry-run use only stdlib."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reading a profile/dry-run must not create local import cache files.
_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    from src.sim2real_config import apply_quadrotor_config, load_quadrotor_config
finally:
    sys.dont_write_bytecode = _write_bytecode
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def nonnegative_finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def scale_disturbances(flags, wind_scale, turbulence_scale):
    """Scale mean/sigma endpoints, preserving the configured temporal/spatial shape."""
    for prefix, factor in (("mean-wind", wind_scale), ("turbulence-sigma", turbulence_scale)):
        if factor == 1.0:
            continue  # Preserve the paper argument strings exactly at defaults.
        for axis in "xyz":
            for suffix in ("", "-end"):
                flag = f"--{prefix}-{axis}{suffix}"
                index = flags.index(flag) + 1
                scaled = float(flags[index]) * factor
                if not math.isfinite(scaled):
                    raise SystemExit(f"Scaling {flag} produced a nonfinite value; use a smaller factor.")
                flags[index] = str(scaled)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", "--method", choices=("t2s", "nominal", "ssi", "stgp", "all"), default="t2s")
    parser.add_argument("--scenario", choices=("mwi", "tii", "combined", "all"), default="mwi")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=42, help="Random seed (0 through 4294967295); paper seeds are 42 through 51")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Run these seeds serially; paper evaluation uses 42 through 51")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "quadrotor.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "quadrotor", help="Base directory; runs use CONTROLLER/SCENARIO/seedN below it")
    parser.add_argument("--duration", type=float, help="Override the 20 s paper duration (a different protocol)")
    parser.add_argument("--wind-scale", type=nonnegative_finite, default=1.0,
                        help="Multiply all mean-wind start/end components (default: 1); turbulence is set separately")
    parser.add_argument("--turbulence-scale", type=nonnegative_finite, default=1.0,
                        help="Multiply all turbulence standard-deviation start/end components (default: 1)")
    parser.add_argument("--sim2real-config", type=Path,
                        help="JSON profile overriding observation delay/noise and first-order motor parameters")
    parser.add_argument("--result-tag", default="public", help="Up to 16 ASCII letters, digits, underscore or hyphen")
    parser.add_argument("--control-cpu-core", type=int, help="Override recorded control CPU 2")
    parser.add_argument("--trainer-cpu-core", type=int, help="Override recorded T2S trainer CPU 4")
    parser.add_argument("--no-affinity", action="store_true", help="Omit original CPU pinning; timing will use a different resource setup")
    parser.add_argument("--dry-run", action="store_true", help="Print exact runs and arguments without importing simulation dependencies")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_sim2real-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._sim2real_sha256 and (not args._worker or args.sim2real_config is None):
        parser.error("Internal sim2real hash requires a worker and --sim2real-config")
    try:
        args.sim2real_profile = (load_quadrotor_config(args.sim2real_config, args._sim2real_sha256)
                                 if args.sim2real_config is not None else None)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if len(args.result_tag) > 16 or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in args.result_tag):
        parser.error("--result-tag must contain at most 16 ASCII letters, digits, '_' or '-'")
    if any(not 0 <= seed <= 2**32 - 1 for seed in (args.seeds or [args.seed])):
        parser.error("seeds must be integers from 0 through 4294967295")
    if args.seeds and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.no_affinity and (args.control_cpu_core is not None or args.trainer_cpu_core is not None):
        parser.error("--no-affinity cannot be combined with CPU overrides")
    if any(cpu is not None and cpu < 0 for cpu in (args.control_cpu_core, args.trainer_cpu_core)):
        parser.error("CPU indices must be nonnegative")
    return args


def selected_runs(args, config):
    methods = list(config["methods"]) if args.controller == "all" else [args.controller]
    scenarios = list(config["scenarios"]) if args.scenario == "all" else [args.scenario]
    control_cpu = args.control_cpu_core if args.control_cpu_core is not None else config["cpu_affinity"]["control"]
    trainer_cpu = args.trainer_cpu_core if args.trainer_cpu_core is not None else config["cpu_affinity"]["trainer"]
    for seed in args.seeds or [args.seed]:
        for scenario in scenarios:
            for controller in methods:
                method = config["methods"][controller]
                output = args.output_dir.resolve() / controller / scenario / f"seed{seed}"
                flags = list(config["common_args"])
                if args.duration is not None:
                    flags[flags.index("--duration") + 1] = str(args.duration)
                flags += config["scenarios"][scenario]["args"] + method["args"]
                scale_disturbances(flags, args.wind_scale, args.turbulence_scale)
                flags, sim2real = apply_quadrotor_config(flags, getattr(args, "sim2real_profile", None))
                flags += ["--seed", str(seed), "--result-tag", args.result_tag, "--output-dir", str(output)]
                affinity = {}
                if not args.no_affinity:
                    flags += ["--control-cpu-core", str(control_cpu)]
                    affinity["control"] = control_cpu
                    if controller == "t2s":
                        flags += ["--trainer-cpu-core", str(trainer_cpu)]
                        affinity["trainer"] = trainer_cpu
                yield {"controller": controller, "scenario": scenario, "seed": seed,
                       "module": method["module"], "output_dir": str(output),
                       "cpu_affinity": affinity,
                       "disturbance_scales": {"mean_wind": args.wind_scale, "turbulence": args.turbulence_scale},
                       "sim2real": sim2real, "args": flags}


def runtime_environment(config):
    env = os.environ.copy()
    for variable in THREAD_VARIABLES:
        env[variable] = str(config["numerical_threads"])
    acados = Path(env.get("ACADOS_SOURCE_DIR", ROOT / "external" / "acados")).expanduser().resolve()
    env["ACADOS_SOURCE_DIR"] = str(acados)
    library_paths = [str(acados / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return env


def validate_runtime(runs, env):
    acados = Path(env["ACADOS_SOURCE_DIR"])
    if not (acados / "lib").is_dir():
        raise SystemExit(f"Missing acados build at {acados}. Build external/acados or set ACADOS_SOURCE_DIR.")
    available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    for run in runs:
        output = Path(run["output_dir"])
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise SystemExit(f"Preserving existing output: {output}. Choose another --output-dir.")
        if available is not None:
            for role, cpu in run["cpu_affinity"].items():
                if cpu not in available:
                    raise SystemExit(f"Requested {role} CPU {cpu} is unavailable. Use CPU overrides or --no-affinity.")


def worker(args, config, runs):
    if len(runs) != 1:
        raise SystemExit("Internal worker requires exactly one run")
    run = runs[0]
    # Imports are delayed until execution; this keeps inspection usable on a clean checkout.
    sys.path.insert(0, str(ROOT))
    experiment = importlib.import_module(run["module"])
    output = Path(run["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(run, config=str(args.config.resolve()), numerical_threads=config["numerical_threads"],
                    note="Public staging implementation; numerical paper reproduction has not been verified.")
    (output / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sys.argv = [str(Path(__file__).resolve()), *run["args"]]
    experiment.main(run["controller"])


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit("Unsupported quadrotor configuration schema")
    runs = list(selected_runs(args, config))
    env = runtime_environment(config)
    if args.dry_run:
        print(json.dumps({"runs": runs, "environment": {key: env[key] for key in (*THREAD_VARIABLES, "ACADOS_SOURCE_DIR", "LD_LIBRARY_PATH")}}, indent=2))
        return
    if args._worker:
        worker(args, config, runs)
        return
    validate_runtime(runs, env)
    for run in runs:
        command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
                   "--controller", run["controller"], "--scenario", run["scenario"],
                   "--seed", str(run["seed"]), "--config", str(args.config.resolve()),
                   "--output-dir", str(args.output_dir.resolve()), "--result-tag", args.result_tag,
                   "--wind-scale", str(args.wind_scale), "--turbulence-scale", str(args.turbulence_scale)]
        if args.sim2real_profile is not None:
            command += ["--sim2real-config", args.sim2real_profile["path"],
                        "--_sim2real-sha256", args.sim2real_profile["sha256"]]
        if args.duration is not None:
            command += ["--duration", str(args.duration)]
        if args.no_affinity:
            command += ["--no-affinity"]
        else:
            command += ["--control-cpu-core", str(run["cpu_affinity"]["control"])]
            if run["controller"] == "t2s":
                command += ["--trainer-cpu-core", str(run["cpu_affinity"]["trainer"])]
        print(f"Running {run['controller']} / {run['scenario']} / seed {run['seed']}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
