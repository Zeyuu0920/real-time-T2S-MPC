#!/usr/bin/env python3
"""Run the separate Go2 sensor/actuator stress test; the paper runner is unchanged."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.go2_sim2real_config import PROTOCOL, load_profile, timing_metadata
from scripts.run_go2 import selected_config, prepare_environment, apply_disturbance_overrides

SOLVER_SUFFIX = "_public_sim2real_intra_v1"


def sim2real_controller_process(connection, config, cpu_core=None, training_core=None):
    """Spawn-safe selection of the original intra-period controller and cone."""
    import src.go2_realism_controller as controller
    import src.go2_paper_acados as acados
    original = acados.PaperAcadosMPC

    def build(**kwargs):
        kwargs["name_suffix"] += SOLVER_SUFFIX
        return original(**kwargs)

    with patch.object(acados, "PAPER_FRICTION_COEFFICIENT", .5), \
         patch.object(controller, "PaperAcadosMPC", side_effect=build):
        controller.controller_process(connection, config, cpu_core, training_core)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim2real-config", type=Path, required=True,
                        help="explicit JSON observation/actuator profile for this independent mode")
    parser.add_argument("--method", choices=("nominal", "ssi", "stgp", "t2s"), default="t2s")
    parser.add_argument("--scenario", choices=("drain", "friction", "combined"), default="combined")
    parser.add_argument("--seed", type=int, choices=range(42, 52), default=42)
    parser.add_argument("--duration", type=float, default=60.,
                        help="positive simulation duration in 0.02 s increments (default: 60)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/go2_sim2real")
    parser.add_argument("--payload-mass", type=float, metavar="KG", help="positive rigid mass; friction only")
    parser.add_argument("--liquid-mass-scale", type=float, metavar="FACTOR", help="positive liquid mass multiplier; drain/combined")
    parser.add_argument("--friction-min", type=float, metavar="MU", help="nonnegative sliding lower bound; friction/combined")
    parser.add_argument("--friction-max", type=float, metavar="MU", help="sliding upper bound >= lower; friction/combined")
    parser.add_argument("--ground-friction", type=float, metavar="MU", help="nonnegative fixed sliding coefficient; drain only")
    parser.add_argument("--dry-run", action="store_true", help="show effective settings without numerical dependencies or simulation")
    parser.add_argument("--_profile-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    intervals = args.duration * 50
    if not math.isfinite(intervals) or intervals <= 0 or not math.isclose(intervals, round(intervals), rel_tol=0, abs_tol=1e-9):
        parser.error("--duration must be finite, positive, and an integer multiple of 0.02 s")
    try:
        options, profile = load_profile(args.sim2real_config, args._profile_sha256)
        config, manifest = selected_config(args.method, args.scenario, args.seed)
        config = replace(config, duration=args.duration)
        config, settings, disturbances = apply_disturbance_overrides(
            config, payload_mass=args.payload_mass, liquid_mass_scale=args.liquid_mass_scale,
            friction_min=args.friction_min, friction_max=args.friction_max,
            ground_friction=args.ground_friction)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args, config, options, profile, manifest, settings, disturbances


def main():
    args, config, options, profile, manifest, settings, disturbances = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    case = f"{config.scenario}_{config.method}_seed{config.seed}_sim2real_delay{20*options.observation_delay_steps}ms_{config.duration:g}s"
    output = output_root / case
    metadata = {"protocol": PROTOCOL, "base_protocol": manifest["source_protocol"],
                "config": asdict(config), "sim2real_profile": profile,
                "timing": timing_metadata(options), "disturbances": disturbances,
                "mpc_friction_coefficient": .5,
                "field": manifest["fields"][str(config.seed)],
                "output_dir": str(output), "fresh_run": True}
    if args.dry_run:
        print(json.dumps(metadata, indent=2, allow_nan=False))
        return
    if output.exists():
        raise SystemExit(f"Preserving existing output: {output}. Choose another --output-dir.")
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < 3:
        raise SystemExit("The Go2 sim2real driver requires at least three available CPU cores.")
    # prepare_environment may re-exec to configure the dynamic loader. Pin the
    # already validated profile bytes across that process boundary.
    if args._profile_sha256 is None:
        sys.argv += ["--_profile-sha256", profile["sha256"]]
    prepare_environment()
    os.chdir(ROOT)
    from src.go2_public_protocol import frozen_protocol
    from src.go2_realism_experiment import run_trial
    from src.go2_icra_experiment import dump_json
    sources = sorted((ROOT / "src").glob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/run_go2.py"]
    metadata["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    metadata["base_config_sha256"] = hashlib.sha256((ROOT / "configs/go2/final.json").read_bytes()).hexdigest()
    try:
        with frozen_protocol(**settings):
            metrics = run_trial(config, options, output, sim2real_controller_process, run_metadata=metadata)
    except BaseException as exc:
        # Initialization failures precede the driver's controller-loop handler.
        # Keep their cause in the newly owned output too.
        if not isinstance(exc, FileExistsError) and output.is_dir() and not (output / "error.json").exists():
            dump_json(output / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    kind = config.method if config.method in ("nominal", "ssi") else "t2s"
    suffix = "_xu24_te32" if config.method == "t2s" else ""
    name = f"go2_paper_fe_acados_{kind}_icra_v2_wrench_{config.method}_seed{config.seed}{suffix}{SOLVER_SUFFIX}"
    shutil.copy2(ROOT / f"{name}.json", output / "effective_ocp.json")
    print(json.dumps({"output": str(output), "protocol": PROTOCOL,
                      "status": metrics["status"], "simulated_seconds": metrics["simulated_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
