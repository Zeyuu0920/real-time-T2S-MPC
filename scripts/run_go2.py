#!/usr/bin/env python3
"""Run a fresh Go2 trial from the selected 60-second, cone-0.5 protocol.

Help and --dry-run use only Python's standard library. No historical results
are required. Simulations require the numerical dependencies and Go2 assets.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs/go2/final.json"


def selected_config(method, scenario, seed):
    from src.go2_icra_protocol import Go2ICRAConfig, Go2ICRATimeReplayT2SConfig
    manifest = json.loads(CONFIG_PATH.read_text())
    raw = manifest["configs"][f"seed{seed}/{scenario}/{method}"]
    cls = Go2ICRATimeReplayT2SConfig if method == "t2s" else Go2ICRAConfig
    config = cls(**raw)
    config.validate()
    if json.loads(json.dumps(asdict(config))) != raw:
        raise ValueError("Configuration does not round-trip exactly")
    field = manifest["fields"][str(seed)]
    digest = hashlib.sha256((ROOT / field["path"]).read_bytes()).hexdigest()
    if digest != field["sha256"]:
        raise ValueError("Frozen friction-map checksum mismatch")
    return config, manifest



def apply_disturbance_overrides(config, *, payload_mass=None, liquid_mass_scale=None,
                               friction_min=None, friction_max=None, ground_friction=None):
    """Validate scene-specific options before loading numerical dependencies."""
    requested = {name: value for name, value in {
        "payload_mass": payload_mass, "liquid_mass_scale": liquid_mass_scale,
        "friction_min": friction_min, "friction_max": friction_max,
        "ground_friction": ground_friction,
    }.items() if value is not None}
    for name, value in requested.items():
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if payload_mass is not None:
        if config.scenario != "friction":
            raise ValueError("--payload-mass applies only to --scenario friction; use --liquid-mass-scale for liquid scenes")
        if payload_mass <= 0:
            raise ValueError("--payload-mass must be positive (kg)")
        config = replace(config, payload_mass=payload_mass)
    if liquid_mass_scale is not None:
        if not config.liquid:
            raise ValueError("--liquid-mass-scale applies only to drain or combined")
        if liquid_mass_scale <= 0:
            raise ValueError("--liquid-mass-scale must be positive")
    if friction_min is not None or friction_max is not None:
        if not config.variable_friction:
            raise ValueError("--friction-min/--friction-max apply only to friction or combined; use --ground-friction for drain")
    if ground_friction is not None and config.scenario != "drain":
        raise ValueError("--ground-friction applies only to --scenario drain")
    settings = {
        "liquid_mass_scale": 1.0 if liquid_mass_scale is None else liquid_mass_scale,
        "friction_min": .5 if friction_min is None else friction_min,
        "friction_max": .8 if friction_max is None else friction_max,
        "ground_friction": .8 if ground_friction is None else ground_friction,
    }
    if not 0 <= settings["friction_min"] <= settings["friction_max"]:
        raise ValueError("friction bounds must satisfy 0 <= --friction-min <= --friction-max")
    if settings["ground_friction"] < 0:
        raise ValueError("--ground-friction must be nonnegative")
    # Density is the physical multiplier: the tank geometry and drain timing
    # stay fixed while liquid mass, inertia, spring and damping scale together.
    scale = settings["liquid_mass_scale"]
    if not math.isfinite(1000.0 * scale):
        raise ValueError("--liquid-mass-scale is too large for a finite density")
    config.validate()
    payload = ({"type": "draining_liquid", "container_mass_kg": .6,
                "liquid_mass_scale": scale, "liquid_density_kg_m3": 1000.0 * scale,
                "initial_liquid_mass_kg": 3.4 * scale, "final_liquid_mass_kg": 1.4 * scale,
                "initial_total_mass_kg": .6 + 3.4 * scale,
                "final_total_mass_kg": .6 + 1.4 * scale,
                "drain_start_s": 5., "drain_end_s": 45.}
               if config.liquid else {"type": "rigid", "mass_kg": config.payload_mass})
    ground = ({"type": "spatial", "sliding_min": settings["friction_min"],
               "sliding_max": settings["friction_max"],
               "mapping": "affine rescaling of frozen [.5, .8] sliding nodes",
               "torsional_rolling": "unchanged frozen spatial field"}
              if config.variable_friction else {"type": "fixed",
                  "sliding": settings["ground_friction"], "torsional": .02, "rolling": .01})
    return config, settings, {"requested_overrides": requested,
                               "payload": payload, "ground": ground}


def prepare_environment():
    acados = Path(os.environ.get("ACADOS_SOURCE_DIR", ROOT / "external/acados")).expanduser().resolve()
    go2 = Path(os.environ.get("GO2_CONVEX_MPC_DIR", ROOT / "external/go2-convex-mpc")).expanduser().resolve()
    required = [acados / "lib/libacados.so",
                go2 / "src/convex_mpc/go2_robot_data.py",
                go2 / "models/MJCF/go2/scene.xml",
                go2 / "models/URDF/go2_description/urdf/go2_description.urdf"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Go2 dependencies; follow the installation guide: " + ", ".join(missing))
    env = os.environ.copy()
    env["ACADOS_SOURCE_DIR"] = str(acados)
    env["GO2_CONVEX_MPC_DIR"] = str(go2)
    env.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    lib = str(acados / "lib")
    libraries = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if lib not in libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([lib, *filter(None, libraries)])
    # The dynamic loader and BLAS must see these settings at process startup.
    if any(os.environ.get(key) != value for key, value in env.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    for path in (go2 / "src", acados / "interfaces/acados_template"):
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("nominal", "ssi", "stgp", "t2s"), default="t2s")
    parser.add_argument("--scenario", choices=("drain", "friction", "combined"), default="combined")
    parser.add_argument("--seed", type=int, choices=range(42, 52), default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/go2")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the effective configuration without simulation")
    parser.add_argument("--payload-mass", type=float, metavar="KG",
                        help="positive rigid payload mass; friction scene only (default: 4 kg)")
    parser.add_argument("--liquid-mass-scale", type=float, metavar="FACTOR",
                        help="positive liquid-density/mass multiplier; drain/combined only (default: 1)")
    parser.add_argument("--friction-min", type=float, metavar="MU",
                        help="nonnegative lower sliding bound; friction/combined only (default: 0.5)")
    parser.add_argument("--friction-max", type=float, metavar="MU",
                        help="upper sliding bound, >= min; friction/combined only (default: 0.8)")
    parser.add_argument("--ground-friction", type=float, metavar="MU",
                        help="nonnegative fixed sliding coefficient; drain only (default: 0.8)")
    args = parser.parse_args()
    config, manifest = selected_config(args.method, args.scenario, args.seed)
    try:
        config, disturbance_settings, disturbances = apply_disturbance_overrides(
            config, payload_mass=args.payload_mass, liquid_mass_scale=args.liquid_mass_scale,
            friction_min=args.friction_min, friction_max=args.friction_max,
            ground_friction=args.ground_friction)
    except ValueError as exc:
        parser.error(str(exc))
    output_root = args.output_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"config": asdict(config), "mpc_friction_coefficient": .5,
                          "field": manifest["fields"][str(args.seed)]["path"],
                          "timing": manifest["timing"],
                          "disturbances": disturbances,
                          "output_dir": str(output_root)}, indent=2))
        return
    prepare_environment()
    # Keep relative compiler paths stable when launched outside the repository.
    os.chdir(ROOT)
    from src.go2_public_protocol import frozen_protocol
    from src.go2_icra_experiment import run_icra_experiment, dump_json
    with frozen_protocol(**disturbance_settings):
        metrics, output = run_icra_experiment(config, output_root)
    kind = config.method if config.method in ("nominal", "ssi") else "t2s"
    suffix = "_xu24_te32" if config.method == "t2s" else ""
    solver_name = f"go2_paper_fe_acados_{kind}_icra_v2_wrench_{config.method}_seed{config.seed}{suffix}_public_cone050_v1"
    shutil.copy2(ROOT / f"{solver_name}.json", output / "effective_ocp.json")
    dump_json(output / "release_protocol.json", {
        "protocol": manifest["source_protocol"],
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "mpc_friction_coefficient": .5,
        "adapter_sha256": hashlib.sha256((ROOT / "src/go2_public_protocol.py").read_bytes()).hexdigest(),
        "field": manifest["fields"][str(config.seed)],
        "fresh_run": True,
        "disturbances": disturbances,
    })
    print(json.dumps({"output": str(output), "status": metrics["status"],
                      "simulated_seconds": metrics["simulated_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
