"""Check the public disturbance CLI without importing a simulator."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_quadrotor


RUNNER = Path(run_quadrotor.__file__).resolve()


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-S", str(RUNNER), *args, "--dry-run"],
        capture_output=True, text=True, check=False,
    )


def runs(*args):
    result = invoke(*args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["runs"]


def flag_value(run, flag):
    values = run["args"]
    return values[values.index(flag) + 1]


def test_default_and_explicit_unit_scales_preserve_paper_arguments():
    options = ("--method", "all", "--scenario", "all")
    defaults = runs(*options)
    explicit = runs(*options, "--wind-scale", "1", "--turbulence-scale", "1")
    assert defaults == explicit
    assert len(defaults) == 12
    for run in defaults:
        assert flag_value(run, "--mean-wind-x") == "0.75"
        assert flag_value(run, "--turbulence-sigma-x") == "0.175"
        assert flag_value(run, "--measurement-delay-steps") == "1"
        assert flag_value(run, "--control-frequency") == "50"
        assert flag_value(run, "--wind-gradient-x") == "4"


def test_scales_apply_to_all_axes_and_scenarios_without_changing_other_parameters():
    options = ("--method", "all", "--scenario", "all", "--seed", "123")
    baseline = runs(*options)
    adjusted = runs(*options, "--wind-scale", "2", "--turbulence-scale", "0.5")
    expected_flags = {
        f"--{prefix}-{axis}{suffix}": scale
        for prefix, scale in (("mean-wind", 2), ("turbulence-sigma", 0.5))
        for axis in "xyz" for suffix in ("", "-end")
    }
    for original, scaled in zip(baseline, adjusted):
        assert scaled["seed"] == 123
        assert scaled["disturbance_scales"] == {"mean_wind": 2, "turbulence": 0.5}
        assert len(original["args"]) == len(scaled["args"])
        changed_indices = set()
        for flag, scale in expected_flags.items():
            index = original["args"].index(flag) + 1
            assert float(scaled["args"][index]) == pytest.approx(float(original["args"][index]) * scale)
            changed_indices.add(index)
        for index, value in enumerate(original["args"]):
            if index not in changed_indices:
                assert scaled["args"][index] == value
    combined = next(run for run in adjusted if run["controller"] == "t2s" and run["scenario"] == "combined")
    assert float(flag_value(combined, "--mean-wind-x-end")) == 3.0
    assert float(flag_value(combined, "--turbulence-sigma-x-end")) == 0.175


def test_zero_scales_remove_imposed_wind_and_turbulence_endpoints():
    run = runs("--scenario", "combined", "--wind-scale", "0", "--turbulence-scale", "0")[0]
    for prefix in ("mean-wind", "turbulence-sigma"):
        for axis in "xyz":
            for suffix in ("", "-end"):
                assert float(flag_value(run, f"--{prefix}-{axis}{suffix}")) == 0
    assert flag_value(run, "--wind-gradient-y") == "-4"
    assert flag_value(run, "--advection-speed") == "1.5"
    assert flag_value(run, "--wind-ramp-duration") == "20"


@pytest.mark.parametrize("option", ["--wind-scale", "--turbulence-scale"])
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_invalid_scale_is_rejected_before_runtime(option, value):
    result = invoke(f"{option}={value}")
    assert result.returncode == 2
    assert "finite and nonnegative" in result.stderr
    assert not result.stdout



@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_duration_is_rejected_before_runtime(value):
    result = invoke(f"--duration={value}")
    assert result.returncode == 2
    assert "--duration must be finite and positive" in result.stderr
    assert not result.stdout


def test_positive_finite_duration_reaches_experiment_arguments():
    run = runs("--duration", "0.5")[0]
    assert float(flag_value(run, "--duration")) == 0.5


def test_scaled_parameter_overflow_is_rejected():
    result = invoke("--scenario", "combined", "--wind-scale", "1.7e308")
    assert result.returncode != 0
    assert "nonfinite value" in result.stderr


@pytest.mark.parametrize("value", ["-1", "4294967296"])
def test_invalid_numpy_seed_is_rejected(value):
    result = invoke("--seed", value)
    assert result.returncode == 2
    assert "0 through 4294967295" in result.stderr


def test_scales_are_forwarded_to_real_worker_arguments(monkeypatch, tmp_path):
    execute = subprocess.run
    captured = []
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--method", "ssi", "--scenario", "combined",
                                     "--seed", "123", "--wind-scale", "1.5", "--turbulence-scale", "0.25",
                                     "--output-dir", str(tmp_path), "--no-affinity"])
    monkeypatch.setattr(run_quadrotor, "validate_runtime", lambda *_: None)
    monkeypatch.setattr(run_quadrotor.subprocess, "run", lambda command, **kwargs: captured.append(command))
    run_quadrotor.main()
    assert len(captured) == 1
    command = captured[0]
    assert "--_worker" in command
    # Exercise the worker's real CLI construction, stopping at dry-run before imports.
    result = execute([*command, "--dry-run"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    run = json.loads(result.stdout)["runs"][0]
    assert float(flag_value(run, "--mean-wind-x-end")) == 2.25
    assert float(flag_value(run, "--turbulence-sigma-x-end")) == 0.0875
    assert run["disturbance_scales"] == {"mean_wind": 1.5, "turbulence": 0.25}
    assert not list(tmp_path.iterdir())
