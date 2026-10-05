#!/usr/bin/env bash
# Install the recorded dependency stack into the active Python environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/setup.sh [--system quadrotor|go2|all] [--dry-run]

Install the selected simulator and shared MPC dependencies (default: quadrotor).
Run inside an active Python 3.10 conda environment or virtual environment.
Linux x86-64, Git, Make, and C/C++ compilers are required. Network access is
needed for package downloads and pinned source checkouts.

  --system NAME  quadrotor, go2, or all
  --dry-run      Print commands without installing, building, or changing files.
                 An active Python 3.10 environment is not required for preview.
  -h, --help     Show this help.

Versions come from requirements/ and docs/dependencies.json. Existing external
checkouts must be clean and at the pinned revision; they are never reset.
The setup uses no sudo, global installs, shell-profile changes, or Go2 patches.
HELP
}
fail() { printf 'setup: %s\n' "$*" >&2; exit 1; }
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if [[ "$dry_run" == false ]]; then "$@"; fi
}

system=quadrotor
dry_run=false
while (($#)); do
    case "$1" in
        --system) (($# >= 2)) || fail '--system requires a value'
                  system=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1 (use --help)" ;;
    esac
done
case "$system" in quadrotor|go2|all) ;; *) fail "unknown system: $system" ;; esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
python=$(command -v python) || fail 'Python is unavailable; create and activate environment.yaml first'

# Inspect the active interpreter and manifest without importing project code.
dependencies=$("$python" -B - "$root" "$system" "$dry_run" <<'PYTHON'
import json, os, sys
from pathlib import Path
root, system, dry_run = sys.argv[1:]
if dry_run == "false":
    prefix = Path(sys.prefix).resolve()
    conda = os.environ.get("CONDA_PREFIX")
    in_conda = bool(conda and Path(conda).resolve() == prefix
                    and (prefix / "conda-meta").is_dir()
                    and not (prefix / "condabin").exists()
                    and os.environ.get("CONDA_DEFAULT_ENV") != "base")
    in_venv = sys.prefix != sys.base_prefix
    if sys.version_info[:2] != (3, 10) or not (in_conda or in_venv):
        sys.exit("setup: activate a non-base Python 3.10 conda environment or venv first")
selected = ["acados", "l4acados"]
if system in ("quadrotor", "all"):
    selected.append("safe-control-gym")
if system in ("go2", "all"):
    selected.append("go2-convex-mpc")
manifest = json.loads((Path(root) / "docs/dependencies.json").read_text())
by_name = {item["name"]: item for item in manifest["dependencies"]}
for name in selected:
    item = by_name[name]
    print(name, item["repository"], item["commit"], sep="\t")
PYTHON
)

if [[ "$dry_run" == false ]]; then
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'this recorded build targets Linux x86-64'
    for tool in git make cc c++; do command -v "$tool" >/dev/null || fail "missing build tool: $tool"; done
fi
[[ ! -L "$root/external" ]] || fail 'external/ is a symlink; use a regular directory'
export GIT_OPTIONAL_LOCKS=0
# Check every existing checkout before making any changes, including pip installs.
while IFS=$'\t' read -r name url commit; do
    path="$root/external/$name"
    if [[ -e "$path" || -L "$path" ]]; then
        [[ -d "$path" && ! -L "$path" ]] || fail "$path is not a regular directory"
        top=$(git -C "$path" rev-parse --show-toplevel 2>/dev/null) || fail "$path is not a Git checkout"
        [[ "$top" == "$path" ]] || fail "$path is not a standalone Git checkout"
        [[ $(git -C "$path" rev-parse HEAD) == "$commit" ]] || fail "$name is not at pinned revision $commit; preserve it elsewhere before setup"
        [[ -z $(git -C "$path" status --porcelain --untracked-files=normal --ignore-submodules=none) ]] || fail "$name has local changes; preserve them before setup"
    fi
done <<< "$dependencies"

printf 'Setting up %s with %s%s\n' "$system" "$python" "$([[ "$dry_run" == true ]] && printf ' (dry run)' || true)"
run mkdir -p "$root/external"
while IFS=$'\t' read -r name url commit; do
    path="$root/external/$name"
    if [[ ! -d "$path" ]]; then
        run git clone --no-checkout "$url" "$path"
        run git -C "$path" checkout --detach "$commit"
    fi
done <<< "$dependencies"

# PyTorch must be installed before L4CasADi builds against it.
prefix=$("$python" -B -c 'import sys; print(sys.prefix)')
pip_install=("$python" -m pip --isolated install --no-user --prefix "$prefix")
run "${pip_install[@]}" -r "$root/requirements/core.txt"
run "${pip_install[@]}" setuptools==83.0.0 scikit-build==0.18.1 cmake==3.31.6 ninja==1.11.1.3 wheel==0.45.1
requirements=()
if [[ "$system" == quadrotor || "$system" == all ]]; then requirements+=(-r "$root/requirements/quadrotor.txt"); fi
if [[ "$system" == go2 || "$system" == all ]]; then requirements+=(-r "$root/requirements/go2.txt"); fi
run "${pip_install[@]}" --no-build-isolation "${requirements[@]}"

acados="$root/external/acados"
run git -C "$acados" submodule update --init --recursive
run cmake -S "$acados" -B "$acados/build" \
    "-DCMAKE_INSTALL_PREFIX=$acados" -DACADOS_WITH_QPOASES=ON \
    -DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_AUTOMATIC -DHPIPM_TARGET=GENERIC
run cmake --build "$acados/build" --parallel
run cmake --install "$acados/build"
run "${pip_install[@]}" --no-deps -e "$acados/interfaces/acados_template"
run "${pip_install[@]}" --no-deps -e "$root/external/l4acados"
if [[ "$system" == quadrotor || "$system" == all ]]; then
    run "${pip_install[@]}" --no-deps -e "$root/external/safe-control-gym"
fi
if [[ "$system" == go2 || "$system" == all ]]; then
    run "${pip_install[@]}" --no-deps -e "$root/external/go2-convex-mpc"
fi
if [[ "$dry_run" == true ]]; then
    printf 'Dry run complete; no files changed and no install commands executed.\n'
else
    printf 'Setup complete. Public experiment runners configure native library paths automatically.\n'
fi
