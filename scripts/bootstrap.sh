#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
INSTALL_REQUIREMENTS=1
INSTALL_TORCH=1
INSTALL_OPEN3D_WHEEL=1
OFFLINE=0

WITH_ACCEL=1
WITH_GDEL3D=1
WITH_PYTORCH3D=1
WITH_KAOLIN=1

TORCH_VARIANT="${TORCH_VARIANT:-auto}" # auto|cpu|cu118|cu124|cu126
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
OPEN3D_PACKAGE="${OPEN3D_PACKAGE:-open3d-cpu}"

require_python_headers() {
  local py_bin="$1"
  local include_dir
  include_dir="$("$py_bin" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include"))
PY
)"
  if [[ -z "$include_dir" || ! -f "$include_dir/Python.h" ]]; then
    cat <<EOF >&2
[bootstrap] Missing Python headers (Python.h).
Install your system Python dev package, e.g.:
  sudo apt-get install python3.12-dev
Then re-run: pip install -e accel
EOF
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  local hint="${2:-}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[bootstrap] Missing dependency: $cmd" >&2
    if [[ -n "$hint" ]]; then
      echo "$hint" >&2
    fi
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [options]

Creates/uses a venv, installs base deps (including PyTorch), updates git submodules, applies local patches, and installs local editable packages.

Options:
  --venv <dir>            venv directory (default: .venv)
  --python <exe>          python executable to create venv (default: python3.12)
  --skip-torch            do not install torch/torchvision
  --skip-requirements     do not run `pip install -r requirements.txt`
  --skip-open3d-wheel     do not install an Open3D wheel
  --offline               imply `--skip-requirements` and use `pip --no-deps` for local installs

  --torch <variant>       one of: auto, cpu, cu118, cu124, cu126 (default: auto)
  --torch-version <ver>   torch version (default: 2.7.1)
  --torchvision-version <ver> torchvision version (default: 0.22.1)
  --open3d-package <pkg>  wheel name (default: open3d-cpu)

  --with-gdel3d           install `pygdel3d` (requires nvcc / CUDA toolchain)
  --with-pytorch3d        install `pytorch3d` (requires torch)
  --with-kaolin           install `kaolin` (requires torch)
  --with-all              enable all of the above

Environment variables:
  VENV_DIR, PYTHON_BIN    override defaults (same as options)
  TORCH_VARIANT           override --torch (auto/cpu/cu118/cu124/cu126)
  TORCH_VERSION           override --torch-version
  TORCHVISION_VERSION     override --torchvision-version
  TORCH_INDEX_URL         override computed index URL (e.g. https://download.pytorch.org/whl/cu126)
  OPEN3D_PACKAGE          override --open3d-package
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-requirements)
      INSTALL_REQUIREMENTS=0
      shift
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --skip-open3d-wheel)
      INSTALL_OPEN3D_WHEEL=0
      shift
      ;;
    --offline)
      OFFLINE=1
      INSTALL_REQUIREMENTS=0
      shift
      ;;
    --torch)
      TORCH_VARIANT="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    --open3d-package)
      OPEN3D_PACKAGE="$2"
      shift 2
      ;;
    --with-gdel3d)
      WITH_GDEL3D=1
      shift
      ;;
    --with-pytorch3d)
      WITH_PYTORCH3D=1
      shift
      ;;
    --with-kaolin)
      WITH_KAOLIN=1
      shift
      ;;
    --with-all)
      WITH_GDEL3D=1
      WITH_PYTORCH3D=1
      WITH_KAOLIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v git >/dev/null 2>&1; then
  echo "Missing dependency: git" >&2
  exit 1
fi

echo "[bootstrap] Updating submodules..."
git -C "$ROOT" submodule update --init --recursive

torch_index_url_for_variant() {
  local variant="$1"
  case "$variant" in
    cpu) echo "https://download.pytorch.org/whl/cpu" ;;
    cu118) echo "https://download.pytorch.org/whl/cu118" ;;
    cu124) echo "https://download.pytorch.org/whl/cu124" ;;
    cu126) echo "https://download.pytorch.org/whl/cu126" ;;
    *)
      echo "" >&2
      return 1
      ;;
  esac
}

detect_torch_variant() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi_out
    smi_out="$(nvidia-smi 2>/dev/null || true)"
    # Example: "CUDA Version: 13.0"
    local smi_release
    smi_release="$(printf '%s' "$smi_out" | sed -n 's/.*CUDA Version: \\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p' | head -n 1)"
    local smi_major="${smi_release%.*}"
    local smi_minor="${smi_release#*.}"
    if [[ -n "$smi_major" && -n "$smi_minor" ]]; then
      if [[ "$smi_major" -eq 13 ]]; then
        echo "cu126"
        return 0
      fi
      if [[ "$smi_major" -eq 12 && "$smi_minor" -ge 6 ]]; then
        echo "cu126"
        return 0
      fi
      if [[ "$smi_major" -eq 12 && "$smi_minor" -ge 4 ]]; then
        echo "cu124"
        return 0
      fi
      if [[ "$smi_major" -eq 11 ]]; then
        echo "cu118"
        return 0
      fi
    fi
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_out
    nvcc_out="$(nvcc -V 2>/dev/null || true)"
    # Example: "Cuda compilation tools, release 12.6, V12.6.85"
    local release
    release="$(printf '%s' "$nvcc_out" | sed -n 's/.*release \\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p' | head -n 1)"
    local major="${release%.*}"
    local minor="${release#*.}"
    if [[ -n "$major" && -n "$minor" ]]; then
      if [[ "$major" -eq 12 && "$minor" -ge 6 ]]; then
        echo "cu126"
        return 0
      fi
      if [[ "$major" -eq 12 && "$minor" -ge 4 ]]; then
        echo "cu124"
        return 0
      fi
      if [[ "$major" -eq 11 ]]; then
        echo "cu118"
        return 0
      fi
    fi
  fi
  echo "cpu"
}

if [[ ! -d "$VENV_DIR" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN or pass --python (e.g. --python python3.12)" >&2
    exit 1
  fi
  echo "[bootstrap] Creating venv at: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
PIP="$VENV_PY -m pip"
export PATH="$VENV_DIR/bin:$PATH"

echo "[bootstrap] Upgrading pip tooling..."
$PIP install -U pip setuptools wheel

if [[ $INSTALL_TORCH -eq 1 ]]; then
  if [[ $OFFLINE -eq 1 ]]; then
    echo "[bootstrap] Skipping torch install (offline mode)."
  else
    if [[ -z "$TORCH_INDEX_URL" ]]; then
      torch_variant_resolved="$TORCH_VARIANT"
      if [[ "$torch_variant_resolved" == "auto" ]]; then
        torch_variant_resolved="$(detect_torch_variant)"
      fi
      TORCH_INDEX_URL="$(torch_index_url_for_variant "$torch_variant_resolved")"
    fi
    echo "[bootstrap] Installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} from: $TORCH_INDEX_URL"
    $PIP install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
  fi
else
  echo "[bootstrap] Skipping torch install."
fi

if [[ $INSTALL_REQUIREMENTS -eq 1 ]]; then
  echo "[bootstrap] Installing requirements.txt..."
  $PIP install -r "$ROOT/requirements.txt"
else
  echo "[bootstrap] Skipping requirements.txt install."
fi

if [[ $INSTALL_OPEN3D_WHEEL -eq 1 ]]; then
  if [[ $OFFLINE -eq 1 ]]; then
    echo "[bootstrap] Skipping Open3D wheel install (offline mode)."
  else
    echo "[bootstrap] Installing Open3D wheel: $OPEN3D_PACKAGE"
    $PIP install "$OPEN3D_PACKAGE"
  fi
else
  echo "[bootstrap] Skipping Open3D wheel install."
fi

echo "[bootstrap] Applying local patches..."
bash "$ROOT/scripts/apply_patches.sh"

PIP_LOCAL_FLAGS=()
if [[ $OFFLINE -eq 1 ]]; then
  PIP_LOCAL_FLAGS+=(--no-build-isolation --no-deps)
fi

if [[ $WITH_ACCEL -eq 1 ]]; then
  echo "[bootstrap] Installing accel (voronoiaccel)..."
  require_python_headers "$VENV_PY"
  $PIP install -e "$ROOT/accel" "${PIP_LOCAL_FLAGS[@]}"
fi

if [[ $WITH_GDEL3D -eq 1 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[bootstrap] Skipping gDel3D: nvcc not found in PATH."
  else
    require_cmd cmake "Install cmake (e.g. sudo apt-get install cmake) or ensure it's in PATH."
    echo "[bootstrap] Installing gDel3D python bindings (pygdel3d)..."
    # gDel3D's pyproject.toml does not declare cmake; avoid build isolation.
    $PIP install -e "$ROOT/3rdparty/gDel3D/python_bindings" --no-build-isolation "${PIP_LOCAL_FLAGS[@]}"
  fi
fi

if [[ $WITH_PYTORCH3D -eq 1 ]]; then
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "[bootstrap] Skipping pytorch3d: torch is not installed in the venv."
  else
    echo "[bootstrap] Installing pytorch3d..."
    # pytorch3d's setup.py imports torch during build.
    $PIP install -e "$ROOT/3rdparty/pytorch3d" --no-build-isolation "${PIP_LOCAL_FLAGS[@]}"
  fi
fi

if [[ $WITH_KAOLIN -eq 1 ]]; then
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "[bootstrap] Skipping kaolin: torch is not installed in the venv."
  else
    echo "[bootstrap] Installing kaolin..."
    # kaolin's setup.py imports torch during build.
    $PIP install -e "$ROOT/3rdparty/kaolin" --no-build-isolation "${PIP_LOCAL_FLAGS[@]}"
  fi
fi

cat <<EOF

[bootstrap] Done.
- Activate: source "$VENV_DIR/bin/activate"
EOF
