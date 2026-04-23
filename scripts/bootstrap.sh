#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
BOOTSTRAP_USER="${USER:-$(id -u 2>/dev/null || printf user)}"
DCCVT_CACHE_DIR="${DCCVT_CACHE_DIR:-${TMPDIR:-/tmp}/dccvt-bootstrap-$BOOTSTRAP_USER}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DCCVT_CACHE_DIR/pip}"
DCCVT_TMPDIR="${DCCVT_TMPDIR:-$DCCVT_CACHE_DIR/tmp}"
DCCVT_MIN_VENV_FREE_GB="${DCCVT_MIN_VENV_FREE_GB:-10}"
DCCVT_MIN_CACHE_FREE_GB="${DCCVT_MIN_CACHE_FREE_GB:-20}"
INSTALL_REQUIREMENTS=1
INSTALL_TORCH=1
INSTALL_OPEN3D_WHEEL=1
OFFLINE=0

WITH_ACCEL=1
WITH_GDEL3D=1
WITH_PYTORCH3D=1
WITH_KAOLIN=1
BUILD_JOBS="${BUILD_JOBS:-}"

TORCH_VARIANT="${TORCH_VARIANT:-auto}" # auto|cu118|cu124|cu126
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
OPEN3D_PACKAGE="${OPEN3D_PACKAGE:-open3d-cpu}"

require_python_headers() {
  local py_bin="$1"
  local include_dir
  local py_series
  include_dir="$("$py_bin" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include"))
PY
)"
  py_series="$("$py_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ -z "$include_dir" || ! -f "$include_dir/Python.h" ]]; then
    cat <<EOF >&2
[bootstrap] Missing Python headers (Python.h).
Install your system Python dev package, e.g.:
  sudo apt-get install python${py_series}-dev
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

venv_python_path() {
  printf '%s\n' "$VENV_DIR/bin/python"
}

venv_has_pip() {
  local venv_py
  venv_py="$(venv_python_path)"
  [[ -x "$venv_py" ]] && "$venv_py" -m pip --version >/dev/null 2>&1
}

uv_available() {
  command -v uv >/dev/null 2>&1
}

run_uv() {
  uv --cache-dir "$UV_CACHE_DIR" "$@"
}

available_kb_for_path() {
  local path="$1"
  local probe="$path"
  if [[ ! -e "$probe" ]]; then
    probe="$(dirname "$probe")"
  fi
  while [[ ! -e "$probe" && "$probe" != "/" ]]; do
    probe="$(dirname "$probe")"
  done
  df -Pk "$probe" | awk 'NR == 2 { print $4 }'
}

warn_free_space() {
  local label="$1"
  local path="$2"
  local min_gb="$3"
  local avail_kb
  local avail_gb

  if ! [[ "$min_gb" =~ ^[0-9]+$ ]] || (( min_gb <= 0 )); then
    return 0
  fi

  avail_kb="$(available_kb_for_path "$path" 2>/dev/null || true)"
  if [[ -z "$avail_kb" ]]; then
    return 0
  fi

  avail_gb=$((avail_kb / 1024 / 1024))
  if (( avail_kb < min_gb * 1024 * 1024 )); then
    echo "[bootstrap] Warning: $label has about ${avail_gb}G free; CUDA wheel installs may need ${min_gb}G+." >&2
    echo "[bootstrap] To use another filesystem, set VENV_DIR, DCCVT_CACHE_DIR, PIP_CACHE_DIR, or DCCVT_TMPDIR." >&2
  fi
}

configure_pip_storage() {
  mkdir -p "$PIP_CACHE_DIR" "$DCCVT_TMPDIR"
  export PIP_CACHE_DIR
  export TMPDIR="$DCCVT_TMPDIR"
  echo "[bootstrap] Using pip cache: $PIP_CACHE_DIR"
  echo "[bootstrap] Using temp dir: $TMPDIR"
}

torch_version_pair_for_variant() {
  local variant="$1"
  case "$variant" in
    cu118|cu126)
      printf '%s %s\n' "2.7.1" "0.22.1"
      ;;
    cu124)
      printf '%s %s\n' "2.6.0" "0.21.0"
      ;;
    *)
      return 1
      ;;
  esac
}

uv_python_request() {
  case "$PYTHON_BIN" in
    python[0-9]*)
      printf '%s\n' "${PYTHON_BIN#python}"
      ;;
    *)
      printf '%s\n' "$PYTHON_BIN"
      ;;
  esac
}

ensure_pip_tool() {
  local cmd="$1"
  local package="${2:-$1}"

  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi

  if [[ $OFFLINE -eq 1 ]]; then
    echo "[bootstrap] Missing dependency: $cmd" >&2
    echo "[bootstrap] Offline mode is enabled, so $package cannot be installed automatically." >&2
    exit 1
  fi

  echo "[bootstrap] Installing build tool into the venv: $package"
  $PIP install -U "$package"

  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[bootstrap] Failed to provision $cmd via pip package $package." >&2
    exit 1
  fi
}

create_venv() {
  local stderr_log
  stderr_log="$(mktemp)"

  echo "[bootstrap] Creating venv at: $VENV_DIR"
  echo "[bootstrap] Preferred Python: $PYTHON_BIN"

  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>"$stderr_log"; then
      rm -f "$stderr_log"
      return 0
    fi
  else
    printf 'Python executable not found on PATH: %s\n' "$PYTHON_BIN" >"$stderr_log"
  fi

  local system_error
  system_error="$(cat "$stderr_log")"
  rm -f "$stderr_log"

  if uv_available; then
    local uv_python
    local -a uv_args
    uv_python="$(uv_python_request)"
    uv_args=(venv --seed --python "$uv_python" "$VENV_DIR")
    mkdir -p "$UV_CACHE_DIR"
    echo "[bootstrap] Falling back to uv-managed Python/venv creation."
    echo "[bootstrap] Using uv cache: $UV_CACHE_DIR"
    rm -rf "$VENV_DIR"
    if [[ $OFFLINE -eq 1 ]]; then
      uv_args=(venv --seed --offline --no-python-downloads --python "$uv_python" "$VENV_DIR")
    fi
    if run_uv "${uv_args[@]}"; then
      return 0
    fi
    echo "[bootstrap] uv fallback failed." >&2
  fi

  printf '%s\n' "$system_error" >&2
  cat <<EOF >&2
[bootstrap] Could not create a Python 3.12 virtual environment automatically.
Either install a working Python 3.12 with venv support, or install uv so bootstrap can provision one.
Examples:
  sudo apt-get install python3.12 python3.12-venv python3.12-dev
  pipx install uv
EOF
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [options]

Creates/uses a venv, installs base deps (including PyTorch), updates git submodules, applies local patches, and installs local editable packages.

Options:
  --venv <dir>            venv directory (default: .venv)
  --python <exe|request>  preferred Python for the venv (default: python3.12)
  --skip-torch            do not install torch/torchvision
  --skip-requirements     do not run `pip install -r requirements.txt`
  --skip-open3d-wheel     do not install an Open3D wheel
  --offline               imply `--skip-requirements` and use `pip --no-deps` for local installs

  --torch <variant>       one of: auto, cu118, cu124, cu126 (default: auto)
  --torch-version <ver>   torch version override (default depends on CUDA variant)
  --torchvision-version <ver> torchvision version override (default depends on CUDA variant)
  --open3d-package <pkg>  wheel name (default: open3d-cpu)

  --with-gdel3d           install `pygdel3d` (requires nvcc / CUDA toolchain)
  --with-pytorch3d        install `pytorch3d` (requires torch)
  --with-kaolin           install `kaolin` (requires torch)
  --with-all              enable all of the above
  --jobs <n>              parallel build jobs for native extensions (default: auto)

Environment variables:
  VENV_DIR, PYTHON_BIN    override defaults (same as options)
  UV_CACHE_DIR            cache dir for uv fallback (default: .uv-cache)
  DCCVT_CACHE_DIR         base dir for bootstrap scratch/cache (default: /tmp/dccvt-bootstrap-$USER)
  PIP_CACHE_DIR           pip wheel/http cache dir (default: $DCCVT_CACHE_DIR/pip)
  DCCVT_TMPDIR            pip temp/unpack dir (default: $DCCVT_CACHE_DIR/tmp)
  DCCVT_MIN_VENV_FREE_GB  warning threshold for the venv filesystem (default: 10)
  DCCVT_MIN_CACHE_FREE_GB warning threshold for pip cache/temp filesystem (default: 20)
  TORCH_VARIANT           override --torch (auto/cu118/cu124/cu126)
  TORCH_VERSION           override auto-selected torch version
  TORCHVISION_VERSION     override auto-selected torchvision version
  TORCH_INDEX_URL         override computed index URL (e.g. https://download.pytorch.org/whl/cu126)
  OPEN3D_PACKAGE          override --open3d-package
  BUILD_JOBS              override --jobs
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
    --jobs)
      BUILD_JOBS="$2"
      shift 2
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

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[bootstrap] Missing dependency: nvcc (CUDA toolkit)." >&2
  echo "DCCVT requires a CUDA-capable GPU and nvcc for gDel3D." >&2
  exit 1
fi

if [[ "$TORCH_VARIANT" == "cpu" ]]; then
  echo "[bootstrap] CPU-only torch installs are not supported." >&2
  echo "Use --torch cu118|cu124|cu126 or set TORCH_VARIANT accordingly." >&2
  exit 1
fi

detect_build_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return 0
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return 0
  fi
  echo 1
}

if [[ -z "$BUILD_JOBS" ]]; then
  BUILD_JOBS="$(detect_build_jobs)"
fi

if [[ -n "$BUILD_JOBS" ]]; then
  if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  fi
  if [[ -z "${MAKEFLAGS:-}" ]]; then
    export MAKEFLAGS="-j$BUILD_JOBS"
  fi
  if [[ -z "${MAX_JOBS:-}" ]]; then
    export MAX_JOBS="$BUILD_JOBS"
  fi
fi

configure_pip_storage

echo "[bootstrap] Updating submodules..."
git -C "$ROOT" submodule update --init --recursive

torch_index_url_for_variant() {
  local variant="$1"
  case "$variant" in
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
  return 1
}

detect_cuda_home() {
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    if [[ -n "$nvcc_path" ]]; then
      (cd "$(dirname "$nvcc_path")/.." && pwd)
      return 0
    fi
  fi
  return 1
}

ensure_cuda_env() {
  if command -v nvcc >/dev/null 2>&1; then
    if [[ -z "${CUDA_HOME:-}" ]]; then
      CUDA_HOME="$(detect_cuda_home || true)"
      if [[ -n "$CUDA_HOME" ]]; then
        export CUDA_HOME
      fi
    fi
    export FORCE_CUDA=1
  fi
}

repair_torch_python_deps() {
  if [[ $INSTALL_TORCH -ne 1 || $OFFLINE -eq 1 ]]; then
    return 0
  fi
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    return 0
  fi

  case "$TORCH_VERSION" in
    2.6.0*)
      echo "[bootstrap] Restoring torch 2.6 dependency pin: sympy==1.13.1"
      $PIP install "sympy==1.13.1"
      ;;
  esac
}

if ! venv_has_pip; then
  if [[ -e "$VENV_DIR" ]]; then
    echo "[bootstrap] Removing incomplete virtual environment at: $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
  create_venv
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
    torch_variant_resolved="$TORCH_VARIANT"
    if [[ "$torch_variant_resolved" == "auto" ]]; then
      torch_variant_resolved="$(detect_torch_variant)" || true
    fi
    if [[ -z "$TORCH_INDEX_URL" ]]; then
      if [[ -z "$torch_variant_resolved" ]]; then
        echo "[bootstrap] Could not determine CUDA version for torch." >&2
        echo "Pass --torch cu118|cu124|cu126 or set TORCH_VARIANT." >&2
        exit 1
      fi
      TORCH_INDEX_URL="$(torch_index_url_for_variant "$torch_variant_resolved")"
    fi
    if [[ -z "$TORCH_VERSION" || -z "$TORCHVISION_VERSION" ]]; then
      if [[ -z "$torch_variant_resolved" ]]; then
        echo "[bootstrap] Could not infer default torch versions from TORCH_INDEX_URL alone." >&2
        echo "Set TORCH_VARIANT, TORCH_VERSION, and TORCHVISION_VERSION explicitly." >&2
        exit 1
      fi
      read -r torch_version_default torchvision_version_default < <(torch_version_pair_for_variant "$torch_variant_resolved")
      if [[ -z "$TORCH_VERSION" ]]; then
        TORCH_VERSION="$torch_version_default"
      fi
      if [[ -z "$TORCHVISION_VERSION" ]]; then
        TORCHVISION_VERSION="$torchvision_version_default"
      fi
    fi
    echo "[bootstrap] Resolved torch variant: $torch_variant_resolved"
    echo "[bootstrap] Installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} from: $TORCH_INDEX_URL"
    warn_free_space "venv filesystem ($VENV_DIR)" "$VENV_DIR" "$DCCVT_MIN_VENV_FREE_GB"
    warn_free_space "pip cache/temp filesystem ($TMPDIR)" "$TMPDIR" "$DCCVT_MIN_CACHE_FREE_GB"
    $PIP install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
  fi
else
  echo "[bootstrap] Skipping torch install."
fi

if [[ $INSTALL_REQUIREMENTS -eq 1 ]]; then
  echo "[bootstrap] Installing requirements.txt..."
  $PIP install -r "$ROOT/requirements.txt"
  repair_torch_python_deps
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
  ensure_pip_tool cmake
  $PIP install -e "$ROOT/accel" "${PIP_LOCAL_FLAGS[@]}"
fi

if [[ $WITH_GDEL3D -eq 1 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[bootstrap] Skipping gDel3D: nvcc not found in PATH."
  else
    ensure_pip_tool cmake
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
    ensure_cuda_env
    # pytorch3d's setup.py imports torch during build.
    $PIP install -e "$ROOT/3rdparty/pytorch3d" --no-build-isolation "${PIP_LOCAL_FLAGS[@]}"
  fi
fi

if [[ $WITH_KAOLIN -eq 1 ]]; then
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "[bootstrap] Skipping kaolin: torch is not installed in the venv."
  else
    echo "[bootstrap] Installing kaolin..."
    ensure_cuda_env
    # kaolin's setup.py imports torch during build.
    $PIP install -e "$ROOT/3rdparty/kaolin" --no-build-isolation --no-deps "${PIP_LOCAL_FLAGS[@]}"
  fi
fi

cat <<EOF

[bootstrap] Done.
- Activate: source "$VENV_DIR/bin/activate"
EOF
