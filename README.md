# DCCVT (Kyushu experiments)

This repo contains DCCVT experiments plus several native / CUDA dependencies vendored as git submodules.

## Quickstart

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd DCCVT
bash scripts/bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` installs PyTorch by default; use `bash scripts/bootstrap.sh --help` to choose a torch wheel (`--torch cu126`, `--torch cpu`, etc.) or to run in `--offline` mode.

Examples:

- CUDA 12.6: `bash scripts/bootstrap.sh --torch cu126`
- CPU-only: `bash scripts/bootstrap.sh --torch cpu`

## Entry points

- `DCCVT_noDeadCode.py`
- `DCCVT.py`
- `sdf_extraction.py`

## Environment / installation (recommended: `venv`)

This project used to be installed via conda (`env.yml` is kept for reference), but the current workflow is a Python `venv` + git submodules.

### 0) Prerequisites

- Linux (Ubuntu-like recommended)
- Python `3.12.x` (the repo currently has local `venv` built with Python `3.12.2`; PyTorch support on `3.13` is not guaranteed)
- Build tools for native extensions:
  - `git`
  - a C/C++ compiler toolchain
  - `cmake` and `ninja`
  - Python headers (e.g. `sudo apt-get install python3.12-dev` on Ubuntu)
- If you want GPU acceleration: a working CUDA toolkit + driver compatible with your chosen PyTorch wheel

### 1) Clone + submodules

You do **not** manually clone the repos inside `3rdparty/` — they are git submodules tracked by this repo.

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd DCCVT
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2) Create and activate a virtual environment

Use any Python 3.12 you have installed (system Python, pyenv, or even conda's _base_ Python).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 3) Install PyTorch first

Pick the appropriate command for your CUDA version. Example for CUDA 12.6:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
```

CPU-only example:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
```

### 4) Install Python dependencies (everything except the local submodule builds)

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` is a snapshot of a known-good environment (generated from the author’s `venv`) to help you compare versions. It is not guaranteed to be directly installable (it may include locally-built packages like `open3d-cpu`).

### 5) Build + install local native/CUDA modules

This repo depends on several libraries which are included as git submodules and/or local C++/CUDA extensions.

General notes:

- Run these commands from the repo root, with your `.venv` activated.
- `pip install -e <path>` installs a package in “editable” mode: the code stays in-place, but any compiled extensions are built and installed into your venv.
- Patches in this repo (`3rdparty/*.patch`, `patches/*.patch`) are meant to be applied on top of the pinned submodule revisions. If `git apply` reports the patch is already applied, you can skip that step.
- If a build fails after you changed compilers/CUDA/Python, try again after deleting the submodule’s `build/` folder (or `pip uninstall <name>` and re-run the install).
- If you want to confirm each step worked, run the “verify” command after each install below.
- Not every repo in `3rdparty/` needs to be installed as a Python package; the list below covers the ones imported by the main scripts in this repo.

#### 5.0 Apply local patches (once)

This repo includes a small number of compatibility patches (CUDA/toolchain/torch-version).

```bash
bash scripts/apply_patches.sh
```

#### 5.1 `voronoiaccel` (local pybind11/CMake extension)

What it is: a small C++ extension (pybind11 + CMake) used by this repo as `import voronoiaccel`.

```bash
pip install -e accel
```

Verify:

```bash
python -c "import voronoiaccel; print('voronoiaccel OK')"
```

#### 5.2 `pygdel3d` (gDel3D python bindings)

What it is: Python bindings for the `3rdparty/gDel3D` CUDA code, exposed as `import pygdel3d`.

```bash
pip install -e 3rdparty/gDel3D/python_bindings
```

Verify:

```bash
python -c "import pygdel3d; print('pygdel3d OK')"
```

#### 5.3 `pytorch3d` (submodule + patch)

What it is: PyTorch3D from source (so it matches your Python/PyTorch/CUDA). The patch is a small compatibility tweak.

Note: `pytorch3d`'s `setup.py` imports `torch`, so the build must run with your already-installed PyTorch available (use `--no-build-isolation`). If you are offline / behind restricted network access, add `--no-deps` and make sure `iopath` is already installed in your environment.

```bash
pip install -e 3rdparty/pytorch3d --no-build-isolation
```

Offline / restricted-network variant:

```bash
pip install -e 3rdparty/pytorch3d --no-build-isolation --no-deps
```

Verify:

```bash
python -c "import pytorch3d; from pytorch3d.ops import knn_points; print('pytorch3d OK')"
```

#### 5.4 `kaolin` (submodule + patch)

What it is: NVIDIA Kaolin from source. The patch loosens the torch version constraint so newer torch releases can install.

```bash
pip install -e 3rdparty/kaolin --no-build-isolation
```

Verify:

```bash
python -c "import kaolin; print('kaolin OK')"
```

If Kaolin refuses your torch version, you can bypass the check:

```bash
IGNORE_TORCH_VER=1 pip install -e 3rdparty/kaolin --no-build-isolation
```

Offline / restricted-network variant:

```bash
pip install -e 3rdparty/kaolin --no-build-isolation --no-deps
```

#### 5.5 `open3d` (submodule + patch + build pip package)

What it is: Open3D built from source (the patch reduces optional components and fixes a small build issue).

```bash
cmake -S 3rdparty/open3d -B 3rdparty/open3d/build -GNinja -DBUILD_PYTHON_MODULE=ON -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build 3rdparty/open3d/build --config Release --target install-pip-package
```

This build may download third-party dependencies during configuration/build, so it typically requires network access.

Verify:

```bash
python -c "import open3d as o3d; print('open3d OK', o3d.__version__)"
```

If you don’t need the patched Open3D build, you can usually skip this and install a wheel instead (CPU-only example):

```bash
pip install open3d-cpu
```

### 6) Quick sanity check

```bash
python -c "import torch; import kaolin; import open3d; import pytorch3d; import pygdel3d; import voronoiaccel; print('OK')"
```

Notes:

- If you see `Warp CUDA error ...` messages, it usually means `warp-lang` can't find a usable CUDA driver/runtime on your machine. This repo typically expects a working CUDA install for GPU runs.

## Legacy conda env (reference only)

If you really want the historical conda environment: `env.yml`.
