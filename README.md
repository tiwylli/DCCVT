# DCCVT: Differentiable Clipped Centroidal Voronoi Tessellation

This repo contains DCCVT experiments plus several native / CUDA dependencies vendored as git submodules. The notes below cover first-party code and workflows (everything outside `3rdparty/`).

GPU requirement: DCCVT relies on `pygdel3d` (gDel3D) for Delaunay tetrahedralization in the core geometry path. gDel3D requires CUDA + `nvcc`, so experiments are not supported on CPU-only machines.

## Repository layout

- `DCCVT.py`: main CLI entrypoint; requires `--args-file`.
- `dccvt/`: core experiment logic (arg parsing, training, mesh extraction).
- `argfiles/`: experiment templates; each line expands into a full CLI invocation.
- `mesh/`: mesh datasets (`.obj` + `.ply`). `--mesh` expects a path without extension; the runner loads `<mesh>.ply`.
- `hotspots_model/`: pretrained HotSpot weights referenced by `--trained_HotSpot`.
- `outputs/`: default output root for generated meshes and artifacts.
- `metrics_figs_scripts/`: metrics, batch renders, and SDF extraction helpers.
- `accel/`: `voronoiaccel` extension used by metrics and utilities.
- `scripts/`: bootstrapping and patch helpers.

## Quickstart (recommended)

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd DCCVT
bash scripts/bootstrap.sh --torch cu126
source .venv/bin/activate
python DCCVT.py --args-file argfiles/DCCVT_figs_teaser.args --mesh-ids 313444
```

CPU-only installs are not supported because `pygdel3d` requires CUDA.

Dry-run the arg expansion without running experiments:

```bash
python DCCVT.py --args-file argfiles/DCCVT_figs_teaser.args --mesh-ids 313444 --dry-run
```

## Running experiments

The entrypoint requires an args template file:

```bash
python DCCVT.py --args-file argfiles/DCCVT_figs_teaser.args
```

Useful flags:

- `--mesh-ids 313444,441708`: override the mesh list used by `{mesh_id}` expansion.
- `--timestamp 20250101_120000`: set the output root under `outputs/`.
- `--dry-run`: print expanded argv lists and exit.

### Argfile format

Each non-comment line is a CLI template; placeholders are filled using defaults from `dccvt/argparse_utils.py`:

```text
@mesh_ids : 313444 441708
--mesh {mesh}{mesh_id} --trained_HotSpot {trained_HotSpot}thingi32/{mesh_id}.pth \
  --output {output}{mesh_id} --w_chamfer 1000 --w_cvt 100 --num_centroids 16
```

Notes:

- `{mesh_id}` expands over the active mesh list (defaults to `DEFAULTS["mesh_ids"]`).
- `@mesh_ids:` changes the active mesh list for subsequent lines.
- Known placeholders (`{mesh}`, `{trained_HotSpot}`, `{output}`, etc.) resolve via defaults.
- Unknown placeholders are left intact for manual post-processing.
- Trailing `\` continues a line.

### Outputs and artifacts

By default, outputs go under `outputs/<timestamp>/`. The runner copies `DCCVT.py` and the fully expanded arg lists into that folder for reproducibility.

For each experiment line, `--output` (often `{output}{mesh_id}`) becomes the per-run folder. Output files include:

- `DCCVT_<upsampling>_<state>_projDCCVT_cvt*_sdfsmooth*.obj` + `.npz`
- `voromesh_<num_centroids>_<state>_DCCVT_cvt*_sdfsmooth*.obj` + `.npz`
- `target.ply` (sampled points)

If all expected final outputs already exist, the runner skips that mesh. To force a re-run, delete the output folder or change `--timestamp`.

## Metrics and renders

Batch render OBJ outputs:

```bash
python metrics_figs_scripts/DCCVT_batch_render.py outputs/<timestamp> \
  --recursive --filter final --resolution 512 512
```

Compute metrics over experiment folders:

```bash
python metrics_figs_scripts/DCCVT_figs_metrics.py \
  --root-dir /path/to/DCCVT \
  --experiments-dir outputs/<timestamp> \
  --include-final
```

Notes:

- `metrics_figs_scripts/DCCVT_metrics.py` uses hard-coded `EXPERIMENTS_DIR` and `GT_DIR`; edit the file or use `DCCVT_figs_metrics.py`.
- `metrics_figs_scripts/DCCVT_metric_check.py` launches an interactive Polyscope view for one OBJ.
- `metrics_figs_scripts/sdf_extraction.py` is a sandbox for SDF extraction; run `--help` for flags.

## Environment variables

- `DCCVT_ROOT`: override the root used for default paths (`mesh/`, `outputs/`, `hotspots_model/`).
- `DCCVT_DEVICE`: force device selection (`cpu`, `cuda:0`, etc.).

`scripts/bootstrap.sh --help` lists additional knobs (torch variant, Open3D package, build jobs).

## Manual installation (if you skip bootstrap)

### 0) Prerequisites

- Linux
- Python `3.12.x`
- Build tools for native extensions: `git`, a C/C++ compiler, `cmake`, `ninja`
- Python headers
- NVIDIA GPU with a working CUDA toolkit + driver (including `nvcc`)

### 1) Clone + submodules

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd DCCVT
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2) Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 3) Install PyTorch first

CUDA 12.6 example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
```

### 4) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5) Build + install local native/CUDA modules

Patches in this repo (`3rdparty/*.patch`) are meant to be applied on top of the pinned submodule revisions:

```bash
bash scripts/apply_patches.sh
```

Then install the local extensions that the main code imports:

#### 5.1 `voronoiaccel` (local pybind11/CMake extension)

```bash
pip install -e accel
```

#### 5.2 `pygdel3d` (gDel3D python bindings)

Required for running experiments (GPU-only).

```bash
pip install -e 3rdparty/gDel3D/python_bindings
```

#### 5.3 `pytorch3d` (submodule + patch)

```bash
pip install -e 3rdparty/pytorch3d --no-build-isolation
```

Offline / restricted-network variant:

```bash
pip install -e 3rdparty/pytorch3d --no-build-isolation --no-deps
```

#### 5.4 `kaolin` (submodule + patch)

```bash
pip install -e 3rdparty/kaolin --no-build-isolation
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

```bash
cmake -S 3rdparty/open3d -B 3rdparty/open3d/build -GNinja \
  -DBUILD_PYTHON_MODULE=ON \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build 3rdparty/open3d/build --config Release --target install-pip-package
```

If you do not need the patched Open3D build, you can usually skip this and install a wheel instead:

```bash
pip install open3d-cpu
```

### 6) Quick sanity check

```bash
python -c "import torch; import kaolin; import open3d; import pytorch3d; import pygdel3d; import voronoiaccel; print('OK')"
```

## Troubleshooting

- `Warp CUDA error ...`: `warp-lang` cannot find a usable CUDA driver/runtime; ensure CUDA is installed.
- `pygdel3d`/`nvcc` missing: DCCVT experiments will not run without a CUDA-capable GPU and `nvcc`.
- `HotSpot dependencies not found`: confirm `3rdparty/HotSpot` is initialized (`git submodule update --init --recursive`).
- `Python.h not found`: install your system Python dev package and retry `pip install -e accel`.
