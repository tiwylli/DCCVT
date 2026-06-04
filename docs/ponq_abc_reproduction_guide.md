# PoNQ ABC Reproduction And Baseline Evaluation Guide

This guide records the exact local workflow used to reproduce the PoNQ ABC
training checkpoint and run the ABC baseline mesh generation and metrics.

It assumes the repository root is:

```bash
export REPO=/export/livia/home/vision/Wcharawi/dev/DCCVT
export PONQ=$REPO/PoNQ-main
export ABC_ROOT=/export/livia/home/vision/Wcharawi/datasets/abc
export ABC_RAW_OBJ=$ABC_ROOT/raw_obj
export ABC_SDFGEN=$ABC_ROOT/tools/SDFGen
export PONQ_HDF5=/tmp/ponq_abc/gt_Quadrics
```

## Current Successful Outputs

The current completed run produced:

- Final checkpoint:
  `$PONQ/data/pretrained_PoNQ_ABC_retrained.pt`
- Training logs:
  `$PONQ/logs/abc_retrain_bs48_20260529_160954/`
- Baseline evaluation logs:
  `$PONQ/logs/baseline_eval_20260603_134041/`
- Latest baseline log symlink:
  `$PONQ/logs/baseline_eval_latest`
- ABC grid-32 meshes:
  `$PONQ/out_retrained/ABC_pretrained_PoNQ_ABC_retrained_32/`
- ABC grid-64 meshes:
  `$PONQ/out_retrained/ABC_pretrained_PoNQ_ABC_retrained_64/`
- ABC metric arrays:
  `$PONQ/src/eval/results/results_ABC_pretrained_PoNQ_ABC_retrained_32.npy`
  and
  `$PONQ/src/eval/results/results_ABC_pretrained_PoNQ_ABC_retrained_64.npy`

Current ABC metric summary:

| Output | CD x 1e-5 | F1 | NC | ECD | EF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ABC_pretrained_PoNQ_ABC_retrained_32` | 1.230 | 0.847 | 0.962 | 0.190 | 0.691 |
| `ABC_pretrained_PoNQ_ABC_retrained_64` | 0.859 | 0.892 | 0.980 | 0.137 | 0.855 |

## Environment

The PoNQ environment lives inside the PoNQ repository:

```bash
cd "$PONQ"
uv venv --python 3.9 .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install the PoNQ-compatible CUDA stack. The completed run used
`torch==1.12.1+cu113`, `pytorch3d==0.7.2`, `numpy==1.23.5`,
`trimesh==3.15.5`, and `h5py==3.14.0`.

```bash
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install pytorch3d==0.7.2 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html

python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

python -m pip install "numpy<1.24" h5py trimesh==3.15.5 libigl tqdm pyyaml \
  scikit-learn scipy scikit-image matplotlib joblib rtree
```

Verify the environment:

```bash
cd "$PONQ"
source .venv/bin/activate
PYTHONPATH=src/utils python - <<'PY'
import torch, pytorch3d, torch_scatter, h5py, igl, trimesh
from SDF_CNN import CNN_3d_multiple_split
print(torch.__version__)
print("PoNQ env OK")
PY
```

## ABC Dataset Layout

The local workflow uses only the first official ABC OBJ chunk, because PoNQ's
training split and ABC validation split used here are inside chunk `0000`.

Important paths:

| Path | Purpose | Current size/count |
| --- | --- | --- |
| `$ABC_ROOT/archives/` | ABC downloaded archives | about 7.4G |
| `$ABC_RAW_OBJ/` | Extracted ABC OBJ folders | about 80G |
| `$ABC_ROOT/tools/SDFGen` | NMC SDF generator binary | 46K |
| `$PONQ_HDF5/` | Generated PoNQ HDF5 files | about 90G, 4914 files |

Fetch and extract the first ABC OBJ chunk:

```bash
mkdir -p "$ABC_ROOT/archives" "$ABC_RAW_OBJ" "$ABC_ROOT/tools"

wget -c --no-check-certificate \
  -O "$ABC_ROOT/archives/abc_0000_obj_v00.7z" \
  https://archive.nyu.edu/rest/bitstreams/89085/retrieve

unar -o "$ABC_RAW_OBJ" "$ABC_ROOT/archives/abc_0000_obj_v00.7z"
```

Build `SDFGen` from NMC:

```bash
cd "$ABC_ROOT/tools"
git clone --depth 1 https://github.com/czq142857/NMC.git
cd NMC/data_utils/get_exact_sdf
cmake .
make -j
cp bin/SDFGen "$ABC_ROOT/tools/SDFGen"
```

## Local Code Changes Used By This Workflow

The rerun workflow relies on these local files:

- `$PONQ/src/utils/data_processing/get_data.py`
  - Supports `--input-dir`, `--output-dir`, `--names-file`, `--sdfgen`,
    `--n-jobs`, `--limit`, and `--skip-existing`.
  - Uses `subprocess.run(..., check=True)` for `SDFGen`.
  - Writes a temporary HDF5 first, then atomically renames it.
- `$PONQ/src/utils/data_processing/normalize_abc_obj_subset.py`
  - Normalizes selected raw ABC folders into `model.obj`.
  - This is used for the ABC eval split instead of normalizing all 10k folders.
- `$PONQ/src/eval/eval_ABC.py`
  - Supports `-gt_dir`.
  - The original hard-coded ABC path remains the default.
- `$PONQ/src/eval/abc_eval_last20.txt`
  - Contains the 1071 validation IDs from the last 20 percent of
    `src/eval/abc_ordered.txt`.
- `$PONQ/configs/eval_cnn_retrained.yaml`
  - Points generation to `/tmp/ponq_abc/` and the retrained checkpoint.

## HDF5 Generation

PoNQ training and evaluation need HDF5 files with these keys:

- `pointcloud`
- `normals`
- `32_sdf`
- `64_sdf`
- `128_sdf`

The current HDF5 root is temporary storage:

```bash
mkdir -p "$PONQ_HDF5"
```

Training HDF5 files use `src/utils/abc_watertight_train.txt` and contain 3843
model IDs:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python src/utils/data_processing/get_data.py \
  --input-dir "$ABC_RAW_OBJ" \
  --output-dir "$PONQ_HDF5" \
  --names-file src/utils/abc_watertight_train.txt \
  --sdfgen "$ABC_SDFGEN" \
  --n-jobs 12 \
  --skip-existing
```

Evaluation HDF5 files use the last 20 percent ABC validation split and contain
1071 model IDs. First normalize missing eval OBJ folders:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python src/utils/data_processing/normalize_abc_obj_subset.py \
  --input-dir "$ABC_RAW_OBJ" \
  --names-file src/eval/abc_eval_last20.txt \
  --skip-existing
```

Then generate the eval HDF5 files:

```bash
PYTHONPATH=src/utils python src/utils/data_processing/get_data.py \
  --input-dir "$ABC_RAW_OBJ" \
  --output-dir "$PONQ_HDF5" \
  --names-file src/eval/abc_eval_last20.txt \
  --sdfgen "$ABC_SDFGEN" \
  --n-jobs 12 \
  --skip-existing
```

Verify HDF5 coverage:

```bash
cd "$PONQ"

find "$PONQ_HDF5" -maxdepth 1 -type f -name '*.hdf5' | wc -l

PYTHONPATH=src/utils python - <<'PY'
from pathlib import Path
import h5py

hdf5_root = Path("/tmp/ponq_abc/gt_Quadrics")
train = Path("src/utils/abc_watertight_train.txt").read_text().splitlines()
eval_ids = Path("src/eval/abc_eval_last20.txt").read_text().splitlines()
names = [Path(name).stem for name in train + eval_ids]
missing = [name for name in names if not (hdf5_root / f"{name}.hdf5").exists()]
print(f"expected ids: {len(names)}")
print(f"missing hdf5: {len(missing)}")
if missing:
    print(missing[:20])
    raise SystemExit(1)

with h5py.File(hdf5_root / f"{names[0]}.hdf5", "r") as fin:
    print(sorted(fin.keys()))
PY
```

Expected current count: `4914`.

## Training

The final successful run used the local batch-48 configs:

- `configs/local_abc_cnn_multiple_quadrics_split_1_bs48_resume.yaml`
- `configs/local_abc_cnn_multiple_quadrics_split_2_bs48.yaml`
- `configs/local_abc_cnn_multiple_quadrics_split_3_bs48.yaml`

The first batch-48 phase resumed from
`models/model_multiple_quadrics_split_phase1_bs16_epoch002.pt`. This resumes
model weights only; optimizer state is not restored by the training script.

Run the three phases:

```bash
cd "$PONQ"
source .venv/bin/activate
mkdir -p models logs

PYTHONPATH=src/utils python src/utils/train_cnn_multiple_quadrics_split.py \
  configs/local_abc_cnn_multiple_quadrics_split_1_bs48_resume.yaml \
  | tee logs/train_phase1_bs48.log
mv model.pt models/model_multiple_quadrics_split.pt
mv loss.png logs/loss_phase1_bs48.png

PYTHONPATH=src/utils python src/utils/train_cnn_multiple_quadrics_split.py \
  configs/local_abc_cnn_multiple_quadrics_split_2_bs48.yaml \
  | tee logs/train_phase2_bs48.log
mv model.pt models/model_multiple_quadrics_split_fine.pt
mv loss.png logs/loss_phase2_bs48.png

PYTHONPATH=src/utils python src/utils/train_cnn_multiple_quadrics_split.py \
  configs/local_abc_cnn_multiple_quadrics_split_3_bs48.yaml \
  | tee logs/train_phase3_bs48.log
mv model.pt data/pretrained_PoNQ_ABC_retrained.pt
mv loss.png logs/loss_phase3_bs48.png
```

The original batch-16 local configs are still available if exact hyperparameter
matching is preferred over speed:

- `configs/local_abc_cnn_multiple_quadrics_split_1.yaml`
- `configs/local_abc_cnn_multiple_quadrics_split_2.yaml`
- `configs/local_abc_cnn_multiple_quadrics_split_3.yaml`

## Checkpoint Verification

Run this before mesh generation:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python - <<'PY'
import torch
from SDF_CNN import CNN_3d_multiple_split

device = "cuda"
model = CNN_3d_multiple_split(grid_n=33, K=4, ef_dim=128, device=device).to(device)
model.load_state_dict(torch.load("data/pretrained_PoNQ_ABC_retrained.pt", map_location=device))
x = torch.zeros(1, 1, 33, 33, 33, device=device)
print([tuple(t.shape) for t in model(x)])
PY
```

Expected shape list:

```text
[(1, 32768, 4, 3), (1, 32768, 4, 3), (1, 32768, 4, 3), (1, 32768, 4, 3, 3), (1, 32768)]
```

## Baseline Mesh Generation

Use `configs/eval_cnn_retrained.yaml`:

```yaml
path:
  out_dir: 'out_retrained/'
  datasets_path: '/tmp/ponq_abc/'
training:
  model_name: 'data/pretrained_PoNQ_ABC_retrained.pt'
```

Run ABC mesh generation at the two PoNQ evaluation resolutions:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python src/utils/generate_mesh_CNN.py \
  configs/eval_cnn_retrained.yaml -dataset ABC -grid_n 33 -n_jobs 8

PYTHONPATH=src/utils python src/utils/generate_mesh_CNN.py \
  configs/eval_cnn_retrained.yaml -dataset ABC -grid_n 65 -n_jobs 8
```

Expected output directories:

```text
out_retrained/ABC_pretrained_PoNQ_ABC_retrained_32/
out_retrained/ABC_pretrained_PoNQ_ABC_retrained_64/
```

## ABC Metrics

Run metrics against the local raw ABC OBJ root:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python src/eval/eval_ABC.py \
  out_retrained/ABC_pretrained_PoNQ_ABC_retrained_32 \
  -gt_dir "$ABC_RAW_OBJ"

PYTHONPATH=src/utils python src/eval/eval_ABC.py \
  out_retrained/ABC_pretrained_PoNQ_ABC_retrained_64 \
  -gt_dir "$ABC_RAW_OBJ"
```

Metric output files are written to:

```text
src/eval/results/results_ABC_pretrained_PoNQ_ABC_retrained_32.npy
src/eval/results/results_ABC_pretrained_PoNQ_ABC_retrained_64.npy
```

## One-Shot Baseline Rerun

This reruns eval OBJ normalization, missing eval HDF5 generation, checkpoint
sanity, mesh generation, and metrics. It does not retrain.

```bash
cd "$PONQ"
source .venv/bin/activate

RUN_DIR="logs/baseline_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
ln -sfn "$(basename "$RUN_DIR")" logs/baseline_eval_latest

{
  date --iso-8601=seconds

  PYTHONPATH=src/utils python src/utils/data_processing/normalize_abc_obj_subset.py \
    --input-dir "$ABC_RAW_OBJ" \
    --names-file src/eval/abc_eval_last20.txt \
    --skip-existing

  PYTHONPATH=src/utils python src/utils/data_processing/get_data.py \
    --input-dir "$ABC_RAW_OBJ" \
    --output-dir "$PONQ_HDF5" \
    --names-file src/eval/abc_eval_last20.txt \
    --sdfgen "$ABC_SDFGEN" \
    --n-jobs 12 \
    --skip-existing

  PYTHONPATH=src/utils python - <<'PY'
import torch
from SDF_CNN import CNN_3d_multiple_split
model = CNN_3d_multiple_split(grid_n=33, K=4, ef_dim=128).cuda()
model.load_state_dict(torch.load("data/pretrained_PoNQ_ABC_retrained.pt"))
print("checkpoint OK")
PY

  PYTHONPATH=src/utils python src/utils/generate_mesh_CNN.py \
    configs/eval_cnn_retrained.yaml -dataset ABC -grid_n 33 -n_jobs 8

  PYTHONPATH=src/utils python src/utils/generate_mesh_CNN.py \
    configs/eval_cnn_retrained.yaml -dataset ABC -grid_n 65 -n_jobs 8

  PYTHONPATH=src/utils python src/eval/eval_ABC.py \
    out_retrained/ABC_pretrained_PoNQ_ABC_retrained_32 \
    -gt_dir "$ABC_RAW_OBJ"

  PYTHONPATH=src/utils python src/eval/eval_ABC.py \
    out_retrained/ABC_pretrained_PoNQ_ABC_retrained_64 \
    -gt_dir "$ABC_RAW_OBJ"

  date --iso-8601=seconds
} 2>&1 | tee "$RUN_DIR/pipeline.log"
```

## Data Move Checklist For Admin

These are the data directories required to rerun without regenerating:

| Directory | Required for | Current size |
| --- | --- | ---: |
| `/tmp/ponq_abc/gt_Quadrics/` | Training, generation, eval HDF5 input | about 90G |
| `/export/livia/home/vision/Wcharawi/datasets/abc/raw_obj/` | Ground-truth OBJ metrics and HDF5 regeneration | about 80G |
| `/export/livia/home/vision/Wcharawi/datasets/abc/tools/` | `SDFGen`, NMC tooling | small, except NMC checkout |
| `/export/livia/home/vision/Wcharawi/datasets/abc/archives/` | Re-extracting ABC chunk 0000 | about 7.4G |

After the move:

1. Update `PONQ_HDF5` in shell commands.
2. Update `path.datasets_path` in `configs/eval_cnn_retrained.yaml` to the
   parent directory of `gt_Quadrics`.
3. Pass the new raw OBJ root to `eval_ABC.py` with `-gt_dir`.
4. If `SDFGen` moves, pass its new path through `--sdfgen`.

## Common Failure Cases

- Missing HDF5 files:
  `/tmp` is temporary. If `/tmp/ponq_abc/gt_Quadrics` disappears, regenerate
  from `$ABC_RAW_OBJ` with `get_data.py`.
- Missing `model.obj`:
  Run `normalize_abc_obj_subset.py` for the relevant names file.
- `eval_ABC.py` uses the wrong ground-truth root:
  Always pass `-gt_dir "$ABC_RAW_OBJ"` unless the default path exists locally.
- CUDA out of memory during training:
  Reduce `training.batch_size` in the local training YAML.
- Slower training than expected:
  The PoNQ training script samples many points per batch. The batch-48 configs
  were used because GPU memory allowed it on the available RTX A6000.
- Exact checkpoint mismatch:
  Bitwise reproduction is not expected. Dataset sampling and DataLoader
  shuffling are not fully seeded in the original code path.

## HotSpot SDF Cache Baseline

The retrained ABC PoNQ checkpoint was also run directly on existing HotSpot dense
SDF caches without converting them to PoNQ HDF5 files.

Input and output paths:

```text
Config: PoNQ-main/configs/eval_hotspot_ponq_retrained.yaml
Checkpoint: PoNQ-main/data/pretrained_PoNQ_ABC_retrained.pt
Input caches: /export/livia/home/vision/Wcharawi/dev/DCCVT/outputs/neural_hotspot_sdf/thingi32_g33/*.npz
GT meshes: /export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32/<mesh_id>.obj
Output: PoNQ-main/out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32/
Metrics: PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32.npy
```

Smoke generation and metric commands:

```bash
cd "$PONQ"
source .venv/bin/activate

PYTHONPATH=src/utils python src/utils/generate_mesh_hotspot_ponq.py \
  configs/eval_hotspot_ponq_retrained.yaml \
  --limit 1 \
  --n_jobs 2 \
  --overwrite

PYTHONPATH=src/utils python src/eval/eval_THINGI.py \
  out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32 \
  -gt_dir /export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32 \
  -all_models src/eval/hotspot_thingi32_g33_smoke.txt \
  -pred_suffix .obj
```

Full generation and metric commands:

```bash
PYTHONPATH=src/utils python src/utils/generate_mesh_hotspot_ponq.py \
  configs/eval_hotspot_ponq_retrained.yaml \
  --n_jobs 8 \
  --overwrite

PYTHONPATH=src/utils python src/eval/eval_THINGI.py \
  out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32 \
  -gt_dir /export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32 \
  -all_models src/eval/hotspot_thingi32_g33_ids.txt \
  -pred_suffix .obj
```

The full run processed 31 caches with zero generation failures. Summary:

```text
CD (x 1e-5), F1, NC, ECD, EF1
HotSpot_pretrained_PoNQ_ABC_retrained_32 & 307.813 & 0.042 & 0.728 & 0.378 & 0.029
```

### HotSpot Alignment Diagnostics

The strict HotSpot score above uses the existing PoNQ/Thingi convention. To
diagnose scale mismatch, run:

```bash
PYTHONPATH=src/utils python src/eval/eval_HOTSPOT.py \
  out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32 \
  -gt_dir /export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32 \
  -all_models src/eval/hotspot_thingi32_g33_ids.txt \
  -pred_suffix .obj \
  -mode all
```

This writes:

```text
src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_ponq_thingi.npy
src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_raw.npy
src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_bbox_aligned.npy
src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_hotspot_summary.csv
```

Current 31-shape diagnostic means:

| Mode | CD x 1e-5 | F1 | NC | ECD | EF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ponq_thingi` | 307.902 | 0.042 | 0.729 | 0.377 | 0.029 |
| `raw` | 1232.188 | 0.015 | 0.729 | 1.510 | 0.008 |
| `bbox_aligned` | 14.024 | 0.261 | 0.896 | 0.240 | 0.079 |

`bbox_aligned` applies only uniform scale plus translation to the prediction.
It is a diagnostic, not the strict baseline. The large improvement indicates the
poor strict metric is strongly affected by scale/alignment mismatch.

Only `grid_n=33` HotSpot caches were used, so this run produces `_32` outputs.
To run `_64`, first generate matching `grid_n=65` HotSpot caches and then use a
separate config with `data.grid_n: 65`.

## Scope Notes

This guide covers the ABC PoNQ baseline only:

- ABC OBJ fetch and normalization
- HDF5 generation
- PoNQ checkpoint retraining
- ABC mesh generation
- ABC metric evaluation
- HotSpot SDF cache inference for the retrained ABC PoNQ checkpoint

DCCVT/neural method development remains a separate experiment stage.
