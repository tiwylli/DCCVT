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

## HybridPoNQ ABC Training At 32 Cubed

The HybridPoNQ experiment reuses the ABC HDF5 files above and adds an exact
point-cloud UDF channel. The 129 cubed UDF is a preprocessing artifact only.
Training and inference both consume 33 cubed SDF/UDF vertex grids and predict
one 32 cubed DCCVT site/SDF field.

There is no 16 cubed or 64 cubed model path in this experiment.

### Architecture And Data Flow

For each shape:

1. Read the one-million-point `pointcloud` from the PoNQ HDF5 file.
2. Compute exact nearest-sample distances on `[-0.5, 0.5]^3` at 129 cubed
   vertices.
3. Store `128_udf` and the exact aligned view
   `32_udf = 128_udf[::4, ::4, ::4]`.
4. Load `32_sdf` and `32_udf`, multiply both by two, and stack them as
   `[SDF, UDF]` in the DCCVT `[-1, 1]^3` frame.
5. Apply the PoNQ-style `Conv3d(kernel_size=2)` encoder to produce 32 cubed
   cell features.
6. Predict bounded canonical-site offsets and residual SDF values.

The internal model channel name `hotspot_sdf` is retained for checkpoint
compatibility, but it contains the exact ABC `32_sdf` field in this experiment.

The comparison variants are:

- `direct`: random two-channel encoder and zero-initialized DCCVT heads.
- `ponq_pretrained`: copy a reproduced PoNQ encoder into the SDF channel,
  zero the new UDF input weights, and zero both DCCVT heads.

Training uses a fixed Delaunay connectivity computed from the canonical 32
cubed lattice. Recomputing Delaunay after microscopic movement of a perfectly
regular lattice creates near-degenerate tetrahedra and non-finite circumcenters.
Delaunay connectivity is already detached from autograd, so the fixed topology
keeps training finite. Final extraction always recomputes exact Delaunay
connectivity from the predicted sites.

### Files And Interfaces

- `configs/hybrid_ponq_abc_dccvt_v1.json`: resolved paths, model settings,
  PoNQ phases, DCCVT steps, losses, seed, and qualification thresholds.
- `scripts/precompute_abc_udf.py`: resumable multi-GPU UDF preprocessing.
- `scripts/train_hybrid_ponq_abc.py`: Python-3.9-compatible PoNQ DDP training.
- `scripts/train_hybrid_dccvt_abc.py`: DCCVT pilot/full training, extraction,
  metrics, qualification, and checkpoint resume.
- `dccvt/neural/abc/`: typed config, ABC dataset, sidecar validation,
  and encoder-transfer logic.
- `dccvt/neural/abc/cli.py`: step-based DDP training and evaluation.

The DCCVT environment requires `h5py==3.14.0` in addition to its existing
PyTorch3D and geometry runtime:

```bash
/tmp/dccvt-venv/bin/python -m pip install h5py==3.14.0
```

The PoNQ pretraining command must use `PoNQ-main/.venv/bin/python`. The DCCVT
training command must use `/tmp/dccvt-venv/bin/python` because `pygdel3d` is
installed only in that environment.

### UDF Preprocessing

Run from the repository root:

```bash
/tmp/dccvt-venv/bin/python scripts/precompute_abc_udf.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --gpus 0,1,2,3 \
  --split all \
  --resume
```

Important arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gpus` | `0,1,2,3` | Deterministic worker/GPU assignment. |
| `--split` | `all` | Process `train`, `validation`, or their ordered union. |
| `--limit` | unset | Restrict shape count for a smoke test. |
| `--resume` | false | Validate and skip complete sidecars. |
| `--fast-resume-check` | false | Skip the full aligned-value comparison. |

Each output is written atomically to:

```text
/tmp/ponq_abc/gt_UDF_128/<model_id>.hdf5
```

Datasets and attributes:

```text
128_udf                 float32 [129,129,129]
32_udf                  float32 [33,33,33]
coordinate_min          -0.5
coordinate_max           0.5
source_point_count       1000000
preprocessing_version    abc_udf_129_stride4_v1
downsample_rule          128_udf[::4,::4,::4]
```

Logs are saved under `/tmp/ponq_abc/gt_UDF_128/logs/`. Failures are written
per GPU as JSONL, and `summary.json` records written, skipped, and failed
counts.

One-shape smoke test:

```bash
/tmp/dccvt-venv/bin/python scripts/precompute_abc_udf.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --gpus 0 \
  --split train \
  --limit 1 \
  --resume
```

### PoNQ Encoder Reproduction

Run the exact three configured phases with global batch size 48:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PoNQ-main/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_ponq_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json
```

`--nproc_per_node` must equal the number of GPUs exposed by
`CUDA_VISIBLE_DEVICES`. CUDA renumbers visible devices locally, so
`CUDA_VISIBLE_DEVICES=0,2,3` exposes only three devices: `cuda:0`, `cuda:1`,
and `cuda:2`. For that case, launch three ranks:

```bash
CUDA_VISIBLE_DEVICES=0,2,3 \
PoNQ-main/.venv/bin/torchrun --standalone --nproc_per_node=3 \
  scripts/train_hybrid_ponq_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --preload-mode sdf
```

The configured global batch size is 48, so a three-GPU launch uses local batch
size 16 instead of 12. If memory is tight, use a temporary config with a smaller
global batch size divisible by three, for example 24:

```bash
cp configs/hybrid_ponq_abc_dccvt_v1.json /tmp/hybrid_ponq_abc_3gpu.json
PoNQ-main/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/hybrid_ponq_abc_3gpu.json")
config = json.loads(path.read_text())
config["ponq_training"]["global_batch_size"] = 24
path.write_text(json.dumps(config, indent=2) + "\n")
PY

CUDA_VISIBLE_DEVICES=0,2,3 \
PoNQ-main/.venv/bin/torchrun --standalone --nproc_per_node=3 \
  scripts/train_hybrid_ponq_abc.py \
  --config /tmp/hybrid_ponq_abc_3gpu.json \
  --preload-mode sdf
```

Use the four-GPU command above for the exact reproduced batch size and local
batch geometry.

The schedule is:

| Phase | Epochs | Surface samples | Learning rate |
| --- | ---: | ---: | ---: |
| 1 | 195 | 500,000 | `6.4e-5` |
| 2 | 195 | 700,000 | `3.2e-5` |
| 3 | 137 | 700,000 | `3.2e-5` |

AdamW uses weight decay `0.01`, betas `(0.9, 0.999)`, AMSGrad, `K=4`, and
loss weights `[100, 100, 0.1, 0.1, 0.1, 1]`.

The original loader sampled once and held the complete dataset in one process.
The DDP implementation preserves the fixed per-phase sample indices and loss
definitions, but defaults to `--preload-mode sdf`: it caches only the small
SDF-derived fields and reads sampled point/normals from HDF5 per batch. This
avoids holding hundreds of 500k to 700k point samples per rank in host RAM.
Exact rank manifests are saved for reproducibility.

Outputs:

```text
outputs/hybrid_ponq_abc/ponq_pretraining/
  resolved_config.json
  rank_splits/rank_<n>.txt
  phase_<n>_latest.pt
  ponq_encoder.pt
```

Use `--resume <checkpoint> --resume-optimizer` to continue. The
`--max-epochs-per-phase` option is for smoke testing only and must not be used
for the reproduction run.

PoNQ loader modes:

| Mode | Behavior | Use |
| --- | --- | --- |
| `sdf` | Cache SDF, near-surface masks, and GT cell masks; read sampled point/normals per batch. | Default and recommended. |
| `none` | Load all fields on demand. | Lowest resident memory, slower. |
| `full` | Materialize sampled point/normals for every rank-local shape. | Legacy behavior; avoid on memory-limited runs. |

The trainer also checks GPU memory before preloading. By default each rank
requires at least 12 GiB free on its assigned GPU. If a run fails this check,
select different `CUDA_VISIBLE_DEVICES`, stop other jobs, or lower
`ponq_training.global_batch_size`. The guard can be disabled with
`--min-free-gb 0`, but doing so can produce cuDNN allocation or algorithm
selection errors later.

cuDNN uses the legacy PoNQ policy by default: deterministic cuDNN is disabled
and benchmarking is enabled. Use `--cudnn-deterministic` only when bitwise
cuDNN determinism matters more than compatibility/performance.

### DCCVT Pilot And Full Training

Both variants use:

- four-GPU DDP with one shape per GPU step;
- 100,000 target points per shape;
- `1000 * Chamfer`;
- `0.01 * mean(site_delta^2)`;
- `0.01 * mean(sdf_residual^2)`;
- no CVT or SDF smoothness loss;
- validation every 250 steps;
- best-checkpoint selection by proxy Chamfer;
- seed `69`.

Direct pilot:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/tmp/dccvt-venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_dccvt_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --variant direct \
  --run pilot \
  --stage all
```

PoNQ-initialized pilot:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/tmp/dccvt-venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_dccvt_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --variant ponq_pretrained \
  --encoder-checkpoint outputs/hybrid_ponq_abc/ponq_pretraining/ponq_encoder.pt \
  --run pilot \
  --stage all
```

After a variant passes the pilot gate, replace `--run pilot` with
`--run full`. The pilot uses 128 seeded training shapes, 32 validation shapes,
and 250 steps. The full run uses all 3,843 training shapes, a fixed 64-shape
validation proxy, and 3,000 steps.

Training-only, evaluation-only, and resume examples:

```bash
# Training only
/tmp/dccvt-venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_dccvt_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --variant direct --run full --stage train

# Resume optimizer and model state
/tmp/dccvt-venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_dccvt_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --variant direct --run full --stage train \
  --resume outputs/hybrid_ponq_abc/direct/full/checkpoints/latest.pt \
  --resume-optimizer

# Evaluate an explicit checkpoint over all 1,071 validation IDs
/tmp/dccvt-venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_hybrid_dccvt_abc.py \
  --config configs/hybrid_ponq_abc_dccvt_v1.json \
  --variant direct --run full --stage evaluate \
  --checkpoint outputs/hybrid_ponq_abc/direct/full/checkpoints/best.pt
```

Pass `--skip-baseline` during evaluation to avoid re-extracting the canonical
baseline when it is already available.

### Outputs And Acceptance

Each run writes:

```text
outputs/hybrid_ponq_abc/<variant>/<pilot|full>/
  resolved_config.json
  splits/{train,validation,validation_proxy}.txt
  validation_step_<step>.json
  checkpoints/{best,latest,step_<step>}.pt
  evaluation/
    validation_ids.txt
    model/{meshes,metrics.npy,metrics.summary.json}
    canonical/{meshes,metrics.npy,metrics.summary.json}
    summary.json
```

Checkpoints contain the model and optimizer states, model config, resolved
experiment config, command arguments, seed, initialization metadata, Git
revision, and proxy metrics.

The pilot qualifies only when:

- all 32 meshes extract;
- squared Chamfer improves by at least 5 percent over canonical DCCVT;
- normal consistency regresses by no more than `0.01`;
- edge F1 regresses by no more than `0.05`.

Metrics use the existing PoNQ ABC definitions: Chamfer, F1, normal
consistency, edge Chamfer, and edge F1. The updated evaluator accepts an
explicit names file, seed, sample count, output prefix, and prediction pattern
while preserving its original positional command.

### Common Failures And Limitations

- `ModuleNotFoundError: h5py`: install `h5py==3.14.0` in the DCCVT environment.
- `ModuleNotFoundError: pygdel3d`: DCCVT training was launched from the PoNQ
  environment; use `/tmp/dccvt-venv`.
- PoNQ pretraining exits with `SIGKILL` during `preload`: the OS killed a rank,
  usually because the older full preload exhausted host RAM or swap. Use the
  default `--preload-mode sdf`, or pass `--preload-mode none` for the lowest
  resident memory.
- `RuntimeError: Unable to find a valid cuDNN algorithm to run convolution`
  during PoNQ pretraining: the selected GPU is usually too full for the local
  batch, or deterministic cuDNN was forced. Check `nvidia-smi`; avoid GPUs with
  other large jobs, keep the default cuDNN policy, or reduce
  `ponq_training.global_batch_size`.
- `RuntimeError: invalid device ordinal` or a preflight error saying torchrun
  launched more local processes than visible CUDA devices: match
  `--nproc_per_node` to the number of entries in `CUDA_VISIBLE_DEVICES`.
  For example, `CUDA_VISIBLE_DEVICES=0,2,3` requires `--nproc_per_node=3`, not
  `4`.
- Missing UDF sidecars: finish preprocessing or rerun with `--resume`.
- Sidecar version mismatch: delete or regenerate the reported file with the
  current config.
- Non-finite training loss: the trainer aborts instead of saving the invalid
  state. Do not change `delaunay_mode` from `canonical_fixed`.
- The UDF is the distance to the supplied one-million-point surface sample,
  not an analytic point-to-triangle distance.
- The aligned 33 cubed UDF preserves exact values at model vertices but cannot
  expose 129 cubed detail to a 32 cubed model.
- Final extraction is expensive because exact 32 cubed Delaunay connectivity
  is recomputed for every validation shape.

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
