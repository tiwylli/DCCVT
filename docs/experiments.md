# Experiments

## Hybrid Direct Mesh-Loss Adaptation Study

- Config: `configs/neural_hybrid_mesh_finetune_cv.json`
- Runner: `scripts/run_hybrid_mesh_finetune_cv.py`
- Starting checkpoint: `outputs/neural_dccvt/hybrid_direct_v1/full_supervised/latest.pt` at epoch `299`
- Input split: `PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt`
- Output root: `outputs/neural_dccvt/hybrid_direct_mesh_finetune_cv/`
- Purpose: test whether differentiable DCCVT mesh losses improve an already supervised hybrid direct predictor.
- Study type: adaptation. Fold test meshes are excluded from mesh fine-tuning, but the shared supervised checkpoint was trained on all 31 meshes. This is not a clean unseen-shape generalization experiment.
- Fold assignment: preserve source-file order and assign each ID to test fold `index % 5`. Test fold sizes are `7, 6, 6, 6, 6`.
- Seed behavior: training, DataLoader ordering, inference, and metric surface/edge sampling use seed `69`. The resolved config and all fold split files are saved under the output root.
- Training schedule: 25 epochs, batch size 1, AdamW with a fresh optimizer, LR `1e-5`, all model parameters trainable, and supervised losses retained.
- Variants:
  - `chamfer_only`: `1000 * L_chamfer`
  - `composite`: `1000 * L_chamfer + 100 * L_cvt + 100 * L_sdfsmooth`
- Strict behavior: mesh fine-tuning aborts on invalid SDF signs, empty clipped surfaces, or geometry failures instead of silently skipping the shape.
- Evaluation mesh: `intDCCVT`; `projDCCVT` remains in each inference directory as a diagnostic artifact.
- Metrics: 100,000 samples with `ponq_thingi`, `raw`, and `bbox_aligned` modes. Primary metrics are bbox-aligned Chamfer and normal consistency.
- Qualification rule: improve bbox-aligned Chamfer in at least 4/5 folds and 16/31 shapes, lose no more than `0.01` aggregate normal consistency, and have zero extraction failures.

Prepare and inspect the commands:

```bash
/tmp/dccvt-venv/bin/python scripts/run_hybrid_mesh_finetune_cv.py \
  --config configs/neural_hybrid_mesh_finetune_cv.json \
  --stage all \
  --dry-run
```

One-fold/one-variant subset execution:

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/dccvt-venv/bin/python \
  scripts/run_hybrid_mesh_finetune_cv.py \
  --config configs/neural_hybrid_mesh_finetune_cv.json \
  --output-root outputs/neural_dccvt/hybrid_direct_mesh_finetune_cv_smoke \
  --stage all \
  --folds 0 \
  --variants chamfer_only
```

Full multi-GPU study:

```bash
/tmp/dccvt-venv/bin/python scripts/run_hybrid_mesh_finetune_cv.py \
  --config configs/neural_hybrid_mesh_finetune_cv.json \
  --stage all \
  --parallel \
  --devices auto \
  --min-free-gb 20 \
  --poll-seconds 60
```

Stages can also be run separately with `--stage prepare`, `train`, `infer`,
`evaluate`, or `summarize`. Use `--folds` and `--variants` to select subsets.
Completed training and inference artifacts are reused; partial checkpoint
directories require `--force` to restart from the configured supervised
checkpoint.

Output convention:

```text
outputs/neural_dccvt/hybrid_direct_mesh_finetune_cv/
  resolved_config.json
  splits/fold_<n>_{train,test}.txt
  runs/<variant>/fold_<n>/checkpoints/
  inference/{baseline,<variant>}/...
  eval_meshes/<method>_intDCCVT/<mesh_id>.obj
  evaluation/results_*.npy
  evaluation/results_*_hotspot_summary.csv
  summary/per_shape_metrics.csv
  summary/fold_summary.csv
  summary/decision_summary.json
```

- Metric status: pending full study execution.

## Hybrid PoNQ-DCCVT Direct Extractor v1

- Guide: [docs/hybrid_ponq_dccvt_guide.md](/export/livia/home/vision/Wcharawi/dev/DCCVT/docs/hybrid_ponq_dccvt_guide.md)
- Config: `configs/neural_hybrid_direct_v1.json`
- Training script: `scripts/train_dccvt_hybrid_direct.py`
- Inference script: `scripts/infer_dccvt_hybrid_direct.py`
- Input caches: `outputs/neural_hotspot_sdf/thingi32_g33/*.npz`
- Supervised labels: `outputs/neural_labels/n32/<mesh_id>/DCCVT_0_final_projDCCVT_cvt100_sdfsmooth100.npz`
- Purpose: predict the full `32^3` DCCVT Voronoi site field and corresponding SDF values directly from HotSpot SDF plus voxelized point-cloud zero-level evidence.
- Architecture: PoNQ-style dense `Conv3d` encoder from four channels (`hotspot_sdf`, `abs_hotspot_sdf`, `point_udf`, `point_confidence`) to `32^3` features; heads predict bounded canonical-site offsets and bounded residual SDF corrections.
- Output convention: checkpoints and resolved configs are saved under `outputs/neural_dccvt/hybrid_direct_v1/checkpoints/`; inference predictions and optional meshes should be written under `outputs/neural_dccvt/hybrid_direct_v1/infer_<mesh_id>/`.
- Seed behavior: training and inference expose `--seed` and save it in checkpoints, resolved configs, and prediction `.npz` files. Dataset point subsampling uses NumPy after this seed is set.
- Smoke train command:

```bash
python scripts/train_dccvt_hybrid_direct.py \
  --config configs/neural_hybrid_direct_v1.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --mesh-ids 313444 \
  --checkpoint-dir outputs/neural_dccvt/hybrid_direct_v1/checkpoints_smoke_313444 \
  --epochs 1 \
  --batch-size 1 \
  --seed 69
```

- Inference smoke command:

```bash
python scripts/infer_dccvt_hybrid_direct.py \
  --checkpoint outputs/neural_dccvt/hybrid_direct_v1/checkpoints_smoke_313444/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/313444.npz \
  --output-dir outputs/neural_dccvt/hybrid_direct_v1/infer_313444_smoke \
  --no-extract \
  --seed 69
```

- Optional mesh fine-tuning: pass `--stage mesh --batch-size 1`; this reuses DCCVT clipped-mesh Chamfer, CVT, and SDF smoothness losses and requires the CUDA/gDel3D runtime.
- Assumptions: inputs and labels use the existing normalized `[-1, 1]^3` convention; label site ordering matches the canonical DCCVT `torch.linspace(-1, 1, 32)` grid; KNN point features are reserved for a later ablation if voxel UDF/density underfits.

### Initial HotSpot Canonical Baseline

- Config: `configs/neural_hybrid_initial_hotspot.json`
- Extraction script: `scripts/extract_hybrid_initial_hotspot.py`
- Input caches: `outputs/neural_hotspot_sdf/thingi32_g33/*.npz`
- Split: `PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt`
- Output convention: per-mesh fields and extracted meshes are written under `outputs/neural_dccvt/hybrid_initial_hotspot/<mesh_id>/`.
- Purpose: measure the initial state before neural prediction or DCCVT optimization by using exact canonical sites and HotSpot SDF values sampled at those sites.
- Field definition: `sites = make_canonical_sites(grid_n)` and `sites_sdf = trilinear_interpolate_sdf(hotspot_sdf_grid, sites)`.
- Seed behavior: extraction records seed `69` in `resolved_config.json`, per-mesh field bundles, and `summary.json`; no stochastic site perturbation is applied.
- Metric status: evaluated with `PoNQ-main/src/eval/eval_HOTSPOT.py` over the 31-shape HotSpot/Thingi32 split.
- Metric outputs:
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_intDCCVT_ponq_thingi.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_intDCCVT_raw.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_intDCCVT_bbox_aligned.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_intDCCVT_hotspot_summary.csv`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_projDCCVT_ponq_thingi.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_projDCCVT_raw.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_projDCCVT_bbox_aligned.npy`
  - `PoNQ-main/src/eval/results/results_HybridInitialHotSpot_projDCCVT_hotspot_summary.csv`
- Default `intDCCVT` result:
  - `ponq_thingi`: `CD x 1e-5 = 274.530`, `F1 = 0.050`, `NC = 0.744`, `ECD = 2.269`, `EF1 = 0.000`
  - `raw`: `CD x 1e-5 = 1099.273`, `F1 = 0.020`, `NC = 0.744`, `ECD = 9.076`, `EF1 = 0.000`
  - `bbox_aligned`: `CD x 1e-5 = 123.232`, `F1 = 0.119`, `NC = 0.799`, `ECD = 9.073`, `EF1 = 0.000`
- Smoke extraction command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /tmp/dccvt-venv/bin/python \
  scripts/extract_hybrid_initial_hotspot.py \
  --config configs/neural_hybrid_initial_hotspot.json \
  --mesh-ids 313444 \
  --output-root outputs/neural_dccvt/hybrid_initial_hotspot_smoke \
  --overwrite
```

### Hybrid Direct Channel Ablation

- Runner: `scripts/run_hybrid_direct_channel_ablation.py`
- Output convention: per-run checkpoints and resolved configs are written to `outputs/neural_dccvt/hybrid_direct_ablation/<run_name>/` unless `--output-root` is overridden.
- Configs:
  - `hotspot_sdf`: `configs/neural_hybrid_direct_ablation_hotspot_sdf.json`
  - `hotspot_point_udf`: `configs/neural_hybrid_direct_ablation_hotspot_point_udf.json`
  - `hotspot_point_udf_abs`: `configs/neural_hybrid_direct_ablation_hotspot_point_udf_abs.json`
  - `hotspot_point_udf_confidence`: `configs/neural_hybrid_direct_ablation_hotspot_point_udf_confidence.json`
  - `full`: `configs/neural_hybrid_direct_ablation_full.json`
- Seed behavior: the runner forwards `--seed` to each training process; each checkpoint and `resolved_config.json` records that seed and its exact `channel_names`.
- Parallel behavior: pass `--parallel` to auto-detect free GPUs with `nvidia-smi`; the runner assigns one ablation per GPU, writes each run log to `<output-root>/<run_name>/train.log`, and waits according to `--poll-seconds` if no GPU has at least `--min-free-gb` free.
- Unchanged behavior: losses, supervised labels, dataset filters, metrics, extraction code, and prediction output formats are unchanged; only the model input channels differ.
- Dry-run command:

```bash
python scripts/run_hybrid_direct_channel_ablation.py \
  --dry-run \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --mesh-ids 313444 \
  --epochs 1 \
  --batch-size 1 \
  --seed 69 \
  --save-every 0
```

- Smoke train command:

```bash
python scripts/run_hybrid_direct_channel_ablation.py \
  --output-root outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444 \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --mesh-ids 313444 \
  --epochs 1 \
  --batch-size 1 \
  --seed 69 \
  --save-every 0
```

- Parallel full command:

```bash
PYTHONUNBUFFERED=1 PoNQ-main/.venv/bin/python \
  scripts/run_hybrid_direct_channel_ablation.py \
  --parallel \
  --devices auto \
  --min-free-gb 20 \
  --poll-seconds 60 \
  --output-root outputs/neural_dccvt/hybrid_direct_ablation \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --stage supervised \
  --epochs 300 \
  --batch-size 1 \
  --lr 6.4e-5 \
  --seed 69 \
  --save-every 25
```

- Full supervised checkpoint: `outputs/neural_dccvt/hybrid_direct_v1/full_supervised/latest.pt`
- Default full extraction output: `outputs/neural_dccvt/hybrid_direct_v1/extract_full/<mesh_id>/DCCVT_0_hybrid_direct_intDCCVT_cvt100_sdfsmooth100.obj`
- Default flat evaluation view: `PoNQ-main/out_hybrid_direct/HybridDirect_full_supervised_intDCCVT/<mesh_id>.obj`
- Default full diagnostic metric outputs:
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_ponq_thingi.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_raw.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_bbox_aligned.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_hotspot_summary.csv`
- Default full diagnostic result over 31 shapes using `PoNQ-main/src/eval/eval_HOTSPOT.py` with `100000` samples:
  - `ponq_thingi`: `CD x 1e-5 = 465.425`, `F1 = 0.027`, `NC = 0.638`, `ECD = 2.269`, `EF1 = 0.000`
  - `raw`: `CD x 1e-5 = 1861.174`, `F1 = 0.010`, `NC = 0.638`, `ECD = 9.076`, `EF1 = 0.000`
  - `bbox_aligned`: `CD x 1e-5 = 77.289`, `F1 = 0.095`, `NC = 0.707`, `ECD = 9.079`, `EF1 = 0.000`
- Diagnostic baseline extraction output: `outputs/neural_dccvt/hybrid_direct_v1/extract_full/<mesh_id>/DCCVT_0_hybrid_direct_projDCCVT_cvt100_sdfsmooth100.obj`
- Diagnostic baseline flat evaluation view: `PoNQ-main/out_hybrid_direct/HybridDirect_full_supervised_projDCCVT/<mesh_id>.obj`
- Diagnostic baseline metric outputs:
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_projDCCVT_ponq_thingi.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_projDCCVT_raw.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_projDCCVT_bbox_aligned.npy`
  - `PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_projDCCVT_hotspot_summary.csv`
- Diagnostic baseline result over 31 shapes using `PoNQ-main/src/eval/eval_HOTSPOT.py` with `100000` samples:
  - `ponq_thingi`: `CD x 1e-5 = 377.587`, `F1 = 0.041`, `NC = 0.592`, `ECD = 2.268`, `EF1 = 0.000`
  - `raw`: `CD x 1e-5 = 1510.220`, `F1 = 0.015`, `NC = 0.593`, `ECD = 9.075`, `EF1 = 0.000`
  - `bbox_aligned`: `CD x 1e-5 = 91.150`, `F1 = 0.095`, `NC = 0.622`, `ECD = 9.078`, `EF1 = 0.000`
- Interpretation: `intDCCVT` is the default mesh for evaluation. It improves bbox-aligned CD and normal consistency over the `projDCCVT` diagnostic baseline, but still trails the HotSpot PoNQ baseline. The near-zero edge F1 suggests the direct DCCVT field needs additional supervision or mesh-loss fine-tuning before it is competitive.

## PoNQ ABC Reproduction Smoke Test

- Config: `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_smoke.yaml`
- Dataset: `/tmp/ponq_abc/gt_Quadrics/`
- Names file: `PoNQ-main/src/utils/abc_watertight_train_smoke.txt`
- Purpose: verify the local PoNQ ABC HDF5 pipeline and one-epoch training path before full three-phase ABC training.
- Output convention: smoke artifacts are moved to `PoNQ-main/logs/model_smoke.pt` and `PoNQ-main/logs/loss_smoke.png`.
- Seed behavior: PoNQ data subsampling and DataLoader shuffling are currently unseeded.

## PoNQ ABC Reproduction Batch-Size 32 Resume

- Configs:
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_1_bs32_resume.yaml`
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_2_bs32.yaml`
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_3_bs32.yaml`
- Dataset: `/tmp/ponq_abc/gt_Quadrics/`
- Names file: `PoNQ-main/src/utils/abc_watertight_train.txt`
- Purpose: continue PoNQ ABC reproduction from the phase-1 epoch-2 checkpoint using a larger batch size to improve GPU utilization on an RTX A6000.
- Resume checkpoint: `PoNQ-main/models/model_multiple_quadrics_split_phase1_bs16_epoch002.pt`
- Output convention: run logs and copied configs are stored under `PoNQ-main/logs/abc_retrain_bs32_<timestamp>/`; final checkpoint is written to `PoNQ-main/data/pretrained_PoNQ_ABC_retrained.pt`.
- Seed behavior: PoNQ data subsampling and DataLoader shuffling are currently unseeded.
- Limitation: optimizer state is not restored because the original PoNQ training script only saves model weights.

## PoNQ ABC Reproduction Batch-Size 48 Resume

- Configs:
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_1_bs48_resume.yaml`
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_2_bs48.yaml`
  - `PoNQ-main/configs/local_abc_cnn_multiple_quadrics_split_3_bs48.yaml`
- Eval config: `PoNQ-main/configs/eval_cnn_retrained.yaml`
- Dataset: `/tmp/ponq_abc/gt_Quadrics/`
- Names file: `PoNQ-main/src/utils/abc_watertight_train.txt`
- ABC eval names file: `PoNQ-main/src/eval/abc_eval_last20.txt`
- Purpose: continue PoNQ ABC reproduction from the phase-1 epoch-2 checkpoint using batch size 48 after batch size 32 showed substantial remaining GPU memory headroom.
- Resume checkpoint: `PoNQ-main/models/model_multiple_quadrics_split_phase1_bs16_epoch002.pt`
- Output convention: run logs and copied configs are stored under `PoNQ-main/logs/abc_retrain_bs48_<timestamp>/`; final checkpoint is written to `PoNQ-main/data/pretrained_PoNQ_ABC_retrained.pt`.
- Seed behavior: PoNQ data subsampling and DataLoader shuffling are currently unseeded.
- Limitation: optimizer state is not restored because the original PoNQ training script only saves model weights.
- Baseline evaluation note: official ABC validation uses the last 20% of `src/eval/abc_ordered.txt`, so missing eval `model.obj` and HDF5 files must be generated before mesh generation and metrics.

## PoNQ Official ABC Checkpoint Baseline Evaluation

- Config: `PoNQ-main/configs/eval_cnn_official_abc.yaml`
- Checkpoint: `PoNQ-main/data/pretrained_PoNQ_ABC.pt`
- Dataset: `/tmp/ponq_abc/gt_Quadrics/`
- Ground-truth OBJ root: `/export/livia/home/vision/Wcharawi/datasets/abc/raw_obj/`
- ABC eval names file: `PoNQ-main/src/eval/abc_eval_last20.txt`
- Purpose: generate ABC validation meshes and metrics for the paper-provided PoNQ checkpoint so it can be compared against `PoNQ-main/data/pretrained_PoNQ_ABC_retrained.pt` and the paper.
- Output convention: run logs and copied configs are stored under `PoNQ-main/logs/official_abc_eval_<timestamp>/`; generated meshes are written under `PoNQ-main/out_official/ABC_pretrained_PoNQ_ABC_32/` and `PoNQ-main/out_official/ABC_pretrained_PoNQ_ABC_64/`.
- Metric outputs: `PoNQ-main/src/eval/results/results_ABC_pretrained_PoNQ_ABC_32.npy` and `PoNQ-main/src/eval/results/results_ABC_pretrained_PoNQ_ABC_64.npy`.
- Seed behavior: mesh generation and metrics are run in eval mode with fixed validation ordering from PoNQ's ABC split; no new training seed is involved.

## PoNQ Retrained Checkpoint On HotSpot SDF Caches

- Config: `PoNQ-main/configs/eval_hotspot_ponq_retrained.yaml`
- Generator: `PoNQ-main/src/utils/generate_mesh_hotspot_ponq.py`
- Checkpoint: `PoNQ-main/data/pretrained_PoNQ_ABC_retrained.pt`
- Input caches: `/export/livia/home/vision/Wcharawi/dev/DCCVT/outputs/neural_hotspot_sdf/thingi32_g33/*.npz`
- Ground-truth OBJ root: `/export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32/`
- Names files:
  - full: `PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt`
  - smoke: `PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt`
- Purpose: test the retrained ABC PoNQ network directly on existing HotSpot dense SDF grids without converting them to ABC/Thingi HDF5 files.
- Output convention: generated PoNQ `.pt` and `.obj` files are written under `PoNQ-main/out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32/`; `metadata.json` and `generation_summary.csv` are saved in the same folder.
- Metric output: `PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32.npy`
- Full run result over 31 cached HotSpot/Thingi32 shapes: `CD x 1e-5 = 307.813`, `F1 = 0.042`, `NC = 0.728`, `ECD = 0.378`, `EF1 = 0.029`.
- Seed behavior: eval-only run over fixed sorted cache IDs; no training seed is involved.
- Limitation: only grid `33` caches are available in this run, so only `_32` PoNQ outputs are generated.

## PoNQ HotSpot Metric Alignment Diagnostics

- Evaluator: `PoNQ-main/src/eval/eval_HOTSPOT.py`
- Prediction folder: `PoNQ-main/out_hotspot_retrained/HotSpot_pretrained_PoNQ_ABC_retrained_32/`
- Ground-truth OBJ root: `/export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32/`
- Names file: `PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt`
- Purpose: compare the strict PoNQ/Thingi metric convention against raw and bbox-aligned diagnostic metrics to expose scale mismatch between HotSpot cache coordinates, PoNQ outputs, and GT meshes.
- Output files:
  - `PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_ponq_thingi.npy`
  - `PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_raw.npy`
  - `PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_bbox_aligned.npy`
  - `PoNQ-main/src/eval/results/results_HotSpot_pretrained_PoNQ_ABC_retrained_32_hotspot_summary.csv`
- Full diagnostic result over 31 shapes:
  - `ponq_thingi`: `CD x 1e-5 = 307.902`, `F1 = 0.042`, `NC = 0.729`, `ECD = 0.377`, `EF1 = 0.029`
  - `raw`: `CD x 1e-5 = 1232.188`, `F1 = 0.015`, `NC = 0.729`, `ECD = 1.510`, `EF1 = 0.008`
  - `bbox_aligned`: `CD x 1e-5 = 14.024`, `F1 = 0.261`, `NC = 0.896`, `ECD = 0.240`, `EF1 = 0.079`
- Interpretation: `bbox_aligned` is diagnostic only. The large CD/F1 improvement indicates the poor strict score is strongly affected by a uniform scale/translation mismatch, not only by local mesh quality.

## HotSpot Hybrid Comparison Table

- Evaluator: `PoNQ-main/src/eval/eval_HOTSPOT.py`
- Split: `PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt`
- Count: `31`
- Sample count: `100000`
- DCCVT rows use the default `intDCCVT` mesh. The `projDCCVT` diagnostic result files are retained separately under `PoNQ-main/src/eval/results/`.

| case | mesh | mode | count | CD x1e-5 | F1 | NC | ECD | EF1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Hybrid initial HotSpot | intDCCVT | `ponq_thingi` | 31 | 274.530 | 0.050 | 0.744 | 2.269 | 0.000 |
| Hybrid initial HotSpot | intDCCVT | `raw` | 31 | 1099.273 | 0.020 | 0.744 | 9.076 | 0.000 |
| Hybrid initial HotSpot | intDCCVT | `bbox_aligned` | 31 | 123.232 | 0.119 | 0.799 | 9.073 | 0.000 |
| HotSpot PoNQ ABC retrained | PoNQ | `ponq_thingi` | 31 | 307.902 | 0.042 | 0.729 | 0.377 | 0.029 |
| HotSpot PoNQ ABC retrained | PoNQ | `raw` | 31 | 1232.188 | 0.015 | 0.729 | 1.510 | 0.008 |
| HotSpot PoNQ ABC retrained | PoNQ | `bbox_aligned` | 31 | 14.024 | 0.261 | 0.896 | 0.240 | 0.079 |
| HybridDirect full supervised | intDCCVT | `ponq_thingi` | 31 | 465.425 | 0.027 | 0.638 | 2.269 | 0.000 |
| HybridDirect full supervised | intDCCVT | `raw` | 31 | 1861.174 | 0.010 | 0.638 | 9.076 | 0.000 |
| HybridDirect full supervised | intDCCVT | `bbox_aligned` | 31 | 77.289 | 0.095 | 0.707 | 9.079 | 0.000 |
| Ablation `hotspot_sdf` | intDCCVT | `ponq_thingi` | 31 | 436.067 | 0.028 | 0.641 | 2.269 | 0.000 |
| Ablation `hotspot_sdf` | intDCCVT | `raw` | 31 | 1745.309 | 0.010 | 0.640 | 9.076 | 0.000 |
| Ablation `hotspot_sdf` | intDCCVT | `bbox_aligned` | 31 | 77.255 | 0.103 | 0.703 | 9.072 | 0.000 |
| Ablation `hotspot_point_udf` | intDCCVT | `ponq_thingi` | 31 | 390.553 | 0.036 | 0.632 | 2.269 | 0.000 |
| Ablation `hotspot_point_udf` | intDCCVT | `raw` | 31 | 1562.331 | 0.013 | 0.632 | 9.076 | 0.000 |
| Ablation `hotspot_point_udf` | intDCCVT | `bbox_aligned` | 31 | 63.832 | 0.117 | 0.709 | 9.076 | 0.000 |
| Ablation `hotspot_point_udf_abs` | intDCCVT | `ponq_thingi` | 31 | 400.670 | 0.032 | 0.631 | 2.269 | 0.000 |
| Ablation `hotspot_point_udf_abs` | intDCCVT | `raw` | 31 | 1602.415 | 0.012 | 0.631 | 9.075 | 0.000 |
| Ablation `hotspot_point_udf_abs` | intDCCVT | `bbox_aligned` | 31 | 63.841 | 0.116 | 0.706 | 9.076 | 0.000 |
| Ablation `hotspot_point_udf_confidence` | intDCCVT | `ponq_thingi` | 31 | 396.248 | 0.033 | 0.630 | 2.268 | 0.000 |
| Ablation `hotspot_point_udf_confidence` | intDCCVT | `raw` | 31 | 1584.442 | 0.012 | 0.631 | 9.075 | 0.000 |
| Ablation `hotspot_point_udf_confidence` | intDCCVT | `bbox_aligned` | 31 | 62.547 | 0.116 | 0.709 | 9.076 | 0.000 |

| case | mesh | mode | count | CD x1e-5 | F1 | NC | ECD | EF1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Hybrid initial HotSpot | intDCCVT | `bbox_aligned` | 31 | 123.232 | 0.119 | 0.799 | 9.073 | 0.000 |
| HotSpot PoNQ ABC retrained | PoNQ | `bbox_aligned` | 31 | 14.024 | 0.261 | 0.896 | 0.240 | 0.079 |
| HybridDirect full supervised | intDCCVT | `bbox_aligned` | 31 | 77.289 | 0.095 | 0.707 | 9.079 | 0.000 |
| Ablation `hotspot_sdf` | intDCCVT | `bbox_aligned` | 31 | 77.255 | 0.103 | 0.703 | 9.072 | 0.000 |
| Ablation `hotspot_point_udf` | intDCCVT | `bbox_aligned` | 31 | 63.832 | 0.117 | 0.709 | 9.076 | 0.000 |
| Ablation `hotspot_point_udf_abs` | intDCCVT | `bbox_aligned` | 31 | 63.841 | 0.116 | 0.706 | 9.076 | 0.000 |
| Ablation `hotspot_point_udf_confidence` | intDCCVT | `bbox_aligned` | 31 | 62.547 | 0.116 | 0.709 | 9.076 | 0.000 |
