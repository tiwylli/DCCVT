# Experiments

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
