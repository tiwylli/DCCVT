# DCCVT Neural Implementation Guide

## Purpose

`dccvt/neural/` contains an experimental PoNQ-style neural implementation for
predicting DCCVT Voronoi generator sites from a dense HotSpot SDF grid.

The learned HotSpot near-surface iterative refinement path is documented
separately in [Neural Iterative Refinement Guide](neural_iterative_refinement_guide.md).

The implemented pipeline is:

```text
mesh + HotSpot weights
  -> dense normalized SDF cache
  -> DCCVTPoNQNet
  -> per-cell sites + activity scores
  -> sampled site SDF values
  -> optional DCCVT mesh extraction
```

The current model is site-only. It predicts `K` sites per grid cell and one
activity score per cell. It does not predict `sites_sdf`; inference samples
`sites_sdf` from the dense cached SDF grid using trilinear interpolation.

This is not the same as a direct point-cloud network. Point samples are used as
training targets and for output `target.ply` files, but the model input is a
dense SDF grid.

## High-Level Architecture

The neural path has four stages:

1. Precompute HotSpot SDF caches with `scripts/precompute_hotspot_sdf.py`.
2. Train `DCCVTPoNQNet` with `scripts/train_dccvt_neural.py`.
3. Run inference with `scripts/infer_dccvt_neural.py`.
4. Optionally pass predicted sites and interpolated SDF values into the shared
   DCCVT mesh extractor.

Stage 1 training uses a PoNQ-style warm-start loss:

- symmetric squared Chamfer distance between selected predicted sites and
  HotSpot manifold samples
- binary cross entropy for cell activity
- offset regularization toward the center of each cell
- domain penalty for predicted sites outside `[-1, 1]^3`

Stage 2 training adds a differentiable DCCVT fine-tuning term:

- predicted sites are selected from active cells
- SDF values are interpolated from the dense grid
- Delaunay, clipped mesh, Chamfer, and CVT losses are computed through the
  existing DCCVT geometry code

By default, Stage 2 freezes the encoder and activity head and trains only the
site head. Pass `--stage2-train-encoder` to train the whole network.

## File-by-File Explanation

### `dccvt/neural/__init__.py`

Exports the main model and grid utilities:

- `DCCVTPoNQNet`
- `default_near_surface_threshold`
- `make_cell_lower_corners`
- `make_coord_grid`
- `make_gt_activity_mask_np`
- `make_near_surface_mask_np`
- `trilinear_interpolate_sdf`

### `dccvt/neural/grid.py`

Defines the normalized grid convention and mask/interpolation helpers.

Important behavior:

- grids live in normalized `[-1, 1]^3`
- `grid_n` is the number of SDF vertices per axis
- the number of cells per axis is `grid_n - 1`
- cell size is `2 / (grid_n - 1)`
- the default near-surface threshold is `cell_size * sqrt(3)`
- `make_near_surface_mask_np()` marks cells whose eight SDF corner values are
  all within the threshold
- `make_gt_activity_mask_np()` marks cells containing target samples
- `trilinear_interpolate_sdf()` samples dense SDF values at predicted sites

### `dccvt/neural/models.py`

Defines the PoNQ-style dense 3D CNN.

Important classes:

- `ResNetBlock`: small 1x1x1 residual block.
- `CellDecoder`: decodes per-cell values from dense cell features.
- `DCCVTPoNQNet`: consumes `(B, 1, G, G, G)` or `(B, G, G, G)` SDF grids and
  returns:
  - `sites`: shape `(B, (G - 1)^3, K, 3)`
  - `raw_offsets`: unbounded offset logits
  - `offset_fraction`: sigmoid-constrained offsets inside each cell
  - `activity_logits`: shape `(B, (G - 1)^3)`
  - `activity`: sigmoid of activity logits
  - `cell_lower_corners`: lower corner coordinates for all cells

The first convolution has `kernel_size=2`, so vertex-grid input with side
length `G` becomes a cell grid with side length `G - 1`.

### `dccvt/neural/dataset.py`

Loads cached `.npz` records created by precompute.

Important functions/classes:

- `resolve_cache_files(cache_root, mesh_ids=None, split_file=None)`: resolves
  cache paths from explicit mesh ids, a split file, or all `.npz` files in a
  directory.
- `HotspotSDFDataset`: returns PyTorch tensors for `sdf_grid`,
  `near_surface_mask`, `gt_activity_mask`, and `target_points`.

`target_subsample` randomly subsamples target points with `numpy.random.choice`.
No local seed is set in this dataset.

### `dccvt/neural/precompute.py`

Builds dense SDF cache records by loading a HotSpot model and evaluating it on a
regular SDF grid.

Important functions:

- `sample_hotspot_sdf_grid(model, grid_n, batch_size)`: evaluates the HotSpot
  model on the dense coordinate grid.
- `build_hotspot_cache_record(...)`: returns a record containing mesh metadata,
  SDF grid, masks, and target samples.
- `main(...)`: CLI entry point used by `scripts/precompute_hotspot_sdf.py`.

### `dccvt/neural/losses.py`

Defines training losses.

Important functions:

- `chamfer_distance_points(points_a, points_b)`: symmetric squared Chamfer
  distance. Uses `pytorch3d.ops.knn_points` when available, otherwise falls
  back to chunked `torch.cdist`.
- `stage1_site_loss(...)`: warm-start loss for site and activity prediction.
- `select_cells_for_dccvt(...)`: bounds the number of selected cells for Stage
  2 fine-tuning.
- `dccvt_finetune_loss(...)`: calls the existing DCCVT geometry code and adds
  differentiable Chamfer plus CVT losses.

### `dccvt/neural/train.py`

Training CLI and checkpointing.

Important functions:

- `build_arg_parser()`: defines all training flags.
- `build_model_from_args(args)`: creates `DCCVTPoNQNet`.
- `save_checkpoint(...)`: writes checkpoint payloads.
- `main(...)`: resolves caches, builds the dataset/dataloader, trains, and
  saves checkpoints.

Checkpoint payload keys:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `model_config`
- `args`
- `stats`

### `dccvt/neural/infer.py`

Inference and optional mesh extraction.

Important functions:

- `_load_checkpoint(path, device)`: restores `DCCVTPoNQNet` from checkpoint.
- `_load_cache(path)`: reads a cache and reconstructs missing masks when
  possible.
- `_select_active_cells(...)`: supports `activity`, `near-surface`, and `topk`
  cell selection.
- `run_inference(...)`: saves neural predictions and optionally extracts DCCVT
  meshes.
- `main(...)`: CLI entry point used by `scripts/infer_dccvt_neural.py`.

### Script Wrappers

- `scripts/precompute_hotspot_sdf.py`: wrapper for `dccvt.neural.precompute`.
- `scripts/train_dccvt_neural.py`: wrapper for `dccvt.neural.train`.
- `scripts/train_dccvt_ponq.py`: alias wrapper for `dccvt.neural.train`.
- `scripts/infer_dccvt_neural.py`: wrapper for `dccvt.neural.infer`.
- `scripts/infer_dccvt_ponq.py`: alias wrapper for `dccvt.neural.infer`.
- `scripts/generate_neural_labels.py`: auxiliary script that runs full DCCVT
  optimization to create fixed-size labels under `outputs/neural_labels/n32`.
  The current dense-SDF trainer does not consume these labels.

## Command-Line Entry Points

All commands below assume the working directory is the repository root.

### Precompute Dense HotSpot SDF Caches

Command:

```bash
python scripts/precompute_hotspot_sdf.py \
  --mesh-ids 313444 \
  --mesh-root mesh/thingi32 \
  --hotspot-root hotspots_model/thingi32 \
  --output-root outputs/neural_hotspot_sdf/g33 \
  --grid-n 33
```

Required inputs:

- `<mesh-root>/<mesh_id>.ply`
- `<hotspot-root>/<mesh_id>.pth`
- HotSpot dependency under `3rdparty/HotSpot`

Generated outputs:

- `<output-root>/<mesh_id>.npz`

The output `.npz` contains:

- `mesh_id`
- `mesh_path`
- `hotspot_weights_path`
- `grid_n`
- `domain_min`
- `domain_max`
- `near_surface_threshold`
- `sdf_grid`
- `near_surface_mask`
- `gt_activity_mask`
- `target_points`

This command modifies files under `--output-root`. It requires the HotSpot model
dependencies and imports the DCCVT runtime device during startup.

Parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mesh-ids` | `None` | Comma or space separated mesh ids. If omitted, uses `DEFAULTS["mesh_ids"]`. |
| `--mesh-ids-file` | `None` | Text file containing mesh ids. Comments and blank lines are ignored. |
| `--mesh-root` | `DEFAULTS["mesh"]` | Root containing `<mesh_id>.ply`. |
| `--hotspot-root` | `DEFAULTS["trained_HotSpot"]` | Root containing HotSpot `.pth` files. In this repo layout, `hotspots_model/thingi32` is often the needed value. |
| `--hotspot-suffix` | `.pth` | Suffix appended to mesh ids when resolving weights. |
| `--output-root` | `outputs/neural_hotspot_sdf/g33` | Directory for generated cache files. |
| `--grid-n` | `33` | SDF grid vertex count per axis. |
| `--sample-count` | `200000` | Number of manifold points requested from HotSpot dataset loading. |
| `--max-amount-sites` | `32` | Passed into HotSpot loading and sample sizing logic. |
| `--query-batch-size` | `65536` | Number of SDF query points evaluated per HotSpot forward pass. |
| `--near-surface-threshold` | `None` | If omitted, uses `2 / (grid_n - 1) * sqrt(3)`. |
| `--overwrite` | `False` | Recompute existing cache files. |

### Train the Neural Model

Stage 1 command:

```bash
python scripts/train_dccvt_neural.py \
  --cache-root outputs/neural_hotspot_sdf/g33 \
  --mesh-ids 313444 \
  --checkpoint-dir outputs/neural_dccvt/ponq_stage1 \
  --epochs 1 \
  --batch-size 1 \
  --device auto
```

Stage 2 command:

```bash
python scripts/train_dccvt_neural.py \
  --cache-root outputs/neural_hotspot_sdf/g33 \
  --mesh-ids 313444 \
  --checkpoint-dir outputs/neural_dccvt/ponq_stage2 \
  --resume outputs/neural_dccvt/ponq_stage1/latest.pt \
  --stage 2 \
  --epochs 1 \
  --batch-size 1 \
  --device auto
```

Required inputs:

- cache `.npz` files generated by precompute
- optional checkpoint when `--resume` is used

Generated outputs:

- `<checkpoint-dir>/latest.pt`
- `<checkpoint-dir>/epoch_XXXX.pt` every `--save-every` epochs and on the final
  local epoch

This command modifies files under `--checkpoint-dir`. Stage 1 can run on CPU if
all imported dependencies are available. Stage 2 calls DCCVT geometry and is
intended for the CUDA/gDel3D environment.

Parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cache-root` | required | Directory of precomputed HotSpot SDF `.npz` caches. |
| `--mesh-ids` | `None` | Optional comma or space separated cache stems. |
| `--split-file` | `None` | Optional text file of cache stems. |
| `--checkpoint-dir` | `outputs/neural_dccvt/checkpoints` | Directory for checkpoints. |
| `--resume` | `None` | Checkpoint to load before training. |
| `--resume-optimizer` | `False` | Restore optimizer state from `--resume`. |
| `--stage` | `1` | Training stage, either `1` or `2`. |
| `--epochs` | `100` | Number of epochs to run from the start or resume point. |
| `--batch-size` | `4` | DataLoader batch size. |
| `--target-subsample` | `20000` | Maximum target points loaded per shape. |
| `--lr` | `6.4e-5` | AdamW learning rate. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--num-workers` | `0` | DataLoader workers. |
| `--grid-n` | `33` | Model SDF grid vertex count. Used for new models, not for resumed checkpoint configs. |
| `--k` | `4` | Sites predicted per cell. |
| `--feature-dim` | `128` | Hidden feature channels. |
| `--encoder-layers` | `5` | Number of 3x3x3 encoder layers after the first cell-grid convolution. |
| `--decoder-layers` | `3` | Number of residual decoder blocks per head. |
| `--w-chamfer` | `100.0` | Stage 1 Chamfer weight. |
| `--w-occupancy` | `1.0` | Activity BCE weight. Set to `0` automatically in default Stage 2. |
| `--w-offset` | `0.1` | Offset regularization weight. |
| `--w-domain` | `1.0` | Domain penalty weight. |
| `--w-dccvt-chamfer` | `1000.0` | Stage 2 DCCVT Chamfer weight. |
| `--w-dccvt-cvt` | `100.0` | Stage 2 DCCVT CVT weight. |
| `--w-dccvt` | `1.0` | Multiplier for the Stage 2 DCCVT fine-tuning loss. |
| `--max-dccvt-sites` | `4096` | Maximum selected sites per shape for Stage 2. |
| `--stage2-train-encoder` | `False` | Train encoder and activity head during Stage 2. |
| `--save-every` | `10` | Epoch interval for numbered checkpoints. `0` disables interval saves. |

### Run Inference

Inference from a precomputed cache:

```bash
python scripts/infer_dccvt_neural.py \
  --checkpoint outputs/neural_dccvt/ponq_stage1/latest.pt \
  --cache outputs/neural_hotspot_sdf/g33/313444.npz \
  --output-dir outputs/neural_dccvt/infer_313444 \
  --selection-mode activity \
  --activity-threshold 0.5
```

Inference without mesh extraction:

```bash
python scripts/infer_dccvt_neural.py \
  --checkpoint outputs/neural_dccvt/ponq_stage1/latest.pt \
  --cache outputs/neural_hotspot_sdf/g33/313444.npz \
  --output-dir outputs/neural_dccvt/infer_313444_no_extract \
  --no-extract
```

Inference with on-the-fly cache creation:

```bash
python scripts/infer_dccvt_neural.py \
  --checkpoint outputs/neural_dccvt/ponq_stage1/latest.pt \
  --mesh mesh/thingi32/313444 \
  --hotspot-weights hotspots_model/thingi32/313444.pth \
  --output-dir outputs/neural_dccvt/infer_313444
```

Required inputs:

- checkpoint `.pt`
- either `--cache`, or both `--mesh` and `--hotspot-weights`

Generated outputs:

- `<mesh_id>_neural_dccvt_prediction.npz`
- if extraction succeeds:
  - `DCCVT_0_neural_intDCCVT_cvt<W>_sdfsmooth<W>.obj`
  - `DCCVT_0_neural_intDCCVT_cvt<W>_sdfsmooth<W>.npz`
  - `DCCVT_0_neural_projDCCVT_cvt<W>_sdfsmooth<W>.obj`
  - `DCCVT_0_neural_projDCCVT_cvt<W>_sdfsmooth<W>.npz`
  - `target.ply`

The prediction `.npz` contains:

- `sites`
- `sites_sdf`
- `active_cell_mask`
- `activity`
- `near_surface_mask`
- `sdf_grid`
- `diagnostics`
- `mesh_id`

This command modifies files under `--output-dir`. Inference with `--no-extract`
only needs model inference and cache loading. Mesh extraction needs enough
predicted sites, both positive and negative sampled SDF values, and the DCCVT
CUDA/gDel3D runtime.

Parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Neural checkpoint path. |
| `--cache` | `None` | Precomputed HotSpot SDF `.npz` cache. |
| `--output-dir` | required | Directory for predictions and optional meshes. |
| `--activity-threshold` | `0.5` | Activity cutoff for `activity` selection mode. |
| `--selection-mode` | `activity` | One of `activity`, `near-surface`, or `topk`. |
| `--max-sites` | `None` | Maximum selected sites. Required by `topk` mode. |
| `--fallback-cells` | `64` | Number of top-activity cells used when selection finds no active cells. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--no-extract` | `False` | Save prediction only and skip DCCVT extraction. |
| `--w-cvt` | `100.0` | CVT weight used in extracted output filename metadata. |
| `--w-sdfsmooth` | `100.0` | SDF smoothing weight used in extracted output filename metadata. |
| `--mesh` | `None` | Optional mesh path/stem for on-the-fly cache creation. |
| `--hotspot-weights` | `None` | Optional HotSpot `.pth` for on-the-fly cache creation. |
| `--grid-n` | `33` | On-the-fly cache grid size. |
| `--sample-count` | `200000` | On-the-fly HotSpot manifold sample count. |
| `--max-amount-sites` | `32` | On-the-fly HotSpot loading parameter. |
| `--query-batch-size` | `65536` | On-the-fly SDF query batch size. |

### Generate Fixed-Size DCCVT Labels

Command:

```bash
python scripts/generate_neural_labels.py \
  --mesh-ids 313444 \
  --mesh-root mesh/thingi32 \
  --hotspot-root hotspots_model/thingi32 \
  --output-root outputs/neural_labels/n32 \
  --num-centroids 32
```

Required inputs:

- `<mesh-root>/<mesh_id>.ply`
- `<hotspot-root>/<mesh_id>.pth`

Generated outputs:

- DCCVT init/final `.obj` and `.npz` files inside
  `<output-root>/<mesh_id>/`
- `target.ply`

This command runs full DCCVT optimization and modifies files under
`--output-root`. It is an auxiliary label-generation script; the current
`dccvt.neural.train` code trains from dense HotSpot SDF caches, not these label
folders.

Parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mesh-ids` | `None` | Comma or space separated mesh ids. If omitted, uses `DEFAULTS["mesh_ids"]`. |
| `--mesh-root` | `DEFAULTS["mesh"]` | Root containing mesh files. |
| `--hotspot-root` | `DEFAULTS["trained_HotSpot"] / "thingi32"` | Root containing HotSpot `.pth` files. |
| `--output-root` | `outputs/neural_labels/n32` | Directory for label runs. |
| `--num-iterations` | `DEFAULTS["num_iterations"]` (`1000`) | DCCVT optimization steps. |
| `--num-centroids` | `32` | Sites per axis for full DCCVT label generation. |
| `--max-amount-sites` | `DEFAULTS["max_amount_sites"]` (`32`) | HotSpot loading parameter. |
| `--w-chamfer` | `DEFAULTS["w_chamfer"]` (`1000`) | DCCVT Chamfer weight. |
| `--w-cvt` | `DEFAULTS["w_cvt"]` (`100`) | DCCVT CVT weight. |
| `--w-sdfsmooth` | `DEFAULTS["w_sdfsmooth"]` (`100`) | SDF smoothing weight. |
| `--lr-sites` | `DEFAULTS["lr_sites"]` (`0.0005`) | Site optimizer learning rate. |
| `--seed` | `69` | Base seed. The script adds the mesh index to this value. |
| `--overwrite` | `False` | Re-run meshes even if final labels exist. |

## Required Inputs

For precompute:

- normalized mesh files readable by HotSpot, normally `.ply`
- matching HotSpot `.pth` weights
- `3rdparty/HotSpot` available on disk

For training:

- `.npz` cache files with `sdf_grid`, `near_surface_mask`,
  `gt_activity_mask`, `target_points`, and `grid_n`

For inference:

- neural checkpoint with `model_state_dict` and `model_config`
- precomputed cache, or mesh plus HotSpot weights for on-the-fly cache creation

For mesh extraction:

- at least 5 selected predicted sites
- selected site SDF values containing both positive and negative signs
- CUDA/gDel3D runtime available

## Output Directory Conventions

Common conventions in the current code:

- precomputed caches: `outputs/neural_hotspot_sdf/g33/<mesh_id>.npz`
- Stage 1 checkpoints: `outputs/neural_dccvt/ponq_stage1/`
- Stage 2 checkpoints: `outputs/neural_dccvt/ponq_stage2/`
- inference outputs: `outputs/neural_dccvt/infer_<mesh_id>/`
- auxiliary full-DCCVT labels: `outputs/neural_labels/n32/<mesh_id>/`

These are conventions from defaults and examples, not enforced by config files.

## Minimal Smoke Test Command

Run the focused neural unit tests:

```bash
python -m pytest tests/test_neural_ponq.py
```

This test checks:

- model forward output shapes
- predicted site domain bounds
- trilinear interpolation on a linear field
- flattened near-surface and GT activity masks

This smoke test does not exercise HotSpot loading, CUDA, Delaunay, Stage 2, or
mesh extraction.

## Common Failure Cases

- `FileNotFoundError: Mesh file not found`: the mesh resolver expects a `.ply`
  file after applying `Path(mesh_path).with_suffix(".ply")`.
- `FileNotFoundError: HotSpot weights not found`: pass the correct
  `--hotspot-root`; in this repo, weights are commonly under
  `hotspots_model/thingi32/`.
- `HotSpot dependencies not found`: initialize submodules and check that
  `3rdparty/HotSpot` exists.
- `--selection-mode topk requires --max-sites`: top-k selection cannot infer a
  site budget.
- `Skipping DCCVT extraction`: inference found fewer than 5 sites or did not
  sample both positive and negative SDF values.
- Stage 2 failures with CPU-only environments: Stage 2 calls DCCVT geometry and
  Delaunay code that is intended for the CUDA/gDel3D setup.
- Very slow Chamfer loss: if PyTorch3D is unavailable, the fallback uses
  chunked `torch.cdist`.
- Cache/grid mismatch: a resumed checkpoint uses its saved `model_config`, so
  cache `grid_n` should match the checkpoint's expected input grid.

## GPU, CUDA, Memory, and Dataset Notes

- The repository-level DCCVT pipeline requires CUDA because `pygdel3d` is used
  for Delaunay tetrahedralization.
- `dccvt.device` selects `cuda:0` by default and raises unless CUDA is available
  or `DCCVT_DEVICE` is set. Precompute imports this device module at startup.
- Precompute and on-the-fly inference cache creation load HotSpot models and
  evaluate `grid_n^3` SDF samples.
- The default `grid_n=33` evaluates `35937` SDF vertices per shape.
- The model predicts `(grid_n - 1)^3 * k` candidate sites. With `grid_n=33` and
  `k=4`, this is `32768 * 4 = 131072` candidate sites before active-cell
  filtering.
- Stage 2 and extraction should use `--max-dccvt-sites` or `--max-sites` to
  bound selected sites.
- Training memory depends strongly on `grid_n`, `k`, `feature_dim`, batch size,
  target point count, and whether Stage 2 DCCVT losses are enabled.

## Known Assumptions and Limitations

- Inputs are assumed to be normalized consistently with the existing DCCVT and
  HotSpot pipeline over `[-1, 1]^3`.
- The model predicts sites only. SDF values are sampled from the cache at
  inference time.
- There is no explicit validation loop or held-out metric in
  `dccvt.neural.train`.
- There are no neural YAML/JSON config files in this repository. The neural
  implementation is configured through argparse flags.
- Training does not currently save a resolved standalone config file; it stores
  `vars(args)` inside checkpoints.
- `scripts/precompute_hotspot_sdf.py` and `scripts/train_dccvt_neural.py` do
  not set or log a seed.
- `HotspotSDFDataset` target subsampling uses NumPy random state without a
  local seed.
- `make_near_surface_mask_np()` marks cells only when all eight SDF corners are
  within the near-surface threshold. It is not a sign-change mask.
- `scripts/generate_neural_labels.py` creates full-DCCVT label outputs but is
  not wired into `dccvt.neural.train`.
- If a behavior is not described above, it needs verification from code before
  relying on it for an experiment.
