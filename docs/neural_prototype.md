# DCCVT Neural Prototype

This document describes the first neural component added to DCCVT. It is a research prototype, not a replacement for the optimization pipeline.

## Goal

The prototype learns a direct mapping from an input point cloud to DCCVT's native mesh representation:

```text
points: (P, 3) -> sites: (N, 3), sites_sdf: (N,) -> DCCVT extraction -> mesh
```

For v0, `N = 32^3 = 32768`. The predicted sites and SDF values are passed to the existing DCCVT extraction code, so the neural model only learns the generator representation.

## Files

- `dccvt/neural/dataset.py`: loads point clouds and paired DCCVT `.npz` labels.
- `dccvt/neural/models.py`: PointNet-style encoder and template-grid decoder.
- `dccvt/neural/train.py`: supervised training loop and checkpoint writer.
- `dccvt/neural/infer.py`: checkpoint inference and DCCVT mesh export bridge.
- `scripts/generate_neural_labels.py`: runs DCCVT to create supervised labels.
- `scripts/train_dccvt_neural.py`: command-line training entrypoint.
- `scripts/infer_dccvt_neural.py`: command-line inference and mesh export entrypoint.

## Data Flow

### 1. Generate Labels

The model is supervised from DCCVT output bundles. Each label bundle must contain:

```text
sites      shape (32768, 3)
sites_sdf  shape (32768,)
```

Generate labels with fixed v0 settings:

```bash
python scripts/generate_neural_labels.py \
  --mesh-ids 313444,441708,64764 \
  --output-root outputs/neural_labels/n32 \
  --num-centroids 32 \
  --num-iterations 1000
```

The script writes one directory per mesh id:

```text
outputs/neural_labels/n32/<mesh_id>/
  DCCVT_0_final_projDCCVT_cvt100_sdfsmooth100.npz
  target.ply
```

`target.ply` is the sampled point cloud used by DCCVT for that run. The neural dataset prefers it when available so the input points match the optimization target.

By default, training and inference sample `9600` input points per shape, matching the bundled DCCVT `target.ply` point-cloud size. This is independent from the number of predicted DCCVT sites.

### 2. Train

Train the PointNet prototype:

```bash
python scripts/train_dccvt_neural.py \
  --label-root outputs/neural_labels/n32 \
  --mesh-root mesh/thingi32 \
  --output-dir outputs/neural_runs/pointnet_n32 \
  --epochs 50 \
  --batch-size 1
```

Outputs:

```text
outputs/neural_runs/pointnet_n32/
  best.pt
  latest.pt
```

The checkpoint stores the model weights, optimizer state, model config, training config, and mesh ids used for training.

For a controlled site-prediction experiment, train only the site head:

```bash
python scripts/train_dccvt_neural.py \
  --label-root outputs/neural_labels/n32 \
  --mesh-ids 72960 \
  --output-dir outputs/neural_runs/overfit_72960_sites \
  --epochs 500 \
  --batch-size 1 \
  --lr 3e-4 \
  --target sites
```

`--target sites` keeps the same model output shape, but the loss only optimizes predicted site positions and the offset regularizer. It intentionally skips SDF regression and sign loss.

### 3. Infer and Extract

Run prediction and DCCVT extraction:

```bash
python scripts/infer_dccvt_neural.py \
  --checkpoint outputs/neural_runs/pointnet_n32/best.pt \
  --point-cloud mesh/thingi32/313444.ply \
  --output-dir outputs/neural_pred/313444
```

The inference script predicts `sites` and `sites_sdf`, then calls the existing `dccvt.mesh_ops.extract_mesh` path. This writes the usual DCCVT mesh artifacts:

```text
DCCVT_0_pred_intDCCVT_cvt0_sdfsmooth0.obj
DCCVT_0_pred_intDCCVT_cvt0_sdfsmooth0.npz
DCCVT_0_pred_projDCCVT_cvt0_sdfsmooth0.obj
DCCVT_0_pred_projDCCVT_cvt0_sdfsmooth0.npz
target.ply
```

For sites-only checkpoints, use HotSpot to evaluate SDF values at the predicted sites:

```bash
python scripts/infer_dccvt_neural.py \
  --checkpoint outputs/neural_runs/overfit_72960_sites/best.pt \
  --point-cloud outputs/neural_labels/n32/72960/target.ply \
  --output-dir outputs/neural_pred/overfit_72960_sites_hotspot \
  --sdf-source hotspot \
  --mesh mesh/thingi32/72960.ply \
  --hotspot hotspots_model/thingi32/72960.pth
```

In this mode, the neural model supplies only the site positions used for extraction. The SDF values come from the same HotSpot model family used by the original DCCVT pipeline.

## Architecture

The first model is `PointNetDCCVT`.

PointNet is a simple point-cloud encoder. It treats the input as an unordered set by applying the same per-point network to every point, then pooling all point features into a global feature vector.

The current encoder:

```text
(B, P, 3)
  -> shared 1x1 Conv/MLP over points
  -> max pool + mean pool over P
  -> global shape feature
```

The decoder starts from a deterministic `32^3` template grid and predicts:

```text
site offset: (B, 32768, 3)
site SDF:    (B, 32768)
```

Final sites are:

```text
sites = template_sites + offset_scale * tanh(predicted_offsets)
```

This keeps site ordering stable and makes direct regression to DCCVT `.npz` labels possible.

## Training Loss

The default v0 training target is `sites_sdf`, with loss:

```text
SmoothL1(pred_sites, target_sites)
+ SmoothL1(pred_sites_sdf, target_sites_sdf)
+ sign_loss_weight * BalancedBCE(-pred_sites_sdf, target_sites_sdf < 0)
+ offset_reg_weight * mean(offsets^2)
```

`SmoothL1Loss` is used instead of plain MSE because DCCVT optimization labels can contain occasional outlier sites or SDF values. Smooth L1 behaves like L2 near zero and like L1 for larger errors.

The balanced sign loss is important because only a small fraction of sites are inside the surface. Without it, the model can minimize SDF regression while predicting all-positive SDF values, which leaves DCCVT with no zero crossing and therefore no mesh.

The offset regularizer discourages large early displacements from the template grid.

The alternate `--target sites` mode uses:

```text
SmoothL1(pred_sites, target_sites)
+ offset_reg_weight * mean(offsets^2)
```

This mode is meant to isolate site prediction quality from SDF prediction quality. If predicted sites plus HotSpot SDF still produce a smooth or low-detail surface, the site predictor is the bottleneck. If predicted sites plus HotSpot SDF extract a good mesh, direct SDF regression was the main bottleneck in the earlier prototype. If target DCCVT sites plus HotSpot SDF produce detail but predicted sites do not, the architecture or site loss needs more local geometric supervision.

## Technical Decisions

### GPU-capable, CPU-testable

The neural code is not CPU-only. Training uses CUDA automatically when available:

```text
--device auto
```

resolves to `cuda:0` if PyTorch can see a GPU. The code is also CPU-testable so model, dataset, and checkpoint logic can be smoke-tested without initializing DCCVT's CUDA Delaunay stack.

DCCVT mesh extraction still requires the existing CUDA/gDel3D runtime.

### Lazy package imports

`dccvt/__init__.py` lazy-loads the existing public runner API. This prevents simple neural imports such as:

```python
from dccvt.neural.models import PointNetDCCVT
```

from immediately importing the DCCVT runner and selecting a CUDA device. The DCCVT runtime is imported only when running optimization or extraction.

### Fixed site count first

The prototype uses `num_centroids=32` and `upsampling=0`, giving exactly `32768` sites per shape.

Adaptive upsampling is intentionally out of scope for v0 because it produces variable site counts. Variable-size labels would require padding, masks, set matching, or a different decoder.

`32^3` labels are eight times larger than the original `16^3` prototype labels. The default neural training batch size is therefore `1`; increase it only after checking GPU memory.

### Direct DCCVT supervision

The first supervision source is DCCVT's own final `.npz` output. This avoids training through Delaunay/Voronoi topology changes and gives a direct test of whether a network can predict a useful DCCVT generator field.

### Template-grid decoder

The decoder predicts offsets from the same kind of structured grid used by the current DCCVT initialization. This avoids permutation ambiguity. A free unordered set decoder would need a permutation-invariant loss such as Chamfer matching between predicted and target sites.

## Current Limitations

- The model is a baseline and has no local neighborhood reasoning beyond global PointNet features.
- The loss supervises labels directly and does not yet include Chamfer, F1, watertightness, or mesh quality terms.
- The output site count is fixed to `num_centroids^3`.
- The inference mesh quality depends on whether predicted SDF signs produce enough useful zero crossings.
- Full mesh export still depends on CUDA/gDel3D through the existing DCCVT extraction path.

## Suggested Next Steps

1. Generate labels for a small stable set, for example `313444`, `441708`, and `64764`.
2. Overfit one mesh and confirm the predicted `.npz` gets close to the DCCVT label.
3. Export a predicted mesh and compare it to the DCCVT optimized mesh.
4. Add metric reporting against ground-truth meshes using the existing metrics utilities.
5. Scale to more mesh ids, then test larger or adaptive representations.
6. Add local features or geometry losses once the direct supervised path is reliable.
