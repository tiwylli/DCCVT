# Hybrid PoNQ-DCCVT Direct Extractor v1

## Purpose

This guide documents the hybrid PoNQ-DCCVT direct extractor implemented under
`dccvt/neural/`. It is meant for future implementers who need to understand the
research idea, the code path, the reproducible workflow, and the current result
quality.

The core idea is to keep PoNQ's useful dense-grid convolutional bias, but change
the prediction target to the field that DCCVT actually optimizes: one Voronoi
site and one SDF value for each canonical `32^3` DCCVT site. Inference should
produce the full DCCVT field directly, then call mesh extraction. It should not
run a per-shape site/SDF optimization loop.

The main implementation is separate from the older active-cell neural baseline.
The older `DCCVTPoNQNet` predicts per-cell sites plus activity scores and samples
SDF values from HotSpot downstream. The hybrid direct model predicts all
`32768` sites and all `32768` SDF values directly.

Code references:

- Hybrid config and channel names: [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:23), [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:26)
- Older active-cell baseline: [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:119)
- Hybrid direct model: [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:221)

## Method Overview

The pipeline is:

```text
target mesh / point cloud
  + HotSpot SDF cache on a 33x33x33 vertex grid
  -> four-channel hybrid input grid
  -> PoNQ-style 3D CNN
  -> full 32^3 site field and SDF field
  -> DCCVT mesh extraction
  -> intDCCVT OBJ used as the default evaluation mesh
```

The input remains in the existing normalized `[-1, 1]^3` coordinate convention.
The HotSpot SDF grid has side length `33`, while the DCCVT output has side
length `32` because the first convolution maps vertex-grid features to site-grid
features.

The four input channels are:

1. `hotspot_sdf`: cached HotSpot SDF values on the `33^3` grid.
2. `abs_hotspot_sdf`: absolute SDF magnitude.
3. `point_udf`: nearest distance from each grid vertex to the target point
   cloud, clipped and normalized by grid cell size.
4. `point_confidence`: Gaussian confidence from the UDF.

The point-cloud channels inject zero-level evidence that HotSpot alone may not
encode reliably for DCCVT extraction. The UDF and confidence implementation is
in [grid.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/grid.py:129) and [grid.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/grid.py:149).

## Architecture

### Architecture Theory

The architecture treats DCCVT reconstruction as dense field prediction rather
than sparse set prediction. DCCVT's optimizer maintains a structured state:
one site and one SDF value for every canonical grid position. The hybrid network
therefore predicts corrections to that structured state instead of inventing an
unordered point set from scratch.

This design has three useful inductive biases:

- **Locality**: nearby SDF/grid vertices should mostly influence nearby DCCVT
  sites. A 3D CNN encodes this assumption directly.
- **Translation sharing**: the same local geometric patterns, such as a surface
  crossing a grid neighborhood, should be handled similarly across the volume.
- **Residual prediction**: HotSpot and canonical DCCVT initialization already
  provide a reasonable coarse field. Predicting bounded corrections is easier
  and less unstable than predicting unconstrained absolute sites and SDF values.

The model is intentionally not a PointNet-style point-cloud encoder. The target
field is dense and grid-aligned, so the network keeps the grid structure all the
way through the encoder and heads. The point cloud enters as UDF/confidence
channels on the same grid, not as an unordered latent vector. This makes the
point evidence spatially comparable with HotSpot SDF values at every vertex.

### Canonical DCCVT Sites

The output sites are not PoNQ cell centers. They are initialized from the
canonical DCCVT grid:

```text
torch.linspace(-1, 1, 32) along x, y, z
```

For `grid_n = 33`, this creates `32^3 = 32768` base sites spanning the full
normalized domain. The helper is [grid.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/grid.py:66).

This distinction matters because the older baseline uses cell lower corners and
in-cell offsets, while the hybrid direct extractor predicts residuals around the
DCCVT canonical grid.

### Encoder

The encoder follows the PoNQ dense-grid pattern:

- input shape: `(B, 4, 33, 33, 33)`
- first layer: `Conv3d(input_channels, feature_dim, kernel_size=2)`
- output feature grid: `(B, feature_dim, 32, 32, 32)`
- additional `3x3x3` convolution blocks preserve the `32^3` resolution

The `kernel_size=2` first convolution is the key vertex-to-site-grid step. It
turns dense SDF vertex features into features aligned with the `32^3` DCCVT
field. The hybrid encoder and heads are defined in [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:221) and [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:234).

The encoder exists to build one learned feature vector per DCCVT site. The
`33^3` input grid stores values at vertices, while the `32^3` DCCVT field has
one output per site-grid location. A `2x2x2` convolution is a natural bridge
between these spaces because each output feature can see the eight neighboring
input vertices around a local grid cell. The later `3x3x3` layers let each site
feature exchange context with adjacent site features, which is important when
the correct Voronoi seed displacement depends on nearby surface orientation or
neighboring cells.

The encoder does not downsample. That is deliberate: DCCVT needs a prediction
for every site. Pooling or strided downsampling would require a later upsampling
decoder and could blur the exact grid-to-site correspondence.

The default JSON config is `configs/neural_hybrid_direct_v1.json`:

```json
{
  "grid_n": 33,
  "input_channels": 4,
  "feature_dim": 128,
  "encoder_layers": 5,
  "decoder_layers": 3,
  "site_delta_scale": 0.3,
  "sdf_residual_scale": 0.5,
  "point_udf_clip": 4.0,
  "point_confidence_sigma_scale": 1.5,
  "channel_names": [
    "hotspot_sdf",
    "abs_hotspot_sdf",
    "point_udf",
    "point_confidence"
  ]
}
```

`channel_names` is now the source of truth for the hybrid input layout.
`input_channels` must match its length; if omitted in a future config, the
typed config derives it from `channel_names`. The only valid names are
`hotspot_sdf`, `abs_hotspot_sdf`, `point_udf`, and `point_confidence`.
`hotspot_sdf` must stay first so checkpoint inference and direct
`model(input_grid)` calls can recover the dense SDF channel consistently.

The checked-in channel ablation configs are:

| Run name | Config | Channels |
| --- | --- | --- |
| `hotspot_sdf` | `configs/neural_hybrid_direct_ablation_hotspot_sdf.json` | `hotspot_sdf` |
| `hotspot_point_udf` | `configs/neural_hybrid_direct_ablation_hotspot_point_udf.json` | `hotspot_sdf`, `point_udf` |
| `hotspot_point_udf_abs` | `configs/neural_hybrid_direct_ablation_hotspot_point_udf_abs.json` | `hotspot_sdf`, `point_udf`, `abs_hotspot_sdf` |
| `hotspot_point_udf_confidence` | `configs/neural_hybrid_direct_ablation_hotspot_point_udf_confidence.json` | `hotspot_sdf`, `point_udf`, `point_confidence` |
| `full` | `configs/neural_hybrid_direct_ablation_full.json` | `hotspot_sdf`, `point_udf`, `point_confidence`, `abs_hotspot_sdf` |

The default `configs/neural_hybrid_direct_v1.json` is unchanged for backward
compatibility. The `full` ablation intentionally uses the user-requested
channel order, and each checkpoint records its own `channel_names`.

### Decoder Blocks

The decoder heads use small residual `1x1x1` blocks before the final output
projection. The shared feature grid already contains spatial context from the
encoder, so these blocks act as per-site nonlinear refiners rather than spatial
upsamplers. They let each head transform the common geometric representation
into the variables it needs without changing the `32^3` layout.

The residual form helps the head learn small corrections without forcing every
layer to relearn an identity mapping. In practical terms, it keeps the decoder
simple: spatial mixing happens in the encoder, and per-site variable conversion
happens in the heads. The reusable decoder block is [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:80), and the per-cell decoder wrapper is [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:94).

### Output Heads

The model has two dense heads:

- `site_delta_head`: predicts a 3D offset per canonical site.
- `sdf_residual_head`: predicts one SDF residual per site.

The heads are separated because site placement and SDF correction are related
but not identical tasks. Site placement is geometric: it moves Voronoi seeds so
the induced cells and clipped surface better match the target. SDF correction is
topological and sign-sensitive: it decides where the implicit zero crossing
falls once those sites are used by DCCVT extraction. Sharing the encoder lets
both tasks use the same local evidence, while separate heads avoid forcing a
single final projection to represent quantities with different dimensions,
scales, and losses.

The final site field is:

```text
sites = canonical_sites + 0.30 * tanh(raw_site_delta)
```

The final SDF field is:

```text
sites_sdf = trilinear_hotspot_sdf(sdf_grid, sites)
            + 0.50 * tanh(raw_sdf_residual)
```

The forward pass and residual composition are in [models.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/models.py:280). Trilinear SDF sampling is implemented in [grid.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/grid.py:230).

This design keeps the prediction anchored to HotSpot and the canonical DCCVT
initialization, while still allowing the network to correct both geometry and
SDF values.

## Implementation Walkthrough

### Dataset And Labels

`HybridDirectDataset` pairs each HotSpot SDF cache with an optimized DCCVT label
file. Cache resolution and split-file handling are shared with the older neural
pipeline. Label file resolution is in [dataset.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/dataset.py:107), and the hybrid dataset is in [dataset.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/dataset.py:128).

For each mesh, the dataset loads:

- `sdf_grid` from `outputs/neural_hotspot_sdf/thingi32_g33/<mesh_id>.npz`
- `target_points` from the same cache
- `sites` and `sites_sdf` labels from `outputs/neural_labels/n32/<mesh_id>/`
- hybrid input channels selected by the model config

The current full supervised run used labels with:

```text
DCCVT_0_final_projDCCVT_cvt100_sdfsmooth100.npz
```

That is an important caveat: the supervised labels are `projDCCVT`, but the
default evaluation mesh is now `intDCCVT`.

### Supervised Loss

Stage A uses direct supervision against optimized DCCVT labels:

- Huber loss on site positions.
- Weighted Huber loss on site SDF values, with higher weight near zero SDF.
- Class-balanced sign loss for inside/outside consistency.
- Residual regularization so SDF corrections stay near HotSpot unless labels
  demand otherwise.

The implementation is [losses.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/losses.py:195). The sign loss is especially important because negative SDF sites are sparse in the labels.

For one batch, the supervised objective is:

```text
L = w_site * L_site
  + w_sdf * L_sdf
  + w_sign * L_sign
  + w_residual * L_residual
```

The default weights are set by the training CLI:

```text
w_site = 1.0
w_sdf = 1.0
w_sign = 0.1
w_residual = 0.01
sdf_near_weight = 4.0
sdf_near_tau = 0.1
sign_temperature = 0.05
```

`L_site` is the Smooth L1, or Huber, loss between predicted sites and label
sites:

```text
L_site = mean_i SmoothL1(x_i, x_i*)
```

Here `x_i` is the predicted DCCVT site and `x_i*` is the optimized DCCVT label
site. This term teaches the site head to reproduce the spatial layout of the
DCCVT optimizer. It is robust to occasional large label differences because
Smooth L1 is quadratic near zero error and linear for larger errors.

`L_sdf` is a near-surface-weighted Smooth L1 loss on the scalar SDF value
attached to each predicted site:

```text
L_sdf = mean_i a_i * SmoothL1(phi_i, phi_i*)
a_i = 1 + sdf_near_weight * exp(-abs(phi_i*) / sdf_near_tau)
```

Here `phi_i` is the final predicted site SDF:

```text
phi_i = HotSpotSDF(x_i) + sdf_residual_i
```

and `phi_i*` is the label SDF. The multiplier `a_i` is largest when the label
SDF is close to zero, so this term spends more capacity on sites near the
surface. This matters because DCCVT extraction depends most strongly on where
the signed field crosses zero, not on far-away SDF magnitudes.

`L_sign` is a binary inside/outside classification loss derived from the
predicted SDF value:

```text
inside_i* = 1 if phi_i* < 0 else 0
logit_i = -phi_i / sign_temperature
L_sign = BCEWithLogits(logit_i, inside_i*)
```

The negative sign in `logit_i` makes negative predicted SDF values correspond to
inside probability. The implementation also uses a positive-class weight:

```text
pos_weight = outside_label_count / inside_label_count
```

In this code, the positive class is `inside_i* = 1`. This balancing is important
because inside, negative-SDF sites are sparse; without it, the model can get a
low average loss while predicting too many outside values. The temperature
controls how sharply SDF magnitude maps to inside/outside confidence. With the
default `0.05`, sign errors close to zero still receive a meaningful gradient.

`L_residual` penalizes the SDF correction itself:

```text
L_residual = mean_i sdf_residual_i^2
```

This keeps the model anchored to HotSpot. The residual head can override
HotSpot when labels demand it, but this term discourages unnecessary correction
of an already reasonable SDF sample. It does not penalize site movement; it only
acts on the additive SDF residual.

The logged `site`, `sdf`, `sign`, and `residual` values are the unweighted loss
components. The logged `loss` is the weighted sum used for backpropagation.
`sign_accuracy` reports whether `phi_i` and `phi_i*` have matching signs, while
`negative_fraction` reports the fraction of label sites with `phi_i* < 0`.

The final supervised checkpoint log ended at epoch `299` with:

```text
loss=0.004954
site=0.000260
sdf=0.003008
sign=0.015328
residual=0.015311
sign_accuracy=0.998955
negative_fraction=0.015535
```

High sign accuracy is not enough by itself. The extracted-mesh metrics below
show that matching site/SDF labels pointwise does not yet produce competitive
mesh quality.

### Mesh-Loss Fine-Tuning Hook

Stage B exists in code but has not been evaluated for this experiment. It uses
the full predicted site/SDF field, computes DCCVT clipped-mesh geometry, and
adds Chamfer, CVT, and SDF smoothness losses. The hook is [losses.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/losses.py:247), and training enables it with `--stage mesh` in [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:212).

The mesh-stage objective is:

```text
L_mesh = w_mesh_chamfer * L_chamfer
       + w_mesh_cvt * L_cvt
       + w_mesh_sdfsmooth * L_sdfsmooth
```

with CLI defaults:

```text
w_mesh_chamfer = 1000.0
w_mesh_cvt = 100.0
w_mesh_sdfsmooth = 100.0
```

`L_chamfer` compares the DCCVT clipped surface points produced by the current
predicted `(sites, sites_sdf)` field against the target point cloud:

```text
L_chamfer = Chamfer(projected_clipped_surface_points, target_points)
```

This is the most direct surface-quality term. Unlike `L_site` and `L_sdf`, it
does not ask the network to match a saved DCCVT label pointwise; it asks the
extracted geometry to lie near the target surface.

`L_cvt` is the DCCVT centroidal Voronoi term computed from the clipped vertices.
It encourages sites to remain well placed relative to the clipped cells they
induce. This keeps the DCCVT representation geometrically regular instead of
letting site positions move only to reduce Chamfer distance.

`L_sdfsmooth` is the same kind of discrete SDF regularization used by the DCCVT
optimizer. In the hybrid mesh hook it is:

```text
L_sdfsmooth = discrete_tet_volume_eikonal_loss / 10
            + tet_sdf_motion_mean_curvature_loss
```

The eikonal term encourages the tetrahedral SDF gradient to behave like a signed
distance field, while the curvature term discourages noisy SDF motion across the
Delaunay tetrahedra. This loss depends on both the predicted site positions and
the predicted site SDF values.

The mesh loss skips a batch item if there are fewer than five sites, if all
predicted SDF values have the same sign, if clipped extraction returns no
surface points, or if the geometry computation raises an exception. The reported
`mesh_used_shapes` and `mesh_skipped_shapes` counters should therefore be
checked whenever Stage B is used.

This is the most direct next experiment because the current failure mode is
mesh quality, not just supervised label fit.

### Training Entry Point And Metadata

Training is implemented in `dccvt/neural/hybrid_train.py` and exposed through
`scripts/train_dccvt_hybrid_direct.py`.

Important code references:

- Reproducibility seed: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:32)
- Resolved config writer: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:49)
- Checkpoint payload: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:59)
- CLI arguments: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:81)
- Dataset construction: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:134)
- Main training loop: [hybrid_train.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_train.py:154)

Checkpoints and `resolved_config.json` save the resolved model config, seed,
channel list, command arguments, and training stats. Training passes
`channel_names` into `HybridDirectDataset`; inference reads `channel_names` from
the checkpoint and rebuilds the same input layout.

### Inference, Prediction Files, And Extraction

Inference is implemented in `dccvt/neural/hybrid_infer.py` and exposed through
`scripts/infer_dccvt_hybrid_direct.py`.

Important code references:

- Inference API: [hybrid_infer.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_infer.py:38)
- Hybrid input channel reconstruction: [hybrid_infer.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_infer.py:65)
- Prediction `.npz` writer: [hybrid_infer.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_infer.py:89)
- Extraction guard and call: [hybrid_infer.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_infer.py:111)
- Inference CLI: [hybrid_infer.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/dccvt/neural/hybrid_infer.py:143)

The prediction `.npz` stores:

- `sites`
- `sites_sdf`
- `site_delta`
- `sdf_residual`
- `hotspot_sdf_at_sites`
- `canonical_sites`
- `input_grid`
- `sdf_grid`
- `target_points`
- `resolved_config`
- `channel_names`
- `diagnostics`
- `command_args`
- `seed`
- `mesh_id`

Extraction is skipped unless the predicted field has at least five sites and
both positive and negative SDF values. A successful extraction writes both
`intDCCVT` and `projDCCVT` OBJ files through the existing DCCVT mesh extraction
path.

### Unit Tests

The current tests cover canonical grid ordering, selected input-channel layouts,
point UDF/confidence channels, reduced-channel forward passes, residual SDF
composition, checkpoint reload, and ablation runner command construction. See
[test_neural_hybrid_direct.py](/export/livia/home/vision/Wcharawi/dev/DCCVT/tests/test_neural_hybrid_direct.py:10).

## Reproduction Workflow

All commands assume the repository root:

```bash
cd /export/livia/home/vision/Wcharawi/dev/DCCVT
```

### Full Supervised Training

The completed full supervised run used the HotSpot grid-33 cache split and
saved to `outputs/neural_dccvt/hybrid_direct_v1/full_supervised`.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 PoNQ-main/.venv/bin/python \
  scripts/train_dccvt_hybrid_direct.py \
  --config configs/neural_hybrid_direct_v1.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_direct_v1/full_supervised \
  --stage supervised \
  --epochs 298 \
  --batch-size 1 \
  --lr 6.4e-5 \
  --device cuda \
  --seed 69 \
  --save-every 25 \
  --resume outputs/neural_dccvt/hybrid_direct_v1/full_supervised/latest.pt \
  --resume-optimizer
```

The corresponding resolved config is:

```text
outputs/neural_dccvt/hybrid_direct_v1/full_supervised/resolved_config.json
```

### Channel Ablation Runner

The generic ablation runner trains all five channel combinations. By default it
runs sequentially. With `--parallel`, it detects free GPUs with `nvidia-smi`,
assigns one ablation per GPU, writes each run log to `<output-root>/<run-name>/train.log`,
and waits/polls when all selected GPUs are busy. It owns `--config` and
`--checkpoint-dir`; all other arguments are forwarded to
`scripts/train_dccvt_hybrid_direct.py`.

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

The per-run checkpoint directories are:

```text
outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444/hotspot_sdf/
outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444/hotspot_point_udf/
outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444/hotspot_point_udf_abs/
outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444/hotspot_point_udf_confidence/
outputs/neural_dccvt/hybrid_direct_ablation_smoke_313444/full/
```

Use `--dry-run` to print the five training commands without running them. The
runner refuses to reuse non-empty per-run checkpoint directories unless
`--allow-existing` is passed.

The full sequential command is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PoNQ-main/.venv/bin/python \
  scripts/run_hybrid_direct_channel_ablation.py \
  --output-root outputs/neural_dccvt/hybrid_direct_ablation \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --label-root outputs/neural_labels/n32 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --stage supervised \
  --epochs 300 \
  --batch-size 1 \
  --lr 6.4e-5 \
  --device cuda \
  --seed 69 \
  --save-every 25
```

The full parallel command is:

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

### No-Extract Inference

Run no-extract inference first to validate that the network predicts the full
field and both SDF signs:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /tmp/dccvt-venv/bin/python \
  scripts/infer_dccvt_hybrid_direct.py \
  --checkpoint outputs/neural_dccvt/hybrid_direct_v1/full_supervised/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/313444.npz \
  --output-dir outputs/neural_dccvt/hybrid_direct_v1/infer_full/313444 \
  --device cuda \
  --no-extract \
  --seed 69
```

Health-check condition:

```text
site_count == 32768
positive_sdf_count > 0
negative_sdf_count > 0
```

### Full Extraction

The extraction run used `/tmp/dccvt-venv/bin/python` because that environment
contains the CUDA/gDel3D dependencies needed by DCCVT extraction.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /tmp/dccvt-venv/bin/python \
  scripts/infer_dccvt_hybrid_direct.py \
  --checkpoint outputs/neural_dccvt/hybrid_direct_v1/full_supervised/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/313444.npz \
  --output-dir outputs/neural_dccvt/hybrid_direct_v1/extract_full/313444 \
  --device cuda \
  --seed 69
```

For the full split, this produced 31 directories under:

```text
outputs/neural_dccvt/hybrid_direct_v1/extract_full/<mesh_id>/
```

The default evaluation OBJ is:

```text
DCCVT_0_hybrid_direct_intDCCVT_cvt100_sdfsmooth100.obj
```

The `projDCCVT` OBJ is retained only as a diagnostic baseline:

```text
DCCVT_0_hybrid_direct_projDCCVT_cvt100_sdfsmooth100.obj
```

### Flat Evaluation View

`PoNQ-main/src/eval/eval_HOTSPOT.py` expects a flat directory containing
`<mesh_id>.obj`. For default `intDCCVT` evaluation, create symlinks:

```bash
PRED_DIR="PoNQ-main/out_hybrid_direct/HybridDirect_full_supervised_intDCCVT"
SPLIT="PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt"
SRC_ROOT="/export/livia/home/vision/Wcharawi/dev/DCCVT/outputs/neural_dccvt/hybrid_direct_v1/extract_full"

mkdir -p "$PRED_DIR"
while read -r id; do
  [ -z "$id" ] && continue
  ln -sfn \
    "$SRC_ROOT/$id/DCCVT_0_hybrid_direct_intDCCVT_cvt100_sdfsmooth100.obj" \
    "$PRED_DIR/$id.obj"
done < "$SPLIT"
```

Verify the flat view:

```bash
find -L PoNQ-main/out_hybrid_direct/HybridDirect_full_supervised_intDCCVT \
  -maxdepth 1 -type f -name '*.obj' | wc -l
```

Expected count: `31`.

### Evaluation

Run the existing HotSpot diagnostic evaluator:

```bash
cd /export/livia/home/vision/Wcharawi/dev/DCCVT/PoNQ-main

.venv/bin/python src/eval/eval_HOTSPOT.py \
  out_hybrid_direct/HybridDirect_full_supervised_intDCCVT \
  -gt_dir /export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32 \
  -all_models src/eval/hotspot_thingi32_g33_ids.txt \
  -pred_suffix .obj \
  -mode all \
  -sample_num 100000
```

Default output files:

```text
PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_ponq_thingi.npy
PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_raw.npy
PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_bbox_aligned.npy
PoNQ-main/src/eval/results/results_HybridDirect_full_supervised_intDCCVT_hotspot_summary.csv
```

The retained `projDCCVT` diagnostic baseline uses the same evaluator with:

```text
PoNQ-main/out_hybrid_direct/HybridDirect_full_supervised_projDCCVT
```

## Result Analysis

The default metric case is `intDCCVT`.

| case | mode | CD x1e-5 | F1 | NC | ECD | EF1 |
|---|---|---:|---:|---:|---:|---:|
| hybrid `intDCCVT` | `ponq_thingi` | 465.425 | 0.027 | 0.638 | 2.269 | 0.000 |
| hybrid `intDCCVT` | `raw` | 1861.174 | 0.010 | 0.638 | 9.076 | 0.000 |
| hybrid `intDCCVT` | `bbox_aligned` | 77.289 | 0.095 | 0.707 | 9.079 | 0.000 |
| hybrid `projDCCVT` | `bbox_aligned` | 91.150 | 0.095 | 0.622 | 9.078 | 0.000 |
| HotSpot PoNQ baseline | `bbox_aligned` | 14.024 | 0.261 | 0.896 | 0.240 | 0.079 |

The main observations are:

- `intDCCVT` improves bbox-aligned CD and normal consistency over the
  `projDCCVT` diagnostic baseline.
- Both hybrid variants still trail the HotSpot PoNQ baseline on surface quality
  and edge quality.
- EF1 near zero indicates weak sharp-edge reconstruction.
- High supervised sign accuracy did not translate into high-quality extracted
  meshes.
- The current run trained on `projDCCVT` labels but uses `intDCCVT` as the
  default evaluation mesh. This label/evaluation mismatch should be considered
  when interpreting the numbers.

The `ponq_thingi`, `raw`, and `bbox_aligned` modes are diagnostic conventions
from `eval_HOTSPOT.py`. `bbox_aligned` should not be treated as the final strict
metric. It is useful here because it separates local mesh quality from global
scale and translation mismatch.

## Current Limitations

- Stage B mesh-loss fine-tuning is implemented but has not been run for this
  experiment.
- The current supervised labels are `projDCCVT`, while default evaluation uses
  `intDCCVT`.
- The point-cloud fusion is voxel UDF/confidence only. KNN point features are a
  reasonable next ablation if voxelization underfits.
- Edge reconstruction is currently poor according to EF1.
- The guide includes line-number references to the current implementation
  snapshot. Those references can drift when source files change.

## Suggested Next Experiments

1. Run Stage B mesh-loss fine-tuning from the full supervised checkpoint.
2. Generate or train against `intDCCVT` labels if default evaluation remains
   `intDCCVT`.
3. Inspect per-shape failures to separate missing geometry, sign errors, and
   extraction artifacts.
4. Add a KNN or local point-feature branch if the voxel UDF/confidence channels
   cannot encode enough surface detail.
5. Track strict and diagnostic metrics together so scale mismatch does not hide
   local reconstruction quality.
