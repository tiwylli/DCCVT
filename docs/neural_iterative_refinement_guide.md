# Neural Iterative Refinement Guide

## Purpose

This guide documents the final HotSpot near-surface iterative refinement model
implemented in `dccvt/neural/iter_refine.py`. It is written for future
implementers who need to understand the research idea, numerical assumptions,
gradient path, code structure, and reproducible workflow well enough to modify
the method safely.

The method does not predict a complete fixed `32^3` DCCVT field. It first
constructs a deterministic, shape-dependent field from the cached HotSpot SDF,
then learns to spawn a bounded number of additional sites around procedurally
selected DCCVT parents. Training uses the differentiable clipped-mesh losses
already present in DCCVT. There are no DCCVT upsampling labels.

The implementation keeps three site sets distinct:

1. **Background sites**: 512 jittered support sites spanning the domain.
2. **Complete initialization**: background sites plus 3,236 near-surface sites,
   normally 3,748 sites total. This is exported under the historical name
   `base_sites`.
3. **Final active field**: the complete initialization plus all accepted learned
   children from every refinement round.

The historical v1 config uses four input channels, one round, at most 128
parents, and four child slots per parent. Its maximum final site count is
therefore

```text
512 background
+ 3236 near-surface sites
+ 128 parents * 4 child slots
= 4260 sites maximum
```

The active v2 comparison configs use only `hotspot_sdf` and `point_udf`, and
add an initialization-only baseline. Their budgets are 3,748 initial sites,
4,260 sites for one round with 128 parents, and 4,772 sites for either two
128-parent rounds or one 256-parent round. Spacing checks can reject children,
so these are maxima rather than guaranteed final counts.

The active v3 comparison keeps the same application-facing input point cloud
stored as cache `target_points`, but exposes more of it to the refinement
decoder. It also overwrites the earlier v3 background support with base16
initialization: `base_grid_n=17` gives `16^3 = 4096` background sites. Together
with the unchanged 3,236 near-surface sites, v3/v4 now start from 7,332 initial
sites and have an 8,356-site two-round maximum before spacing rejections. No
ground-truth mesh surface sampling is introduced.

Primary code references:

- [`dccvt/neural/iter_refine.py`](../dccvt/neural/iter_refine.py): config,
  initialization, parent selection, model, training, inference, and export.
- [`configs/neural_hybrid_iter_refine_v1.json`](../configs/neural_hybrid_iter_refine_v1.json):
  historical four-channel v1 defaults.
- `configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json` and
  `configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_*.json`: active
  two-channel baseline and comparison configs.
- `configs/neural_hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128.json`:
  two-channel global input plus local `65^3` UDF and KNN parent features.
- `configs/neural_hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128.json`:
  fair v3 ablation that replaces the dense feature path with Delaunay graph
  message passing.
- [`dccvt/neural/point_udf_sidecar.py`](../dccvt/neural/point_udf_sidecar.py):
  exact point-UDF sidecar generation and validation.
- [`tests/test_neural_iter_refine.py`](../tests/test_neural_iter_refine.py):
  executable behavioral specification.

## Method Overview

The end-to-end path is:

```text
HotSpot cache
  sdf_grid (33^3) + target_points
       |
       +--> deterministic HotSpot-only initialization on CPU
       |      4096 jittered background sites for v3/v4
       |    + 1618 projected surface anchors * 2 signed sites
       |    = 7332 initial sites for v3/v4
       |
       +--> two-channel input grid (2 x 33 x 33 x 33)
       +--> optional v3 local point evidence
       |      65^3 exact point-UDF sidecar + parent KNN stats
              |
              v
          dense 3D CNN encoder or v4 Delaunay GCNN
              |
              v
  current sites + current SDF values
              |
      Delaunay and procedural parent scoring
              |
      sample encoder features and optional local point features at parents
              |
      learned child offset and SDF residual decoder
              |
      reject collisions and append accepted children
              |
       repeat for configured rounds
              |
      differentiable clipped DCCVT mesh loss
              |
      final prediction NPZ and optional meshes
```

The design separates responsibilities deliberately:

- HotSpot supplies a signed field and the initial near-surface geometry.
- Procedural DCCVT logic decides where refinement is currently useful.
- The neural model decides where each selected parent should place its children
  and how their sampled HotSpot SDF values should be corrected.
- The mesh loss evaluates the final geometric consequence rather than matching
  children to precomputed refinement labels.

### v4 Delaunay GCNN Feature Path

The v4 config keeps the v3 inputs, initialization, local `65^3` UDF sidecar,
local KNN statistics, parent budget, and mesh loss. The only intended change is
the learned feature path: instead of sampling a dense `Conv3d` feature volume at
selected parents, it runs message passing on the current Delaunay graph.

Each refinement round builds graph nodes from the active DCCVT sites and graph
edges from the Delaunay tetrahedra already needed by procedural parent
selection. Each tetrahedron contributes six undirected edges, expanded to both
directions for message passing.

Node features include:

- global input channels sampled at the site;
- current site SDF;
- raw site xyz plus Fourier positional encoding with frequencies `1,2,4,8`;
- the local UDF sidecar value at the site when enabled;
- local KNN statistics from cache `target_points` when enabled.

Directed edge features are:

```text
dx, dy, dz, distance, dir_x, dir_y, dir_z, sdf_dst - sdf_src
```

Messages are built from source node features, destination node features, and
edge features. They are aggregated into destination nodes with `index_add`,
normalized by destination degree, and applied with an MLP residual update. No
PyTorch Geometric or extra graph dependency is required.

Unlike a full fixed-field predictor, the model spends most of its site budget
near the inferred surface. Unlike unconstrained point generation, it retains a
coarse background field, fixed child slots, bounded residuals, and explicit
spacing checks.

## Required Inputs

### HotSpot Cache

Training and inference consume one `.npz` cache per mesh. The default root is:

```text
outputs/neural_hotspot_sdf/thingi32_g33/<mesh_id>.npz
```

The iterative dataset directly requires:

| Key | Expected value | Use |
| --- | --- | --- |
| `sdf_grid` | Float array `(33, 33, 33)` by default | Initialization, first input channel, and child SDF sampling. |
| `target_points` | Float array `(N, 3)` | Point-UDF input channel and mesh-loss target. Historical four-channel v1 also used point confidence. |
| `grid_n` | Scalar integer | Must equal `hotspot_grid_n`. |
| `mesh_id` | Scalar string, optional | Logging and output naming; cache stem is the fallback. |

The standard precompute cache also contains `near_surface_mask`,
`gt_activity_mask`, source paths, and domain metadata. The iterative refinement
dataset does not read those masks.

All geometry is assumed to use the existing normalized `[-1, 1]^3` domain.
The code verifies grid size but does not independently validate cache coordinate
normalization.

### Target-Point Role

`target_points` have two roles:

1. They build the `point_udf` neural input channel. Historical four-channel
   v1 also built `point_confidence`.
2. They are the target set for clipped-mesh Chamfer loss.

They do **not** place the background sites, projected anchors, or initial signed
pairs. Near-surface initialization depends only on `sdf_grid` and the model
config. This distinction is important when reasoning about what information
the initializer uses.

When `--target-subsample K` is passed, v1/v2 behavior subsamples points before
both channel construction and loss evaluation. The subset can therefore affect
the input tensor as well as the Chamfer target. v3 still uses that subsampled
tensor for the coarse `33^3` channel and mesh loss, but its local KNN features
receive the full cache `target_points` tensor.

### v3 Local Feature Path

v3 keeps the global neural input exactly two-channel:

```text
hotspot_sdf + point_udf on the 33^3 cache grid
```

The additional signal is local to selected refinement parents. It is not a new
dense global encoder branch.

For every selected parent, the decoder receives:

| Feature group | Width | Source |
| --- | ---: | --- |
| Coarse encoder feature | `feature_dim` | `grid_sample()` from the `33^3` encoder feature grid. |
| Parent state | `4` | Parent XYZ plus parent SDF. |
| Local `65^3` point-UDF samples | `1 + slots_per_parent` | Sidecar UDF sampled at the parent and at the fixed child stencil offsets. |
| Local KNN statistics | `7` | Full cache `target_points`, not the `--target-subsample` subset. |

With the v3 config defaults this makes the decoder input width:

```text
128 encoder features
+ 4 parent state values
+ 5 local UDF samples
+ 7 KNN statistics
= 144 values per parent
```

The local point-UDF samples are raw nearest-input-point distances stored in the
sidecar. At runtime they are normalized by the `65^3` cell size and clipped by
`point_udf_clip`, matching the scale of the coarse `point_udf` channel.

The KNN statistics are:

- nearest-point distance divided by `local_knn_radius`;
- mean offset vector from parent to the K nearest input points, divided by
  `local_knn_radius`;
- mean KNN distance divided by `local_knn_radius`;
- number of input points within `local_knn_radius`, divided by `local_knn_k`;
- local covariance anisotropy from the KNN offsets.

This design preserves the refinement program: parent selection remains
procedural, children are still bounded child slots plus bounded SDF residuals,
and the differentiable clipped-mesh loss remains the training objective.

## Near-Surface Initialization

`build_hotspot_near_surface_initialization()` constructs the initial field on
CPU and returns its sites, SDF values, component arrays, validity state, and
diagnostics. `HybridIterRefineDataset` memoizes this result by dataset index, so
the same initialization is reused across epochs when `num_workers=0`.

### Background Support

`make_canonical_sites(base_grid_n)` creates `(base_grid_n - 1)^3` sites. With
the v3/v4 base16 configs, `base_grid_n=17` gives:

```text
(17 - 1)^3 = 16^3 = 4096 sites
```

Historical v1/v2 iterative-refinement configs still use `base_grid_n=9`, which
gives 512 background sites.

The initializer adds Gaussian noise with standard deviation
`background_jitter_scale=0.005`, using `bootstrap_seed=69`. Coordinates are
then clamped to `[-1 + 1e-4, 1 - 1e-4]`.

The background serves two purposes:

- it gives Delaunay triangulation support away from the surface;
- jitter breaks exact regular-lattice degeneracies.

Background SDF values are sampled from the cached grid by trilinear
interpolation.

### Sign-Changing Cells

For every cell in the `G^3` HotSpot vertex grid, the initializer gathers its
eight corner values. A cell is eligible only when

$$
\min_{c \in C} \phi(c) < 0
\quad\text{and}\quad
\max_{c \in C} \phi(c) > 0.
$$

Strict inequalities are used. A cell whose corners are all one sign is not a
bootstrap source even when some values are close to zero.

For `G=33`, the grid-cell size is

$$
h = \frac{2}{G-1} = \frac{2}{32} = 0.0625.
$$

### Candidate Sampling and Projection

The target number of surface anchors is half of `surface_pair_count`:

```text
3236 / 2 = 1618 anchors
```

For each configured candidate multiplier `m` in `(4, 8, 16)`, the initializer
requests `1618 * m` candidates and distributes an equal integer number of
uniform random samples across the sign-changing cells. Pass `j` uses seed
`bootstrap_seed + j`.

Each candidate is projected toward the local HotSpot zero set. Gradients are
estimated by central differences with step `h/4`:

$$
\frac{\partial \phi}{\partial x_k}(p)
\approx
\frac{\phi(p + \frac{h}{4}e_k) -
      \phi(p - \frac{h}{4}e_k)}{2\frac{h}{4}}.
$$

One Newton-style update is

$$
u(p) = \frac{\phi(p)\nabla\phi(p)}
             {\max(\|\nabla\phi(p)\|^2, 10^{-12})},
$$

followed by a step-length bound

$$
u \leftarrow u \min\left(1, \frac{h}{\max(\|u\|,10^{-12})}\right)
$$

and a cell-bound clamp

$$
p \leftarrow \operatorname{clip}_{\text{source cell}}(p-u).
$$

The default performs three projection steps. Keeping the point inside its
original sign-changing cell prevents an unstable gradient estimate from moving
the candidate into unrelated geometry.

### Signed Surface Pairs

After projection, the gradient defines the unit normal

$$
n(p) = \frac{\nabla\phi(p)}
             {\max(\|\nabla\phi(p)\|,10^{-12})}.
$$

The initializer proposes two sites:

$$
p^- = p - \delta n(p),
\qquad
p^+ = p + \delta n(p),
$$

where `surface_pair_offset=0.03125`, or half a default HotSpot grid cell.
HotSpot SDF values are resampled at both points. A pair is valid only when:

- the projected point, pair coordinates, gradients, and sampled SDF values are
  finite;
- the gradient norm is greater than `1e-8`;
- both points lie inside `[-1, 1]^3`;
- the two sampled SDF values have strictly opposite signs.

This gives each accepted anchor an explicit inside/outside bracket. The code
does not assume whether `p^-` or `p^+` is the negative endpoint; it verifies the
sampled signs directly.

### Coverage and Spacing

Valid anchors are ordered by deterministic farthest-point sampling:

1. choose the point farthest from the candidate centroid;
2. repeatedly choose the point maximizing distance to the selected set;
3. stop if the next anchor is closer than `bootstrap_min_distance`.

Pairs are accepted in that order through a spatial hash. Both endpoints must
remain at least `bootstrap_min_distance=0.005` from:

- every background site;
- every endpoint of previously accepted pairs;
- the other endpoint in the same pair.

If the current candidate multiplier does not yield 1,618 accepted anchors, the
initializer retries with the next multiplier. It never repeats a site to fill a
budget.

### Validity and Output

The initialization is valid when it contains at least
`min_surface_anchors=128`. Possible status reasons are:

- `ok`;
- `no_sign_changing_cells`;
- `insufficient_valid_surface_pairs`.

With the checked-in Thingi32 caches, v3/v4 normally reach the full base16
budget:

```text
4096 background sites + 1618 anchors * 2 = 7332 initial sites
```

The initialization algorithm can be summarized as:

```text
background = jitter(canonical_grid(base_grid_n), seed)
crossing_cells = cells whose corner SDF values contain both signs

for multiplier in [4, 8, 16]:
    candidates = sample_uniformly_in(crossing_cells,
                                     target_anchors * multiplier)
    projected = bounded_newton_projection(candidates, sdf_grid)
    pairs = projected +/- pair_offset * normalized_sdf_gradient
    valid_pairs = finite AND in_domain AND opposite_signs
    selected = farthest_point_and_spacing_filter(valid_pairs, background)
    if len(selected) == target_anchors:
        break

valid = len(selected) >= min_surface_anchors
initial_sites = concatenate(background, flatten(selected_pairs))
initial_sdf = trilinear_sample(sdf_grid, initial_sites)
```

## Neural Input and Encoder

The active comparison input has two channels, built by
`build_hybrid_input_channels_np()` in
[`dccvt/neural/grid.py`](../dccvt/neural/grid.py):

| Channel | Meaning |
| --- | --- |
| `hotspot_sdf` | Signed HotSpot values on the `33^3` grid. Must remain the first channel. |
| `point_udf` | Distance from each grid vertex to the nearest target point, normalized by grid-cell size and clipped. |

Default input shape:

```text
(B, C, G, G, G) = (1, 2, 33, 33, 33)
```

The first encoder layer is:

```text
Conv3d(2, 128, kernel_size=2, stride=1)
LeakyReLU(0.01)
```

The historical v1 config used four channels: `hotspot_sdf`,
`abs_hotspot_sdf`, `point_udf`, and `point_confidence`. Keep that config for
old checkpoint reproduction instead of resuming old checkpoints into a
two-channel config.

It converts the `33^3` vertex grid to a `32^3` feature grid:

```text
(1, 2, 33, 33, 33) -> (1, 128, 32, 32, 32)
```

Five default `3x3x3` convolutions with padding 1 preserve that shape. There is
no pooling or feature upsampling. The feature grid is computed once per forward
pass and reused in every refinement round.

Parent features are sampled at continuous parent coordinates using 3D
`grid_sample(..., mode="bilinear", align_corners=True)`. Coordinates are
reordered from repository `(x, y, z)` convention to grid-sample `(z, y, x)`
ordering before sampling.

## Procedural Parent Selection

Every round selects parents from the current active field, including children
from earlier rounds.

### Delaunay Neighborhood

`compute_delaunay_simplices()` triangulates the detached active sites with
`pygdel3d`. Each tetrahedron contributes its six undirected edges. Duplicate
edges are removed to produce the site-neighbor graph.

Selection exits safely with no parents when:

- the budget is zero;
- there are fewer than five active sites;
- the active SDF values do not contain both signs;
- Delaunay triangulation returns no tetrahedra;
- no Delaunay edge crosses zero.

### Zero-Crossing Candidates

An edge `(i,j)` crosses zero when

$$
\phi_i\phi_j \leq 0.
$$

Every endpoint of a crossing edge becomes a candidate parent. Candidate indices
are unique.

### Distance and Curvature Score

The selector estimates an SDF gradient at each site from the current Delaunay
tetrahedra. Let

$$
\hat n_i = \frac{g_i}{\|g_i\| + \epsilon}
$$

be the estimated unit gradient. The curvature-like score is

$$
c_i = \frac{1}{|N(i)|}
      \sum_{j \in N(i)}
      \left(0.8\|\hat n_i-\hat n_j\|^2 + 0.2\right).
$$

The local distance score is the shortest incident Delaunay-edge length:

$$
d_i = \min_{j \in N(i)} \|x_i-x_j\|.
$$

For zero-crossing candidates `Z`, the final score is median-normalized:

$$
q_i =
\frac{d_i}{\operatorname{median}_{k\in Z}(d_k)}
\frac{c_i}{\operatorname{median}_{k\in Z}(c_k)}.
$$

NaN and infinite scores are replaced with zero. The selector returns the
highest-scoring unique sites, capped by `max_parents_per_round=128`. It does not
repeat parents to force a fixed count.

Parent selection runs under `torch.no_grad()` on detached sites and SDF values.
It is a procedural routing decision, not a learned or differentiable activity
head.

## Learned Child Generation

### Decoder Input and Shape

For `P` selected parents, the decoder input concatenates:

- sampled encoder feature: `feature_dim=128` values;
- parent coordinate: 3 values;
- parent SDF: 1 value.

The default decoder input shape is therefore `(P, 132)`. Two linear layers with
LeakyReLU preserve dimension 132. The output layer predicts four values for
each child slot:

```text
(P, 132) -> (P, slots_per_parent * 4) -> (P, 4, 4)
```

The last dimension is three offset residuals plus one SDF residual.

For config version 2, the final decoder layer is initialized to zero. The first
forward pass therefore starts from the fixed child stencil and zero SDF
residual rather than random child corrections.

### Tetrahedral Stencil and Bounded Offset

The four default unit stencil directions are normalized versions of:

```text
( 1,  1,  1)
( 1, -1, -1)
(-1,  1, -1)
(-1, -1,  1)
```

For parent `i` and slot `k`, the child coordinate is

$$
x_{ik} = \operatorname{clip}_{[-1,1]^3}
\left(
x_i + \alpha v_k + \beta\tanh(o_{ik})
\right),
$$

where:

- `alpha = child_stencil_scale = 0.015625`;
- `beta = child_offset_scale = 0.03125`;
- `v_k` is a unit tetrahedral direction;
- `o_ik` is the learned three-vector.

The stencil prevents all zero-initialized slots from occupying the parent or
each other. The bounded learned term limits early geometric disruption.

### SDF Prediction

The child SDF is

$$
\phi_{ik} =
\phi_{\text{HotSpot}}(x_{ik})
+ \gamma\tanh(r_{ik}),
$$

with `gamma = sdf_residual_scale = 0.0625`. HotSpot interpolation is
differentiable with respect to the accepted child coordinate inside a grid
cell. The residual lets the network correct the cached field locally.

### Child Collision Filtering

Candidates closer than `spawn_min_distance=0.0025` to any current active site
are rejected. Remaining candidates are processed in decoder order and rejected
when they are too close to an earlier accepted child in the same round.

The keep mask is computed from detached coordinates under `torch.no_grad()`.
Accepted tensor entries retain their original autograd graph; rejected entries
do not enter the final field. The number of rejected children is exported per
round.

Accepted children and SDF values are concatenated to the active field. The next
round, when configured, recomputes Delaunay topology and parent scores using
this larger field.

## Differentiability and Gradient Path

The method is only partially differentiable. This is intentional.

```text
HotSpot initialization ---------------------------- fixed
       |
current sites -> [detach] -> Delaunay/top-k parents -> fixed routing
                                      |
encoder features --------------------> decoder
                                      |
                         child coordinates/SDF
                                      |
                     detached collision keep mask
                                      |
                       accepted child tensors
                                      |
                    clipped DCCVT mesh loss
                                      |
                      gradients to decoder
                         and encoder
```

No gradient flows through:

- HotSpot near-surface initialization;
- Delaunay simplex construction;
- zero-crossing and curvature parent selection;
- top-k parent indices;
- collision accept/reject decisions.

Gradients do flow through:

- encoder features sampled at selected parent coordinates;
- decoder offset and SDF-residual predictions;
- accepted child coordinates;
- HotSpot trilinear interpolation at accepted child coordinates;
- clipped-mesh geometry for the current fixed Delaunay topology;
- Chamfer, CVT, and SDF-smoothness terms.

The base initialization itself is not learned in v1. Only spawned children can
move through network output.

## Training Objective

Training calls `hybrid_direct_mesh_loss()` from
[`dccvt/neural/losses.py`](../dccvt/neural/losses.py). It computes a fresh
Delaunay triangulation of the final field, then constructs clipped DCCVT surface
points and Voronoi vertices.

The total loss is

$$
L = w_{ch}L_{chamfer} + w_{cvt}L_{cvt} + w_{sdf}L_{smooth}.
$$

### Chamfer

`L_chamfer` is symmetric squared point-set Chamfer distance between projected
clipped-mesh points `P` and cached target points `Q`:

$$
L_{chamfer} =
\frac{1}{|P|}\sum_{p\in P}\min_{q\in Q}\|p-q\|^2
+
\frac{1}{|Q|}\sum_{q\in Q}\min_{p\in P}\|q-p\|^2.
$$

PyTorch3D KNN is used when available; otherwise the implementation uses a
chunked `torch.cdist` fallback.

### CVT and SDF Smoothness

`L_cvt` uses the existing clipped Voronoi-vertex CVT loss. It is skipped
entirely when `w_mesh_cvt=0`.

The smoothness term is

$$
L_{smooth} = 0.1L_{eikonal} + L_{curvature},
$$

using the existing discrete tetrahedral volume eikonal and mean-curvature
motion losses. It is skipped entirely when `w_mesh_sdfsmooth=0`.

Checked-in CLI defaults are:

```text
w_mesh_chamfer  = 1000
w_mesh_cvt      = 100
w_mesh_sdfsmooth = 100
```

### Invalid Mesh-Loss Batches

A shape cannot contribute mesh loss when it has fewer than five sites, lacks
one SDF sign, produces no projected points, or raises a geometry exception.

Default behavior records a skipped shape. `--strict-mesh-loss` converts these
conditions into errors for debugging. When the returned loss has no gradient,
the training loop records `mesh_no_grad_batch=1` and does not call backward or
step the optimizer.

Initialization invalidity is handled earlier. The default records a structured
`initialization_skip_<reason>` statistic. `--strict-initialization` raises
instead.

## Runtime and Reproducibility

### Device Responsibilities

- Cache loading, input-channel construction, and near-surface initialization
  occur on CPU.
- The model and mesh-loss tensors use `--device`, which defaults to CUDA when
  available and CPU otherwise.
- Delaunay construction uses the repository `pygdel3d` extension.
- Optional inference extraction initializes the shared DCCVT runtime and calls
  `dccvt.mesh_ops.extract_mesh()`.

The practical training and extraction path requires the repository's working
PyTorch, PyTorch3D-compatible operations, `pygdel3d`, gDel3D, and CUDA setup.
CPU-only behavior for the full experiment is **Needs verification**.

### Batch Size

Training enforces `--batch-size 1`. Delaunay topology and accepted site counts
vary per shape, so the implementation does not batch multiple active fields.

### Seeds

`seed_everything()` seeds Python, NumPy, CPU PyTorch, and all CUDA devices. It
also enables deterministic cuDNN behavior and disables cuDNN benchmarking.

There are two relevant defaults:

- CLI `--seed=69` controls training/inference order and network randomness.
- Config `bootstrap_seed=69` controls background jitter and candidate sampling.

Changing only the CLI seed does not change a config-fixed bootstrap seed.

With `num_workers>0`, each worker owns its own dataset copy and initialization
memoization cache. The default `num_workers=0` computes each shape's
initialization once in the main process.

## Implementation Map

| File | Ownership |
| --- | --- |
| [`dccvt/neural/iter_refine.py`](../dccvt/neural/iter_refine.py) | Typed config, initialization, parent selection, dataset, model, train loop, inference, checkpointing, and prediction export. |
| [`dccvt/neural/grid.py`](../dccvt/neural/grid.py) | Canonical grids, hybrid input channels, point UDF/confidence, and differentiable SDF interpolation. |
| [`dccvt/neural/point_udf_sidecar.py`](../dccvt/neural/point_udf_sidecar.py) | Exact point-UDF sidecar generation, validation, and loading for v3 local features. |
| [`dccvt/neural/losses.py`](../dccvt/neural/losses.py) | Symmetric Chamfer and clipped-mesh training objective. |
| [`dccvt/geometry.py`](../dccvt/geometry.py) | Delaunay wrapper, clipped geometry, CVT geometry, and differentiable projections. |
| [`dccvt/sdf_gradients.py`](../dccvt/sdf_gradients.py) | Site/tetrahedron SDF gradients, eikonal, curvature, and smoothing support. |
| [`dccvt/mesh_ops.py`](../dccvt/mesh_ops.py) | Final `intDCCVT` and `projDCCVT` extraction. |
| [`scripts/train_hybrid_iter_refine.py`](../scripts/train_hybrid_iter_refine.py) | Thin training wrapper. |
| [`scripts/infer_hybrid_iter_refine.py`](../scripts/infer_hybrid_iter_refine.py) | Thin inference wrapper. |
| [`scripts/precompute_hybrid_iter_refine_udf65.py`](../scripts/precompute_hybrid_iter_refine_udf65.py) | Thin v3 point-UDF sidecar precompute wrapper. |
| [`tests/test_neural_iter_refine.py`](../tests/test_neural_iter_refine.py) | Initialization, parent, round-growth, collision, compatibility, and export tests. |

Public neural exports from `dccvt.neural` are:

- `HybridIterRefineConfig`
- `HybridIterRefineDataset`
- `DCCVTHybridIterRefineNet`
- `build_hotspot_near_surface_initialization`
- `select_procedural_refinement_parents`
- `run_iterative_refinement`
- `run_initialization_extraction`

## Local Point-UDF Sidecars

v3 sidecars are generated from existing HotSpot cache `target_points`, not from
ground-truth mesh surface resampling. Each sidecar stores a compressed `65_udf`
array, source cache path, point count, domain `[-1, 1]`, seed, command args, and
preprocessing version.

The sidecar root convention is:

```text
outputs/neural_hotspot_sdf/thingi32_g65_point_udf/<mesh_id>.npz
```

Each sidecar contains:

| Key | Expected value | Use |
| --- | --- | --- |
| `65_udf` | Float array `(65,65,65)` | Exact nearest distance from each grid vertex to the cache `target_points`. |
| `metadata` | JSON string | Full sidecar metadata, including command args. |
| `preprocessing_version` | `point_udf_sidecar_v1` | Compatibility check. |
| `grid_n` | Integer scalar, normally `65` | Must match `local_udf_grid_n`. |
| `coordinate_min`, `coordinate_max` | `-1.0`, `1.0` | Domain convention. |
| `source_cache_path` | String | Cache used to build the sidecar. |
| `source_point_count` | Integer scalar | Number of input points used. |
| `seed` | Integer scalar | Reproducibility metadata. |

```bash
python scripts/precompute_hybrid_iter_refine_udf65.py \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --output-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf \
  --grid-n 65 \
  --fail-fast
```

The script writes one `<mesh_id>.npz` sidecar per cache plus
`resolved_config.json` and `summary.json` in the sidecar root. Use
`--overwrite` to replace existing valid sidecars.

Sidecar command options:

| Argument | Default | Behavior |
| --- | --- | --- |
| `--cache-root` | `outputs/neural_hotspot_sdf/thingi32_g33` | Source HotSpot cache root. |
| `--split-file` | `None` | Optional mesh-id split file. |
| `--mesh-ids` | `None` | Optional comma/space-separated mesh IDs. |
| `--output-root` | `outputs/neural_hotspot_sdf/thingi32_g65_point_udf` | Sidecar output root. |
| `--grid-n` | `65` | Side length of the exact point-UDF grid. v3 config expects 65. |
| `--query-chunk-size` | `2048` | Number of grid query points per `torch.cdist` chunk. |
| `--device` | `auto` | Uses CUDA when available, otherwise CPU. |
| `--seed` | `69` | Recorded in sidecar metadata. |
| `--overwrite` | `False` | Replace existing valid sidecars. |
| `--fail-fast` | `False` | Stop at the first sidecar generation failure. |

## Configuration Reference

The source of truth is `HybridIterRefineConfig`. The active v2 comparison JSON
files write `input_channels=2`; the historical v1 JSON writes
`input_channels=4`. The dataclass itself accepts `None` and derives the value
from `channel_names`.

| Field | Default | Meaning and constraint |
| --- | --- | --- |
| `config_version` | `2` | Supports versions 1 and 2. Missing version is migrated as legacy version 1. |
| `hotspot_grid_n` | `33` | Side length of the cubic HotSpot vertex grid; must be at least 2. |
| `initialization_mode` | `hotspot_near_surface` | `hotspot_near_surface` or compatibility mode `canonical`. |
| `base_grid_n` | `9`; `17` in v3/v4 | Background grid parameter; produces `(base_grid_n-1)^3` sites. |
| `background_jitter_scale` | `0.005` | Standard deviation of background Gaussian jitter; non-negative. |
| `surface_pair_count` | `3236` | Maximum number of initial near-surface sites; must be non-negative and even. |
| `min_surface_anchors` | `128` | Minimum accepted pair anchors for a valid initialization; positive. |
| `projection_steps` | `3` | Number of bounded Newton projection steps; positive. |
| `surface_pair_offset` | `0.03125` | Distance from projected anchor to each pair endpoint; positive. |
| `bootstrap_min_distance` | `0.005` | Minimum distance among initial sites; non-negative. |
| `bootstrap_seed` | `69` | Seed for background jitter and candidate sampling. |
| `bootstrap_candidate_multipliers` | `[4,8,16]` | Positive candidate oversampling factors attempted in order. |
| `input_channels` | `2` in active v2 JSON | Must equal the number of `channel_names`; dataclass `None` derives it. |
| `feature_dim` | `128` | Encoder channel width and sampled parent-feature width. |
| `encoder_layers` | `5` | Number of padded `3x3x3` convolutions after the first vertex-to-feature convolution. |
| `decoder_layers` | `2` | Number of hidden linear/LeakyReLU blocks before child output. |
| `slots_per_parent` | `4` | Child slots per parent; version 2 permits 1 through 4. |
| `max_parents_per_round` | `128` | Maximum unique procedural parents; non-negative. |
| `num_refinement_rounds` | `1` | Maximum iterative rounds; non-negative. |
| `child_stencil_scale` | `0.015625` | Fixed tetrahedral offset magnitude; non-negative. |
| `child_offset_scale` | `0.03125` | Bound on learned coordinate residual after `tanh`; non-negative. |
| `sdf_residual_scale` | `0.0625` | Bound on learned SDF residual after `tanh`; non-negative. |
| `spawn_min_distance` | `0.0025` | Minimum distance for accepted learned children; non-negative. |
| `point_udf_clip` | `4.0` | Maximum normalized point-UDF input value. |
| `point_confidence_sigma_scale` | `1.5` | Gaussian confidence sigma in HotSpot cell-size units. |
| `local_udf_grid_n` | `65` in v3, `0` otherwise | Exact local point-UDF sidecar grid size. |
| `local_udf_samples` | `true` in v3, `false` otherwise | Enables parent and child-stencil UDF samples from the sidecar. |
| `local_knn_features` | `true` in v3, `false` otherwise | Enables local parent KNN statistics from `target_points`. |
| `local_knn_k` | `8` | Neighbor count for local KNN statistics. |
| `local_knn_radius` | `0.0625` | Radius used to normalize KNN distances and count local density. |
| `local_feature_mode` | `udf65_knn_stats` in v3, `none` otherwise | Local decoder feature mode. |
| `parent_selection` | `procedural_zero_crossing_curvature` | Only implemented parent-selection mode. |
| `training_objective` | `mesh_loss_only` | Only implemented training objective. |
| `channel_names` | `["hotspot_sdf", "point_udf"]` in active v2 JSON | Ordered channel list; first entry must be `hotspot_sdf`, names must be valid and unique. |
| `architecture` | `dense_cnn` | `dense_cnn` for v1-v3, `delaunay_gcnn` for the v4 graph ablation. |
| `graph_layers` | `3` | Number of Delaunay message-passing layers for `delaunay_gcnn`. |
| `graph_hidden_dim` | `feature_dim` | Graph hidden width; smoke overrides of `--feature-dim` also shrink this when tied to `feature_dim`. |
| `site_position_encoding` | `fourier` | Site-coordinate encoding mode for graph nodes. |
| `site_position_num_frequencies` | `4` | Fourier bands use frequencies `1,2,4,8`. |
| `graph_edge_features` | `relative_xyz_distance_direction_sdf_delta` | Directed Delaunay edge feature set. |

Legacy config dictionaries without `config_version` are interpreted as version
1, canonical initialization, no background jitter, no child stencil, and no
spawn spacing filter. This exists for checkpoint loading, not as the recommended
v1 experiment setting.

## Training Command Reference

Entry point:

```bash
python scripts/train_hybrid_iter_refine.py [options]
```

| Argument | Default | Behavior |
| --- | --- | --- |
| `--config` | `configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p128.json` | JSON model config. Pass `configs/neural_hybrid_iter_refine_v1.json` only to reproduce the historical four-channel run. |
| `--cache-root` | `outputs/neural_hotspot_sdf/thingi32_g33` | Directory containing cache NPZ files. |
| `--local-udf-root` | `None` | Directory containing v3 point-UDF sidecars. Required when the config enables local UDF samples unless `--allow-missing-local-features` is passed. |
| `--allow-missing-local-features` | `False` | Use zero local UDF features when sidecars are missing. Intended only for smoke/debug runs. |
| `--split-file` | `None` | Text file of mesh IDs; takes precedence over `--mesh-ids`. |
| `--mesh-ids` | `None` | Comma/space-separated IDs. If both selectors are absent, all cache NPZ files are used. |
| `--checkpoint-dir` | `outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r1_p128/checkpoints` | Checkpoints and resolved config output. |
| `--resume` | `None` | Checkpoint to resume. |
| `--resume-optimizer` | `False` | Restore optimizer state; otherwise optimizer starts fresh. |
| `--epochs` | `100` | Number of epochs for this invocation. On resume these are additional epochs. |
| `--batch-size` | `1` | Must remain 1. |
| `--target-subsample` | `None` | Random target-point count used for channels and loss. |
| `--lr` | `6.4e-5` | AdamW learning rate. |
| `--device` | `auto` | `cuda` when available, otherwise `cpu`, or an explicit torch device. |
| `--num-workers` | `0` | DataLoader workers. |
| `--seed` | `69` | Training, shuffle, NumPy, and torch seed. |
| `--w-mesh-chamfer` | `1000.0` | Chamfer loss weight. |
| `--w-mesh-cvt` | `100.0` | CVT loss weight; zero skips CVT computation. |
| `--w-mesh-sdfsmooth` | `100.0` | SDF smoothness weight; zero skips smoothing computation. |
| `--strict-mesh-loss` | `False` | Raise on invalid mesh-loss geometry instead of skipping. |
| `--strict-initialization` | `False` | Raise on invalid HotSpot initialization instead of skipping. |
| `--save-every` | `10` | Number of local epochs between numbered snapshots; `latest.pt` is saved every epoch and the final local epoch also gets a numbered snapshot. |
| `--initialization-mode` | Config value | Override initialization mode. |
| `--hotspot-grid-n` | Config value | Override HotSpot grid size. |
| `--base-grid-n` | Config value | Override background-grid parameter. |
| `--feature-dim` | Config value | Override encoder width. |
| `--encoder-layers` | Config value | Override additional convolution count. |
| `--decoder-layers` | Config value | Override hidden decoder block count. |
| `--slots-per-parent` | Config value | Override child slots. |
| `--max-parents-per-round` | Config value | Override parent cap. |
| `--num-refinement-rounds` | Config value | Override round count. |
| `--child-offset-scale` | Config value | Override learned coordinate residual bound. |
| `--sdf-residual-scale` | Config value | Override learned SDF residual bound. |

The CLI does not expose every config field. Pair budgets, projection parameters,
candidate multipliers, spacing thresholds, bootstrap seed, stencil scale, input
channel parameters, and fixed mode names must be changed through a JSON config.

On resume, the checkpoint's model config is used after verifying that its
initialization mode matches the requested config. Other requested model
overrides do not replace the resumed checkpoint architecture.

## Inference Command Reference

Entry point:

```bash
python scripts/infer_hybrid_iter_refine.py [options]
```

| Argument | Default | Behavior |
| --- | --- | --- |
| `--checkpoint` | Required | Iterative-refinement checkpoint. |
| `--cache` | Required | One HotSpot cache NPZ. |
| `--local-udf-root` | `None` | Directory containing v3 point-UDF sidecars. Required when the checkpoint config enables local UDF samples unless missing features are explicitly allowed. |
| `--allow-missing-local-features` | `False` | Use zero local UDF features if the sidecar is missing. Intended only for smoke/debug runs. |
| `--output-dir` | Required | Prediction and optional mesh output directory. |
| `--device` | `auto` | Model device selection. |
| `--seed` | `69` | Inference and extraction seed. |
| `--no-extract` | `False` | Save prediction only; do not call shared mesh extraction. |
| `--w-cvt` | `100.0` | Value used in extracted artifact naming and extraction args. |
| `--w-sdfsmooth` | `100.0` | Value used in extracted artifact naming and extraction args. |

Inference always rebuilds the deterministic initialization from the cache and
the checkpoint's saved model config. Extraction requires at least five final
sites and both SDF signs.

## Generated Outputs

### Training Directory

The checkpoint directory contains:

| File | Contents |
| --- | --- |
| `resolved_config.json` | Resolved model config, seed, and parsed CLI arguments. |
| `latest.pt` | Checkpoint overwritten every epoch. |
| `epoch_XXXX.pt` | Numbered snapshots controlled by `--save-every`. |

Checkpoint payload keys are:

- `config_version`
- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `model_config`
- `seed`
- `args`
- `stats`

### Prediction NPZ

Inference writes:

```text
<output-dir>/<mesh_id>_hybrid_iter_refine_prediction.npz
```

Core arrays:

| Key | Meaning |
| --- | --- |
| `sites`, `sites_sdf` | Final active field after accepted refinement children. |
| `base_sites`, `base_sites_sdf` | Complete initialization field, not only the background grid. |
| `background_sites`, `background_sites_sdf` | Jittered domain-support field. |
| `surface_anchors` | Projected zero-level anchors, one per signed pair. |
| `surface_sites`, `surface_sites_sdf` | Flattened inside/outside initialization pairs. |
| `input_grid` | Hybrid neural input `(C,G,G,G)`. |
| `sdf_grid` | Cached HotSpot grid. |
| `target_points` | Cache target points used during inference. |
| `local_udf_path` | Sidecar path used by v3 inference, empty for non-v3 configs. |
| `local_udf_valid` | Boolean sidecar validity flag recorded for v3 inference. |
| `diagnostics` | JSON string with final and initialization diagnostics. |
| `resolved_config` | Checkpoint model config as JSON. |
| `command_args` | Inference arguments as JSON. |
| `seed`, `mesh_id` | Reproducibility metadata. |

For each round `RR`, the NPZ also contains:

- `round_RR_parent_indices`
- `round_RR_parent_scores`
- `round_RR_spawned_sites`
- `round_RR_spawned_sdf`
- `round_RR_rejected_spawn_count`

### Extraction Artifacts

When extraction is enabled, the shared extractor writes both variants:

```text
DCCVT_<rounds>_hybrid_iter_refine_intDCCVT_cvt<W>_sdfsmooth<W>.obj
DCCVT_<rounds>_hybrid_iter_refine_intDCCVT_cvt<W>_sdfsmooth<W>.npz
DCCVT_<rounds>_hybrid_iter_refine_projDCCVT_cvt<W>_sdfsmooth<W>.obj
DCCVT_<rounds>_hybrid_iter_refine_projDCCVT_cvt<W>_sdfsmooth<W>.npz
target.ply
inference_result.json
```

`inference_result.json` records prediction path, extraction status, site count,
and initialization validity/reason.

Important v3 diagnostics in the training log and prediction NPZ include:

- `local_feature_mode`;
- `local_udf_grid_n`;
- `local_udf_samples`;
- `local_udf_path`;
- `local_udf_valid`;
- `local_knn_features`;
- `local_target_point_count`;
- `round_XX_parent_count`;
- `round_XX_spawned_site_count`;
- `round_XX_rejected_spawn_count`.

## Reproduction Workflow

Run all commands from the repository root:

```text
/export/livia/home/vision/Wcharawi/dev/DCCVT
```

Activate the repository environment first. The verified environment used
`/tmp/dccvt-venv/bin/python`; a normal activated environment can use `python`.

| Workflow | Required input | Files written | Runtime requirement |
| --- | --- | --- | --- |
| Initialization-only baseline | HotSpot cache split and initialization v2 config | Initialization field NPZ, resolved config, optional meshes, and summary JSON | No learned checkpoint; mesh extraction requires DCCVT runtime. |
| Minimal training smoke | One cache selected by the smoke split and a two-channel v2 config | Resolved config and checkpoints | Verified with CUDA, `pygdel3d`, and gDel3D. |
| v3 sidecar precompute | HotSpot cache split | `65^3` point-UDF sidecars, resolved config, and summary JSON | CUDA recommended; CPU support is slower but expected. |
| v3 local refinement smoke | One cache, matching sidecar, and v3 config | Resolved config, checkpoints, prediction NPZ, and optional meshes | Verified with CUDA, `pygdel3d`, and gDel3D. |
| Inference and extraction | One checkpoint and one cache | Prediction NPZ, result JSON, both mesh variants, mesh bundles, and target PLY | Verified with CUDA, `pygdel3d`, and gDel3D. |
| No-extract inference | One checkpoint and one cache | Prediction NPZ and result JSON | Still requires Delaunay parent selection; CPU-only support is Needs verification. |
| Resume training | Matching checkpoint, config, caches, and split | Updates `latest.pt` and writes configured snapshots | Same runtime requirements as training. |
| Focused Thingi32 overfits | All 31 caches, full split, and two-channel v2 configs | Long-running checkpoint series and resolved configs | Full 1,000-epoch runtime and memory are Needs verification. |

### Initialization-Only Baseline

This exports the 3,748-site HotSpot near-surface initialization without learned
children. Pass `--no-extract` for a cheap field-only smoke test:

```bash
python scripts/extract_hybrid_iter_refine_initial.py \
  --config configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt \
  --output-root outputs/neural_dccvt/hybrid_iter_refine_initial_v2_hotspot_point_udf_smoke \
  --no-extract \
  --overwrite \
  --fail-fast
```

### Minimal Training Smoke Test

This keeps the default geometry and site budgets but reduces neural width and
disables expensive auxiliary mesh losses:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128_smoke/checkpoints \
  --epochs 1 \
  --target-subsample 64 \
  --feature-dim 8 \
  --encoder-layers 1 \
  --decoder-layers 1 \
  --w-mesh-cvt 0 \
  --w-mesh-sdfsmooth 0 \
  --strict-initialization \
  --save-every 1
```

Expected outputs include `resolved_config.json`, `latest.pt`, and
`epoch_0000.pt` in the checkpoint directory.

### v3 Local-UDF/KNN Smoke Test

First precompute the matching sidecar for the smoke split:

```bash
python scripts/precompute_hybrid_iter_refine_udf65.py \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt \
  --output-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf_smoke \
  --grid-n 65 \
  --overwrite \
  --fail-fast
```

Then train one reduced-width epoch with v3 local features:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --local-udf-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf_smoke \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128_smoke/checkpoints \
  --epochs 1 \
  --target-subsample 64 \
  --feature-dim 8 \
  --encoder-layers 1 \
  --decoder-layers 1 \
  --w-mesh-cvt 0 \
  --w-mesh-sdfsmooth 0 \
  --strict-mesh-loss \
  --save-every 1
```

The verified smoke for mesh `252119` reported both refinement rounds active,
valid local UDF features, and `local_target_point_count=9600`.

Run v3 inference and extraction with the same sidecar root:

```bash
python scripts/infer_hybrid_iter_refine.py \
  --checkpoint outputs/neural_dccvt/hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128_smoke/checkpoints/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/252119.npz \
  --local-udf-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf_smoke \
  --output-dir outputs/neural_dccvt/hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128_smoke/252119
```

### v4 Delaunay-GCNN Smoke Test

Reuse the same `65^3` point-UDF sidecar generated for v3. Train one
reduced-width epoch with the graph architecture:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --local-udf-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf_smoke \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_smoke.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128_smoke/checkpoints \
  --epochs 1 \
  --target-subsample 64 \
  --feature-dim 8 \
  --w-mesh-cvt 0 \
  --w-mesh-sdfsmooth 0 \
  --strict-mesh-loss \
  --save-every 1
```

Run no-extract inference first to verify checkpoint loading and prediction
export before invoking the DCCVT extraction runtime:

```bash
python scripts/infer_hybrid_iter_refine.py \
  --checkpoint outputs/neural_dccvt/hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128_smoke/checkpoints/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/252119.npz \
  --local-udf-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf_smoke \
  --output-dir outputs/neural_dccvt/hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128_smoke/252119_no_extract \
  --no-extract
```

### Inference and Extraction

```bash
python scripts/infer_hybrid_iter_refine.py \
  --checkpoint outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128_smoke/checkpoints/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/252119.npz \
  --output-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128_smoke/252119
```

This writes the prediction NPZ, both DCCVT mesh variants, `target.ply`, and
`inference_result.json`.

### Prediction Without Mesh Extraction

```bash
python scripts/infer_hybrid_iter_refine.py \
  --checkpoint outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128_smoke/checkpoints/latest.pt \
  --cache outputs/neural_hotspot_sdf/thingi32_g33/252119.npz \
  --output-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128_smoke/252119_no_extract \
  --no-extract
```

This still runs procedural parent selection, which requires Delaunay support.
It only skips final OBJ/mesh-bundle extraction.

### Resume Training

`--epochs` is the number of additional epochs in the resumed invocation:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128/checkpoints \
  --resume outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128/checkpoints/latest.pt \
  --resume-optimizer \
  --epochs 100 \
  --target-subsample 4096
```

The requested config must have the same initialization mode as the checkpoint.

### Focused Thingi32 Overfit Runs

This trains repeatedly on all 31 cached shapes with no held-out validation
split:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r1_p128/checkpoints \
  --epochs 1000 \
  --target-subsample 4096 \
  --lr 6.4e-5 \
  --save-every 25 \
  --seed 69

python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r2_p128/checkpoints \
  --epochs 1000 \
  --target-subsample 4096 \
  --lr 6.4e-5 \
  --save-every 25 \
  --seed 69

python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p256.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r1_p256/checkpoints \
  --epochs 1000 \
  --target-subsample 4096 \
  --lr 6.4e-5 \
  --save-every 25 \
  --seed 69
```

These full 1,000-epoch commands have not been completed as part of the current
verification.

### v3 Full Thingi32 Run

Precompute sidecars for the full 31-shape split:

```bash
python scripts/precompute_hybrid_iter_refine_udf65.py \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --output-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf \
  --grid-n 65 \
  --fail-fast
```

Train the active v3 local-refinement comparison:

```bash
python scripts/train_hybrid_iter_refine.py \
  --config configs/neural_hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128.json \
  --cache-root outputs/neural_hotspot_sdf/thingi32_g33 \
  --local-udf-root outputs/neural_hotspot_sdf/thingi32_g65_point_udf \
  --split-file PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt \
  --checkpoint-dir outputs/neural_dccvt/hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128/checkpoints \
  --epochs 1000 \
  --target-subsample 4096 \
  --lr 6.4e-5 \
  --save-every 25 \
  --seed 69
```

This run uses the full cache `target_points` for local KNN features even though
the coarse `point_udf` channel and mesh-loss target are subsampled to 4096
points per batch.

## Verified Behavior

The following observations were verified during implementation. They are
runtime diagnostics, not reconstruction-quality conclusions.

### Initialization Audit

All 31 caches listed in
`PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt` produced valid default
initializations:

- valid shapes: `31 / 31`;
- initial sites per shape: `3748` for v1/v2; overwritten v3/v4 use `7332`;
- surface anchors per shape: `1618`;
- minimum observed site distance: approximately `0.005001`;
- every field contained positive and negative SDF values.

The audit output was observed in the verification terminal and was not saved as
a dedicated repository result file.

### Historical v1 One-Shape Smoke

For mesh `252119`, the historical four-channel v1 saved prediction under
`outputs/neural_dccvt/hybrid_iter_refine_v1_smoke/252119_near/` contains:

- 512 background sites;
- 1,618 surface anchors;
- 3,236 paired surface sites;
- 3,748 complete initialization sites;
- 128 selected parents;
- 511 accepted children and one rejected child;
- 4,259 final sites;
- 1,652 negative and 2,607 positive final SDF values;
- successful `intDCCVT` and `projDCCVT` extraction.

The final post-implementation one-epoch smoke checkpoint reported:

```text
mesh_loss       = 5.7519064
mesh_chamfer    = 0.0057519064
mesh_used_shapes = 1
mesh_skipped_shapes = 0
```

CVT and SDF smoothness weights were zero for this smoke command.

### Reduced-Network Full-Split Smoke

A one-epoch run over all 31 shapes, using default geometry budgets but
`feature_dim=8`, one encoder layer, one decoder layer, 64 target points, and
zero CVT/smoothness weights completed without a geometry skip. Its checkpoint
reported averages:

```text
mesh_loss          = 8.6885609
mesh_chamfer       = 0.0086885609
site_count         = 4259.7096774
initial_site_count = 3748
surface_anchors    = 1618
mesh_used_shapes   = 1.0 per batch
mesh_skipped_shapes = 0.0 per batch
```

No full-width 1,000-epoch model or final `ponq_thingi`, `raw`, and
`bbox_aligned` quality evaluation has been completed. Comparative Chamfer,
normal consistency, F1, and edge metrics are therefore **Needs verification**.

### v3 Local-UDF/KNN Smoke

For mesh `252119`, the v3 smoke generated one `65^3` sidecar from the existing
cache `target_points`, then ran one reduced-width training epoch and inference
with extraction. The training checkpoint reported:

```text
site_count                   = 4703
local_udf_grid_n              = 65
local_knn_features            = 1
local_target_point_count      = 9600
local_udf_valid               = 1
round_00_parent_count         = 128
round_00_spawned_site_count   = 511
round_00_rejected_spawn_count = 1
round_01_parent_count         = 128
round_01_spawned_site_count   = 444
round_01_rejected_spawn_count = 68
```

The subsequent inference/export smoke wrote a prediction with:

```text
round_count     = 2
site_count      = 4702
local_udf_valid = true
```

Both `intDCCVT` and `projDCCVT` extraction completed. This verifies feature
plumbing and extraction, not reconstruction-quality improvement.

## Tests

`tests/test_neural_iter_refine.py` verifies:

- unique procedural parent indices and safe empty selection;
- deterministic, finite, sign-balanced initialization;
- the default 3,748/4,260 site-budget arithmetic;
- no-crossing invalid status;
- finite in-domain one-round children;
- monotonic multi-round site growth;
- exact `65^3` point-UDF sidecar distances and metadata;
- finite local KNN statistics when fewer than K target points are available;
- v3 config local-feature settings;
- v3 default-initialization forward growth with local features;
- collision rejection;
- legacy config migration and resume-mode rejection;
- exported initialization and per-round metadata.

Run the focused suite with:

```bash
python -m pytest tests/test_neural_iter_refine.py -q
```

## Common Failure Cases

### `no_sign_changing_cells`

The HotSpot grid has no cell whose corners contain both strict signs. The
near-surface initializer cannot establish a signed surface bracket.

Actions:

- inspect `sdf_grid.min()` and `sdf_grid.max()`;
- verify normalization and HotSpot checkpoint quality;
- use `--strict-initialization` to stop at the first affected shape.

### `insufficient_valid_surface_pairs`

Fewer than `min_surface_anchors` pairs survive finite-value, domain, sign, FPS,
and spacing tests.

Actions:

- inspect initialization diagnostics for crossing/candidate counts;
- verify `surface_pair_offset` is appropriate for the grid resolution;
- test spacing and candidate-multiplier changes in a separate config.

### Final Count Below 4,260

This is normally expected. `spawn_min_distance` can reject child candidates.
Inspect `round_00_rejected_spawn_count` rather than assuming a fixed final
shape.

### Empty Parent Selection

The active field lacks both signs, has no zero-crossing Delaunay edge, or has no
valid Delaunay tetrahedra. The round is recorded with empty children and forward
iteration stops.

### Mesh Loss Is Skipped

Inspect `mesh_used_shapes`, `mesh_skipped_shapes`, and `mesh_no_grad_batch`.
Use `--strict-mesh-loss` to expose the underlying geometry exception.

### Missing Local UDF Sidecar

v3 configs require `--local-udf-root` during training and inference. Missing or
invalid sidecars raise an error by default.

Actions:

- run `scripts/precompute_hybrid_iter_refine_udf65.py` for the same split;
- verify the sidecar root contains `<mesh_id>.npz`;
- verify sidecar metadata uses `grid_n=65` and preprocessing version
  `point_udf_sidecar_v1`;
- use `--allow-missing-local-features` only for deliberate smoke/debug runs,
  because it replaces sidecar samples with zeros.

### Local KNN Count Looks Subsampled

For v3, `local_target_point_count` should report the full cache point count for
the loaded mesh. If it equals `--target-subsample`, inspect dataset loading or
the cache itself; the intended behavior is full-point KNN features and
subsampled coarse channel/loss.

### gDel3D Assertions or Process Termination

Likely causes include duplicate/near-duplicate sites, degenerate geometry,
invalid coordinates, or runtime incompatibility. Inspect:

- `unique_site_count` and `minimum_site_distance`;
- per-round rejected counts;
- finiteness and domain bounds;
- CUDA, gDel3D, and `pygdel3d` build compatibility.

Python exception handling cannot recover from every native assertion or process
termination.

### Resume Initialization Mismatch

Training rejects a checkpoint whose saved initialization mode differs from the
requested config. Use the original config for resume or start a new checkpoint
directory.

### Cache Grid Mismatch

`grid_n` must equal `hotspot_grid_n`, and `sdf_grid` must be cubic. Regenerate
the cache or use a matching model config; do not silently reshape it.

### Missing Mesh Artifacts

`--no-extract` intentionally creates no OBJ files. Otherwise check
`inference_result.json`, initialization validity, final SDF signs, and the
shared extraction runtime.

## Known Assumptions and Limitations

- Batch size is fixed to one.
- Base/background and initial paired sites are procedural and not learned.
- Parent activity is procedural, non-differentiable, and recomputed with
  Delaunay every round.
- Collision filtering is a hard detached decision.
- The same encoder feature grid is reused across all rounds.
- Input point channels and mesh-loss targets come from the same cached target
  points; this experiment is currently an overfit setting, not a clean unseen
  shape generalization study.
- v3 sidecars are derived from the same cached target points; they do not add
  ground-truth mesh samples.
- v3 local KNN features use the full cache point set, while
  `--target-subsample` still controls the coarse `point_udf` channel and
  mesh-loss target.
- Initialization is memoized only within each dataset-process instance.
- Parent-selection Delaunay and final mesh-loss Delaunay are computed
  separately.
- Runtime grows with site count and number of rounds; no large-round scaling
  study has been completed.
- CPU-only full training and extraction are **Needs verification**.
- Full training quality and standard metric comparisons are **Needs
  verification**.

## Focused Future Ablations

These are proposed experiments, not implemented conclusions:

1. **Round count**: compare one and two rounds while logging site growth,
   Delaunay time, rejection rate, and mesh quality.
2. **Site allocation**: hold the final budget fixed while varying background
   count versus signed-pair count.
3. **Pair geometry**: ablate projection steps and `surface_pair_offset` relative
   to HotSpot cell size.
4. **Spacing**: sweep initialization and spawn minimum distances while tracking
   native Delaunay failures and rejected children.
5. **Parent budget**: compare 32, 64, and 128 parents with four slots each.
6. **Loss composition**: isolate Chamfer, CVT, and SDF smoothness after a stable
   Chamfer-only warmup.
7. **Parent selection**: compare the current procedural score against a learned
   activity head only after a differentiable or supervised selection strategy
   is specified.
8. **Base learning**: test bounded background-site motion separately from child
   refinement so attribution remains clear.
