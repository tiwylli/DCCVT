# DCCVT acceleration benchmark

## Summary

The `dccvt-accel` work produced a measured **2.83x end-to-end training speedup**, reducing training time by **64.64%** on the recorded control case.

| Measurement | Pre-optimization | Accelerated | Change |
| --- | ---: | ---: | ---: |
| Training time | 312.987 s (5 min 12.99 s) | 110.662 s (1 min 50.66 s) | -202.326 s |
| Average time per iteration | 312.987 ms | 110.662 ms | -202.326 ms |
| Relative runtime | 100% | 35.36% | **-64.64%** |
| Throughput speedup | 1.00x | **2.83x** | +182.83% |

Calculations:

```text
speedup = 312.9872496128082 / 110.66170358657837
        = 2.8283248809x

runtime reduction = (312.9872496128082 - 110.66170358657837)
                    / 312.9872496128082 * 100
                  = 64.643382846%
```

## Git provenance

- Pre-optimization commit: `f624acf`
- Acceleration commit: `e7c633f` (`dccvt-accel`)
- Merge commit: `a4201d4`
- Merge subject: `Merge pull request #3 from tiwylli/dccvt-accel`
- Acceleration commit date: June 22, 2026
- Merge date: June 23, 2026

The repository has no release tag marking an official baseline. Therefore, this document uses the direct parent of the acceleration commit, `f624acf`, as the pre-optimization baseline.

## Control configuration

Both recorded runs used the same nominal DCCVT configuration:

| Parameter | Value |
| --- | ---: |
| Object | `313444` |
| Configuration | `313444_b32_nearsample_ups` |
| Iterations | 1,000 |
| Initial sites | 8,192 |
| Final sites | 36,260 pre-opt / 36,120 accelerated |
| Upsampling stages | 10 |
| Near-surface samples | 4,096 |
| Maximum site parameter | 32 |
| CVT weight | 100 |
| SDF smoothing weight | 100 |
| Chamfer weight | 1,000 |
| Site learning rate | 0.0005 |

The slight difference in final site count is consistent with the adaptive/stochastic upsampling path. It should be considered when interpreting the result as a single-run control rather than a multi-run statistical benchmark.

The recorded `train_time` covers `run_dccvt_training(...)`. Initial and final mesh extraction are outside this timer; see [`dccvt/runner.py`](../dccvt/runner.py).

## Changes responsible for the acceleration

The `dccvt-accel` commit changed the expensive geometry and SDF-gradient paths:

1. **Cheaper tetrahedron circumcenters**

   The original implementation constructed four batched 4x4 matrices and evaluated four determinants per tetrahedron. The accelerated implementation uses the cross-product circumcenter formula. It retains the determinant implementation only as a fallback for nearly flat tetrahedra.

2. **Direct solves instead of an SVD pseudoinverse for every tetrahedron**

   SDF gradient fitting originally called `torch.linalg.pinv` for every normal-equation matrix. The accelerated path uses `torch.linalg.solve` for well-conditioned systems and reserves the pseudoinverse for the small ill-conditioned subset.

3. **Less tensor construction and repeated indexing**

   Site coordinates, SDF values, tetrahedron gradients, and volumes are gathered in compact batched tensors. Repeated four-way indexing and stacking were consolidated.

4. **Lower-overhead centroid accumulation**

   CVT centroid accumulation no longer expands all Voronoi centers with `repeat_interleave`. It accumulates the four tetrahedron corners directly with `index_add_`.

5. **Fewer unnecessary copies and device transfers**

   Examples include using `np.asarray` instead of an unconditional array copy and preserving the input tensor's device when creating tetrahedron tensors.

6. **Cheaper gradient clearing**

   Training changed from `optimizer.zero_grad()` to `optimizer.zero_grad(set_to_none=True)`, avoiding needless zero writes and allowing gradient buffers to be recreated as needed.

The main implementations are in [`dccvt/geometry.py`](../dccvt/geometry.py), [`dccvt/sdf_gradients.py`](../dccvt/sdf_gradients.py), and [`dccvt/training.py`](../dccvt/training.py).

## Output-quality control

The paired quality evaluation indicates that the acceleration broadly preserved reconstruction quality, although this was only one control object.

### Interpolated DCCVT extraction

| Metric | Pre-optimization | Accelerated | Relative change |
| --- | ---: | ---: | ---: |
| Chamfer distance 1 | 574.145606 | 575.956982 | +0.315% |
| Chamfer distance 2 | 2.644732 | 2.685164 | +1.529% |
| F1 score | 0.598846 | 0.599026 | +0.030% |
| Normal consistency | 0.966081 | 0.965785 | -0.031% |
| Hausdorff distance 2 | 42.001282 | 49.669331 | +18.257% |

### Projected DCCVT extraction

| Metric | Pre-optimization | Accelerated | Relative change |
| --- | ---: | ---: | ---: |
| Chamfer distance 1 | 623.322325 | 605.717441 | -2.824% |
| Chamfer distance 2 | 4.135806 | 3.342174 | -19.189% |
| F1 score | 0.588979 | 0.588868 | -0.019% |
| Normal consistency | 0.888302 | 0.892008 | +0.417% |
| Hausdorff distance 2 | 514.768005 | 316.830780 | -38.452% |

These variations should not be interpreted as statistically significant improvements or regressions without repeated runs across more objects.

## Profiler snapshot: November 17, 2026

The accelerated code was profiled again at commit `dd015ad`, the current `master`/`paragram` tip and a descendant of the `dccvt-accel` merge. The run used an NVIDIA GeForce RTX 3080 (10 GiB), PyTorch `2.7.1+cu126`, and CUDA 12.6.

Two complementary profiles were collected:

1. A full 1,000-iteration run with 10 adaptive upsampling stages, starting from 8,192 sites and ending at 36,196 sites. Python `cProfile`, synchronized wall timing, PyTorch CUDA peak-memory counters, and process maximum RSS were enabled.
2. A CUDA operator trace of 10 iterations at the final 36,120-site scale. Three warm-up iterations ran first to exclude one-time CUDA module loading. This trace had no upsampling.

The full profiled run took **114.592 s**, or **114.592 ms/iteration** averaged across all site counts. That is 3.55% slower than the recorded 110.662 s accelerated control, which is reasonable profiler overhead and run-to-run variation. The operator trace took 2.154 s including tracing and forced synchronization; that elapsed value is not a benchmark.

### End-to-end time hotspots

The following table uses non-overlapping `cProfile` self time from the full run. CUDA is asynchronous, so a Python/C boundary can inherit time from synchronization with earlier GPU work. The separate CUDA trace below is the better source for individual GPU operators.

| Hotspot | Self time | Share of full profiled runtime | Interpretation |
| --- | ---: | ---: | --- |
| Autograd engine backward pass | 31.448 s | 27.45% | Backpropagation through clipping, SDF, CVT, and Chamfer computations |
| Chamfer single-direction computations | 27.442 s | 23.95% | Two nearest-neighbor directions per iteration |
| `pygdel3d.triangulate` | 27.329 s | 23.85% | Delaunay reconstruction from the current sites every iteration |
| SDF least-squares weights | 8.368 s | 7.30% | Batched normal matrices, direct solves, and ill-conditioned fallback |
| `torch.linalg.eigh` | 4.192 s | 3.66% | Plane normals used to project zero-crossing vertices |
| Zero-crossing site-pair discovery | 3.834 s | 3.35% | Edge construction and sign-change filtering |
| **Total shown** | **102.613 s** | **89.55%** | The remaining operations are individually much smaller |

The clearest high-level conclusion is that the remaining time is split among three similarly important areas: backward propagation, Chamfer nearest-neighbor search, and rebuilding the Delaunay triangulation.

### CUDA operator hotspots at 36,120 sites

The warmed 10-iteration trace recorded 975.373 ms of aggregate self CUDA time. Parent operations and their underlying kernel rows describe the same work, so only the framework-level operation is listed here to avoid double counting.

| CUDA operation | Self CUDA time | Share of self CUDA time | Main source |
| --- | ---: | ---: | --- |
| Batched matrix multiplication (`aten::bmm`) | 353.425 ms | 36.23% | SDF least-squares matrices and their backward pass |
| Nearest-neighbor search (`_knn_points`) | 278.751 ms | 28.58% | Bidirectional Chamfer loss |
| Indexed writes (`aten::_index_put_impl_`) | 46.391 ms | 4.76% | Geometry masks, scatter-style updates, and backward |
| Sort (`aten::sort`) | 44.729 ms | 4.59% | Geometry/topology processing |
| Symmetric eigensolve (`aten::_linalg_eigh`) | 25.820 ms | 2.65% | Fitted plane normals |
| Tensor copies (`aten::copy_`) | 23.910 ms | 2.45% | Data movement among intermediate tensors |

The Delaunay cost is distributed over many `gDel3D` kernels and CPU-side orchestration rather than one PyTorch operator. Its impact is represented more accurately by the 27.329 s `pygdel3d.triangulate` wall-time entry in the full profile.

### Current accelerated memory footprint

These are per-process PyTorch CUDA allocator measurements for the full 1,000-iteration run. The baseline was taken after loading the model and inputs, clearing the cache, and immediately before training.

| CUDA memory measurement | Baseline | Peak | End of training | Increase from baseline to peak |
| --- | ---: | ---: | ---: | ---: |
| Live tensor memory (`memory_allocated`) | 9.125 MiB | **407.552 MiB** | 18.230 MiB | **398.427 MiB** |
| PyTorch caching pool (`memory_reserved`) | 22.000 MiB | **1,010.000 MiB** | 1,010.000 MiB | **988.000 MiB** |

The complete process reached **1,868.445 MiB maximum host RSS**. This includes Python, imported native libraries, the model, inputs, profiler state, and CPU geometry data, so it is not an incremental training-only measurement.

The much smaller 18.230 MiB of live CUDA tensors at completion versus the 407.552 MiB peak indicates temporary working tensors rather than a retained-tensor leak. The 1,010 MiB reserved value is not simultaneously live data: PyTorch keeps previously acquired CUDA blocks in its caching allocator for reuse.

The final-scale operator trace identifies allocation turnover rather than simultaneous residency. Across its 10 iterations, generic allocator-producing operations (`empty`, `empty_strided`, and `resize_`) accounted for 8.23 GiB of cumulative allocations; elementwise `mul`, `div`, `neg`, `remainder`, and `sub` accounted for about 5.09 GiB. `bmm` allocated 781.46 MiB cumulatively, `sort` 335.70 MiB, indexing 306.96 MiB of self allocations, and the batched solve 206.44 MiB. These totals exceed the peak because buffers are repeatedly created, freed, and reused.

### Most promising next targets

1. **Chamfer KNN:** at 28.58% of self CUDA time, sampling fewer comparison points, evaluating Chamfer less frequently, or using a cheaper nearest-neighbor strategy would directly attack a major remaining cost.
2. **Batched SDF algebra and backward:** `bmm` is the largest GPU operator and also creates substantial temporary tensors. Reducing saved intermediates, fusing adjacent operations, or using a custom backward are candidates for measurement.
3. **Per-iteration Delaunay rebuild:** triangulation consumes 23.85% of full-run time. Reusing topology for several small site updates, if quality remains stable, could avoid the CPU/GPU round trip and repeated reconstruction.
4. **Temporary allocation churn:** buffer reuse or fused elementwise expressions may lower allocator pressure. The present peak is modest on a 10 GiB GPU, so this is more likely a throughput optimization than a capacity requirement.

## Memory result

**No controlled pre/post peak-memory metric was recorded for the `dccvt-accel` comparison.**

The paired NPZ files contain only:

- `sites`
- `sites_sdf`
- `train_time`
- `args`

Historical notebooks contain occasional `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` snapshots, but they use different workloads and are not a valid baseline-versus-accelerated comparison. The acceleration changes should reduce temporary tensor pressure, particularly by avoiding repeated determinant matrices, unconditional pseudoinverses, and `repeat_interleave`, but the repository does not contain a defensible measured memory-reduction percentage.

The current accelerated footprint is now measured in the profiler snapshot above, but a memory **improvement percentage** still cannot be calculated without running the same instrumentation on `f624acf` under the same environment.

A future benchmark should record at least:

```python
torch.cuda.reset_peak_memory_stats()
# Run the controlled workload.
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

It should repeat both commits on the same GPU, input, seed, dependency versions, and CUDA clock settings.

## Evidence files

The timing and quality results are present in the local workspace:

- Pre-optimization timing: [`outputs/preopt_control_20260622/313444_b32_nearsample_ups/DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100.npz`](../outputs/preopt_control_20260622/313444_b32_nearsample_ups/DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100.npz)
- Accelerated timing: [`outputs/optimized_control_20260622/313444_b32_nearsample_ups/DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100.npz`](../outputs/optimized_control_20260622/313444_b32_nearsample_ups/DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100.npz)
- Pre-optimization quality metrics: [`outputs/preopt_control_20260622/metrics.csv`](../outputs/preopt_control_20260622/metrics.csv)
- Accelerated quality metrics: [`outputs/optimized_control_20260622/metrics.csv`](../outputs/optimized_control_20260622/metrics.csv)
- Pre-optimization run log: [`outputs/preopt_control_20260622/run.log`](../outputs/preopt_control_20260622/run.log)

The `outputs/` directory is ignored by Git, so these evidence files are not preserved in repository history. If this report will be shared or committed for long-term reference, archive the small CSV files and benchmark metadata in a tracked benchmark directory. The NPZ files can remain external if repository size is a concern.
