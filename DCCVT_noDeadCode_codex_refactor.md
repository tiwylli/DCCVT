# DCCVT_noDeadCode_codex.py refactor plan and analysis

## Behavior audit (inputs, outputs, side effects, randomness)

### Inputs
- CLI args parsed from `--args-file` (template lines expanded into per-mesh argv lists).
- Optional CLI overrides: `--mesh-ids`, `--timestamp`, `--dry-run`.
- Filesystem inputs: mesh `.ply` files, HotSpot model `.pth` files, arg templates.
- Optional GPU/driver availability (`nvidia-smi`, CUDA runtime).

### Outputs
- Mesh `.obj` files for multiple states and variants (DCCVT, voromesh, MT).
- `.npz` metadata (sites, SDF values, training time, args).
- Target point cloud `.ply`.
- Copied entrypoint script + `arg_lists.txt` for reproducibility.
- Console logging (progress, diagnostics, estimations).

### Side effects (filesystem)
- Creates output directories under `${ROOT_DIR}/outputs/${timestamp}`.
- Writes artifacts listed above.
- Copies the entrypoint script and arg list into the output directory.

### Randomness / determinism
- Seeds set once at startup: `torch.manual_seed(69)`, `np.random.seed(69)`.
- Uses CUDA deterministic mode and disables cuDNN benchmarking.
- Randomness still occurs within training/upsampling steps (expected, seeded).

### External calls and heavy dependencies
- HotSpot dataset/model loading (`3rdparty/HotSpot`).
- gDel3D triangulation and SciPy Delaunay.
- Kaolin marching tetrahedra.
- PyTorch3D chamfer and KNN.
- Gudhi + trimesh for alpha shape SDF.

### Hotspots
- Delaunay/triangulation and Voronoi vertex computation.
- SDF gradient estimation and clipping.
- Training loop with mesh extraction and Chamfer loss.

## Target module layout (proposed)

```
DCCVT_noDeadCode_codex.py        # entrypoint wrapper (CLI preserved)

/dccvt
  __init__.py
  runtime.py                     # device + seeding (side effects)
  config.py                      # ROOT_DIR, DEFAULTS, timestamp
  argparse_utils.py              # args template parsing + per-mesh parser
  paths.py                       # output filename helpers
  io_utils.py                    # save_obj/save_npz/save_target_pc_ply/copy_script
  model_utils.py                 # model + site initialization + SDF resolution
  pipeline.py                    # geometry + training + extraction + process_single_mesh
  main.py                        # CLI wiring for args-file expansion
```

## Refactor steps (safe, mechanical)
1. Separate runtime initialization (device + seeds) into `dccvt/runtime.py`.
2. Move defaults/timestamp to `dccvt/config.py` with a small timestamp updater.
3. Extract args-file parsing and per-mesh parser into `dccvt/argparse_utils.py`.
4. Extract IO helpers and output-path formatting into `dccvt/io_utils.py` + `dccvt/paths.py`.
5. Extract model/SDF init into `dccvt/model_utils.py`.
6. Move the remaining geometry + training pipeline into `dccvt/pipeline.py`.
7. Replace the original script with a thin entrypoint that calls `dccvt.main.main()`.

## Non-goals
- No changes to numerical behavior, thresholds, or control flow.
- No changes to CLI flags or default values.
- No dependency changes.

## Next-step concrete split plan (pending approval)

### Goal
Reduce `dccvt/pipeline.py` into focused modules without changing logic, control flow, or numerics.

### Proposed modules and contents
- `dccvt/geometry.py`
  - Voronoi / Delaunay / clipping utilities:
    - `get_clipped_mesh_numba`
    - `compute_zero_crossing_vertices_3d`
    - `compute_zero_crossing_sites_pairs`
    - `compute_all_bisectors_vectorized`
    - `compute_vertices_3d_vectorized`
    - `interpolate_sdf_of_vertices`
    - `interpolate_sdf_grad_of_vertices`
    - `quaternion_slerp_barycentric`
    - `quaternion_slerp`
    - `tet_plane_clipping`
    - `newton_step_clipping`
    - `circumcenter_torch`
    - `compute_voronoi_cell_centers_index_based_torch`
    - `compute_cvt_loss_vectorized_delaunay`
    - `compute_cvt_loss_CLIPPED_vertices`
    - `compute_cvt_loss_true`
    - `faces_via_dict`
    - `get_faces`
    - Numba helpers: `batch_sort_numba`, `sort_face_loop_numba`
    - Small vector helpers: `_compute_normal`, `_normalize`, `_angle`

- `dccvt/sdf_gradients.py`
  - SDF gradient and curvature utilities:
    - `sdf_space_grad_pytorch_diego_sites_tets`
    - `sdf_space_grad_pytorch_diego`
    - `volume_tetrahedron`
    - `smoothed_heaviside`
    - `tet_sdf_motion_mean_curvature_loss`
    - `discrete_tet_volume_eikonal_loss`
    - `estimate_eps_H`

- `dccvt/upsampling.py`
  - Adaptive upsampling:
    - `upsampling_adaptive_vectorized_sites_sites_sdf`
    - `build_tangent_frame`

- `dccvt/mesh_ops.py`
  - Mesh extraction + sampling:
    - `cvt_extraction`
    - `extract_mesh`
    - `sample_mesh_points_heitz`

- `dccvt/training.py`
  - Optimization loops and losses:
    - `train_DCCVT`
    - `Voroloss_opt`

- `dccvt/alpha_shape.py`
  - Alpha-shape SDF helper:
    - `complex_alpha_sdf`

- `dccvt/runner.py`
  - Experiment glue:
    - `process_single_mesh`
    - `check_if_already_processed`

### Migration sequence (small, reviewable steps)
1. Move geometry-only helpers to `dccvt/geometry.py` and update imports.
2. Move gradient/curvature utilities to `dccvt/sdf_gradients.py` and update imports.
3. Move upsampling helpers to `dccvt/upsampling.py` and update imports.
4. Move mesh extraction/sample helpers to `dccvt/mesh_ops.py` and update imports.
5. Move training loop + `Voroloss_opt` to `dccvt/training.py` and update imports.
6. Move alpha-shape SDF to `dccvt/alpha_shape.py` and update imports.
7. Move process/skip logic to `dccvt/runner.py` and reduce `pipeline.py` to re-exports (or delete `pipeline.py` if unused).

### Guardrails
- Do not change function bodies, control flow, or numerical operations.
- Only adjust imports and module boundaries.
- Keep CLI behavior and output naming identical.
