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
