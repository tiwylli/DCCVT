"""Neural DCCVT components based on PoNQ's dense SDF-grid formulation."""

from dccvt.neural.grid import (
    HYBRID_DIRECT_CHANNELS,
    build_hybrid_input_channels,
    build_hybrid_input_channels_np,
    default_near_surface_threshold,
    make_canonical_sites,
    make_cell_lower_corners,
    make_coord_grid,
    make_gt_activity_mask_np,
    make_near_surface_mask_np,
    point_udf_grid,
    trilinear_interpolate_sdf,
    validate_hybrid_channel_names,
)
from dccvt.neural.hybrid_initial import HybridInitialHotSpotConfig, build_initial_hotspot_field
from dccvt.neural.iter_refine import (
    DCCVTHybridIterRefineNet,
    HybridIterRefineConfig,
    HybridIterRefineDataset,
    build_hotspot_near_surface_initialization,
    run_iterative_refinement,
    select_procedural_refinement_parents,
)
from dccvt.neural.models import DCCVTHybridDirectNet, DCCVTPoNQNet, HybridDirectConfig
from dccvt.neural.sparse_refine import HybridSparseRefineConfig, build_sparse_base_field, refine_sparse_field

__all__ = [
    "DCCVTHybridDirectNet",
    "DCCVTHybridIterRefineNet",
    "DCCVTPoNQNet",
    "HYBRID_DIRECT_CHANNELS",
    "HybridDirectConfig",
    "HybridInitialHotSpotConfig",
    "HybridIterRefineConfig",
    "HybridIterRefineDataset",
    "HybridSparseRefineConfig",
    "build_hybrid_input_channels",
    "build_hybrid_input_channels_np",
    "build_hotspot_near_surface_initialization",
    "build_initial_hotspot_field",
    "build_sparse_base_field",
    "default_near_surface_threshold",
    "make_canonical_sites",
    "make_cell_lower_corners",
    "make_coord_grid",
    "make_gt_activity_mask_np",
    "make_near_surface_mask_np",
    "point_udf_grid",
    "refine_sparse_field",
    "run_iterative_refinement",
    "select_procedural_refinement_parents",
    "trilinear_interpolate_sdf",
    "validate_hybrid_channel_names",
]
