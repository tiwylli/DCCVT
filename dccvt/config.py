"""Configuration defaults and output path helpers for DCCVT experiments."""

import datetime
import os

# Generate a timestamp string for unique output folders
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# timestamp = "alphashape"
# timestamp = "ALL_CASE_DCCVT"
# timestamp = "FIGURE_CASE_441708"
# timestamp = "FIGURE_CASE_TEASER"
# timestamp = "MT_UNCONV_MAGA"
# timestamp = "DCCVT_UNCONV_MAGA"
# timestamp = "U_150K"
# timestamp = "video_150k"
# timestamp = "MT_150k"
# timestamp = "ABLATION_UNCONV_SDF_NU"
# timestamp = "ROBUS_HYBRID_BARY_INTERPOL"
# timestamp = "Ablation_64764"

# Default parameters for the DCCVT experiments
ROOT_DIR = "/home/wc1172/dev/DCCVT"
# User beltegeuse:
if os.environ.get("USER", "") == "beltegeuse":
    ROOT_DIR = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments"

DEFAULTS = {
    "output": f"{ROOT_DIR}/outputs/{timestamp}/",
    "mesh": f"{ROOT_DIR}/mesh/thingi32/",  # "mesh": f"{ROOT_DIR}/mesh/thingi32_150k/",
    "trained_HotSpot": f"{ROOT_DIR}/hotspots_model/",
    "input_dims": 3,
    "num_iterations": 1000,
    "num_centroids": 16,  # ** input_dims
    "sample_near": 0,  # # ** input_dims
    "target_size": 32,  # 32 # ** input_dims
    "clip": False,
    "grad_interpol": "robust",  # , hybrid, barycentric",  # False
    "marching_tetrahedra": False,  # True
    "true_cvt": False,  # True
    "extract_optim": False,  # True
    "no_mp": False,  # True
    "ups_extraction": False,
    "build_mesh": False,
    "video": False,
    "sdf_type": "hotspot",  # "hotspot", "sphere", "complex_alpha"
    "w_cvt": 0,
    "w_sdfsmooth": 0,
    "w_voroloss": 0,  # 1000
    "w_chamfer": 0,  # 1000
    "w_vertex_sdf_interpolation": 0,
    "w_mt": 0,  # 1000
    "w_mc": 0,  # 1000
    # "w_bpa": 0,  # 1000
    "upsampling": 0,  # 0
    "ups_method": "tet_frame",  # "tet_random", "random" "tet_frame_remove_parent"
    "score": "conservative",  # "legacy" "density", "cosine", "conservative"
    "lr_sites": 0.0005,
    "mesh_ids": [  # 64764],
        # "252119",
        # "313444",  # lucky cat
        # "316358",
        # "354371",
        # # "398259", this mesh destroys our results
        # "441708",  # bunny
        # "44234",
        # "47984",
        # "527631",
        # "53159",
        # "58168",
        # "64444",
        "64764",  # gargoyle
        # "68380",
        # "68381",
        # "72870",
        # "72960",
        # "73075",
        # "75496",
        # "75655",
        # "75656",
        # "75662",
        # "75665",
        # "76277",
        # "77245",
        # "78671",
        # "79241",
        # "90889",
        # "92763",
        # "92880",
        # "95444",
        # "96481",
    ],
}


def update_timestamp(new_timestamp: str) -> None:
    """Update the global timestamp and keep DEFAULTS["output"] consistent."""
    global timestamp
    timestamp = new_timestamp
    DEFAULTS["output"] = f"{ROOT_DIR}/outputs/{timestamp}/"
