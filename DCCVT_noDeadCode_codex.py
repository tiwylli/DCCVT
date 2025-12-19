# this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
import argparse
import datetime
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from time import time

import gudhi
import kaolin
import numpy as np
import polyscope as ps
import pygdel3d
import torch
import tqdm as tqdm
import trimesh
from numba import njit, prange
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from torch import nn

sys.path.append("3rdparty/HotSpot")
from dataset import shape_3d
import models.Net as Net


# cuda devices
device = torch.device("cuda:0")
print("Using device: ", torch.cuda.get_device_name(device))
# Improve reproducibility
torch.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)

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


class _SafeDict(dict):
    def __missing__(self, key):
        # leave unknown placeholders intact, e.g. "{mesh_id}"
        return "{" + key + "}"


def load_arg_lists_from_file(path: str, defaults: dict, mesh_ids=None):
    if mesh_ids is None:
        mesh_ids = list(defaults.get("mesh_ids", []))

    arg_lists = []
    active_mesh_ids = mesh_ids
    buf = ""

    def process_buffer(s: str):
        nonlocal arg_lists, active_mesh_ids
        s = s.strip()
        if not s:
            return

        # handle directives (not part of continued blocks)
        if s.lower().startswith("@mesh_ids"):
            _, rhs = s.split(":", 1)
            items = re.split(r"[,\s]+", rhs.strip())
            active_mesh_ids = [it for it in items if it]
            return

        # 1) safely format known {placeholders} from defaults
        templated = s.format_map(_SafeDict(**defaults))
        # 2) expand env vars like $HOME
        templated = os.path.expandvars(templated)

        # 3) fan out over {mesh_id} if present
        if "{mesh_id}" in templated:
            for mid in active_mesh_ids:
                filled = templated.replace("{mesh_id}", str(mid))
                arg_lists.append(shlex.split(filled))
        else:
            arg_lists.append(shlex.split(templated))

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not buf and (not stripped or stripped.startswith("#")):
                continue

            # continuation if there's an odd number of trailing backslashes
            m = re.search(r"(\\+)$", line)
            trailing_bs = len(m.group(1)) if m else 0
            is_cont = trailing_bs % 2 == 1

            if is_cont:
                line = line[:-1]  # drop exactly one "\" for continuation
                buf += line
                continue
            else:
                buf += line
                process_buffer(buf)  # complete logical line
                buf = ""

    if buf.strip():
        process_buffer(buf)

    return arg_lists


def _add_bool_arg(parser, flag, default, help_text):
    parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_text)


def define_options_parser(arg_list=None):
    parser = argparse.ArgumentParser(description="DCCVT experiments")
    parser.add_argument("--input_dims", type=int, default=DEFAULTS["input_dims"], help="Dimensionality of the input")
    parser.add_argument("--output", type=str, default=DEFAULTS["output"], help="Output directory")
    parser.add_argument("--mesh", type=str, default=DEFAULTS["mesh"], help="Mesh directory")
    parser.add_argument(
        "--trained_HotSpot", type=str, default=DEFAULTS["trained_HotSpot"], help="Trained HotSpot model directory"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=DEFAULTS["num_iterations"], help="Number of iterations for optimization"
    )
    parser.add_argument("--num_centroids", type=int, default=DEFAULTS["num_centroids"], help="Number of centroids")
    parser.add_argument("--sample_near", type=int, default=DEFAULTS["sample_near"], help="Samples drawn near each site")
    parser.add_argument("--target_size", type=int, default=DEFAULTS["target_size"], help="Target size for sampling")
    _add_bool_arg(parser, "--clip", DEFAULTS["clip"], "Enable/disable clipping")
    parser.add_argument(
        "--grad_interpol",
        type=str,
        default=DEFAULTS["grad_interpol"],
        help="Gradient interpolation method: robust, hybrid, barycentric",
    )
    _add_bool_arg(
        parser, "--marching_tetrahedra", DEFAULTS["marching_tetrahedra"], "Enable/disable marching_tetrahedra"
    )
    _add_bool_arg(parser, "--true_cvt", DEFAULTS["true_cvt"], "Enable/disable true CVT loss")
    _add_bool_arg(parser, "--extract_optim", DEFAULTS["extract_optim"], "Enable/disable extraction optimization")
    parser.add_argument(
        "--sdf_type",
        type=str,
        default=DEFAULTS["sdf_type"],
        help="SDF type: hotspot, sphere, complex_alpha",
    )
    _add_bool_arg(parser, "--no_mp", DEFAULTS["no_mp"], "Enable/disable multiprocessing")
    _add_bool_arg(parser, "--ups_extraction", DEFAULTS["ups_extraction"], "Enable/disable upsampling extraction")
    _add_bool_arg(parser, "--build_mesh", False, "Enable/disable build mesh")
    _add_bool_arg(parser, "--video", False, "Enable/disable video output")
    parser.add_argument("--w_cvt", type=float, default=DEFAULTS["w_cvt"], help="Weight for CVT regularization")
    parser.add_argument(
        "--w_vertex_sdf_interpolation",
        type=float,
        default=DEFAULTS["w_vertex_sdf_interpolation"],
        help="Weight for vertex SDF interpolation",
    )
    parser.add_argument("--w_sdfsmooth", type=float, default=DEFAULTS["w_sdfsmooth"], help="Weight for SDF smoothing")
    parser.add_argument("--w_voroloss", type=float, default=DEFAULTS["w_voroloss"], help="Weight for Voronoi loss")
    parser.add_argument(
        "--w_chamfer", type=float, default=DEFAULTS["w_chamfer"], help="Weight for Chamfer distance on points"
    )
    # parser.add_argument("--w_bpa", type=float, default=DEFAULTS.get("w_bpa", 0), help="flag to use BPA instead of DCCVT")
    parser.add_argument("--w_mc", type=float, default=DEFAULTS["w_mc"], help="Weight for MC loss")
    parser.add_argument("--w_mt", type=float, default=DEFAULTS["w_mt"], help="Weight for MT loss")
    parser.add_argument("--upsampling", type=int, default=DEFAULTS["upsampling"], help="Upsampling factor")
    parser.add_argument(
        "--ups_method",
        type=str,
        default=DEFAULTS["ups_method"],
        help="Upsampling method either tet_frame or tet_random or random",
    )
    parser.add_argument("--lr_sites", type=float, default=DEFAULTS["lr_sites"], help="Learning rate for sites")
    parser.add_argument(
        "--save_path", type=str, default=None, help="(optional) full save path; if omitted, computed from other flags"
    )
    parser.add_argument(
        "--score",
        type=str,
        default=DEFAULTS["score"],
        help="Score computation [legacy, density, sqrt_curvature, cosine]",
    )
    return parser.parse_args(arg_list)


def resolve_sdf_values(model, sites, *, verbose=False):
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        if verbose:
            print("Using SDFGrid")
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        if verbose:
            print("Using Tensor")
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        if verbose:
            print("Using nn.Module / callable model")
        sdf_values = model(sites).detach()
    return sdf_values.squeeze()


def compute_d3d_simplices(sites, marching_tetrahedra):
    sites_np = sites.detach().cpu().numpy()
    if marching_tetrahedra:
        return Delaunay(sites_np).simplices
    d3dsimplices, _ = pygdel3d.triangulate(sites_np)
    return np.array(d3dsimplices)


def build_dccvt_obj_path(args, state, variant):
    prefix = "marching_tetrahedra" if args.marching_tetrahedra else "DCCVT"
    return (
        f"{args.save_path}/{prefix}_{args.upsampling}_{state}_{variant}_"
        f"cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    )


def build_voromesh_obj_path(args, state):
    return (
        f"{args.save_path}/voromesh_{args.num_centroids}_{state}_DCCVT_"
        f"cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    )


def load_model(mesh, target, trained_HotSpot):
    # LOAD MODEL WITH HOTSPOT
    loss_type = "igr_w_heat"
    loss_weights = [350, 0, 0, 1, 0, 0, 20]
    train_set = shape_3d.ReconDataset(
        file_path=mesh + ".ply",
        n_points=target * target * 150,  # 15000, #args.n_points,
        n_samples=10001,  # args.n_iterations,
        grid_res=256,  # args.grid_res,
        grid_range=1.1,  # args.grid_range,
        sample_type="uniform_central_gaussian",  # args.nonmnfld_sample_type,
        sampling_std=0.5,  # args.nonmnfld_sample_std,
        n_random_samples=7500,  # args.n_random_samples,
        resample=True,
        compute_sal_dist_gt=(True if "sal" in loss_type and loss_weights[5] > 0 else False),
        scale_method="mean",  # "mean" #args.pcd_scale_method,
    )
    model = Net.Network(
        latent_size=0,  # args.latent_size,
        in_dim=3,
        decoder_hidden_dim=128,  # args.decoder_hidden_dim,
        nl="sine",  # args.nl,
        encoder_type="none",  # args.encoder_type,
        decoder_n_hidden_layers=5,  # args.decoder_n_hidden_layers,
        neuron_type="quadratic",  # args.neuron_type,
        init_type="mfgi",  # args.init_type,
        sphere_init_params=[1.6, 0.1],  # args.sphere_init_params,
        n_repeat_period=30,  # args.n_repeat_period,
    )
    model.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    test_data = next(iter(test_dataloader))
    mnfld_points = test_data["mnfld_points"].to(device)
    mnfld_points.requires_grad_()
    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(trained_HotSpot, weights_only=True, map_location=map_location))
    return model, mnfld_points


def init_sites(mnfld_points, num_centroids, sample_near, input_dims):
    noise_scale = 0.005
    domain_limit = 1
    if input_dims == 2:
        # throw error not yet implemented
        raise NotImplementedError("2D not yet implemented")
    elif input_dims == 3:
        x = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        y = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        z = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        meshgrid = torch.meshgrid(x, y, z)
        meshgrid = torch.stack(meshgrid, dim=3).view(-1, 3)
        meshgrid += torch.randn_like(meshgrid) * noise_scale

    sites = meshgrid.to(device, dtype=torch.float32).requires_grad_(True)
    # add mnfld points with random noise to sites
    N = mnfld_points.squeeze(0).shape[0]
    if sample_near > 0:
        # num_samples = sample_near**input_dims - num_centroids**input_dims
        num_samples = sample_near
        idx = torch.randint(0, N, (num_samples,))
        sampled = mnfld_points.squeeze(0)[idx]
        perturbed = sampled + (torch.rand_like(sampled) - 0.5) * noise_scale
        sites = torch.cat((sites, perturbed), dim=0)
    # make sites a leaf tensor
    sites = sites.detach().requires_grad_()
    return sites


def init_sdf(model, sites):
    sdf_values = model(sites)
    sdf_values = sdf_values.detach().squeeze(-1).requires_grad_()
    return sdf_values


class Voroloss_opt(nn.Module):
    def __init__(self):
        super(Voroloss_opt, self).__init__()
        self.knn = 16

    def __call__(self, points, spoints):
        """points, self.points"""
        # WARNING: fecthing for knn
        with torch.no_grad():
            indices = knn_points(points[None, :], spoints[None, :], K=self.knn).idx[0]
        point_to_voronoi_center = points - spoints[indices[:, 0]]
        voronoi_edge = spoints[indices[:, 1:]] - spoints[indices[:, 0, None]]
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, None, :] * voronoi_edge).sum(-1) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(1)[0]


def get_clipped_mesh_numba(
    sites,
    model,
    d3dsimplices,
    clip=True,
    sites_sdf=None,
    build_mesh=False,
    quaternion_slerp=False,
    grad_interpol="robust",
    no_mp=False,
):
    """
    sites:           (N,3) torch tensor (requires_grad)
    model:           SDF model: sites -> (N,1) tensor of signed distances
    d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
    """
    device = sites.device
    vertices_sdf = None
    vertices_sdf_grad = None
    sdf_verts = None
    grads = None
    proj_vertices = None
    tet_probs = None
    if d3dsimplices is None:
        print("Computing Delaunay simplices...")
        sites_np = sites.detach().cpu().numpy()
        d3dsimplices, _ = pygdel3d.triangulate(sites_np)
        print("Number of Delaunay simplices:", len(d3dsimplices))
        print("Delaunay simplices shape:", d3dsimplices)
        print("Max vertex index in simplices:", d3dsimplices.max())
        print("Min vertex index in simplices:", d3dsimplices.min())
        print("Site index range:", sites_np.shape[0])

    d3d = torch.tensor(d3dsimplices).to(device).detach()  # (M,4)

    if build_mesh:
        # print("-> tracing mesh")
        all_vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)
        faces = get_faces(d3dsimplices, sites, all_vor_vertices, model, sites_sdf)  # (R0, List of simplices)
        # Compact the vertex list
        used = {idx for face in faces for idx in face}
        old2new = {old: new for new, old in enumerate(sorted(used))}
        new_vertices = all_vor_vertices[sorted(used)]
        new_faces = [[old2new[i] for i in face] for face in faces]
        if not clip:
            # print("-> not clipping")
            return new_vertices, new_faces, None, None, None
        else:
            # print("-> clipping")
            vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(sites, sites_sdf, d3d)  # (M,3)

            if grad_interpol == "barycentric":
                # Use barycentric weights for interpolation
                vertices_sdf_grad, bary_w = interpolate_sdf_grad_of_vertices(
                    all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
                )
                sdf_verts = vertices_sdf[sorted(used)]
                grads = vertices_sdf_grad[sorted(used)]
                proj_vertices = newton_step_clipping(grads, sdf_verts, new_vertices)
            elif grad_interpol == "robust":
                proj_vertices, tet_probs = tet_plane_clipping(
                    d3d[sorted(used)], sites, sites_sdf, sites_sdf_grad, new_vertices
                )
                # proj_vertices = tet_grads_clipping(
                #     new_vertices, vertices_sdf[sorted(used)], tets_sdf_grads[sorted(used)]
                # )
            elif grad_interpol == "hybrid":
                # print("-> using barycentric weights for interpolation")
                # Use barycentric weights for interpolation
                vertices_sdf_grad, bary_w = interpolate_sdf_grad_of_vertices(
                    all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
                )
                sdf_verts = vertices_sdf[sorted(used)]
                grads = vertices_sdf_grad[sorted(used)]
                proj_vertices = newton_step_clipping(grads, sdf_verts, new_vertices)

                tpc_proj_v, tet_probs = tet_plane_clipping(
                    d3d[sorted(used)], sites, sites_sdf, sites_sdf_grad, new_vertices
                )
                # replace proj_vertices with tpc_proj_v where bary_w has negative component
                neg_row_mask = (bary_w[sorted(used)] < 0).any(dim=1)  # (K,)
                # print("bary_w", neg_row_mask.shape, "num bad:", neg_row_mask.sum().item())
                proj_vertices[neg_row_mask] = tpc_proj_v[neg_row_mask]

            return proj_vertices, new_faces, sites_sdf_grad, tets_sdf_grads, W
    else:
        # print("-> not tracing mesh")
        all_vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)
        vertices_to_compute, bisectors_to_compute, used_tet = compute_zero_crossing_vertices_3d(
            sites, None, None, d3dsimplices, sites_sdf
        )
        vertices = compute_vertices_3d_vectorized(sites, vertices_to_compute)
        bisectors = compute_all_bisectors_vectorized(sites, bisectors_to_compute)
        # points = torch.cat((vertices, bisectors), 0)
        if not clip:
            # print("-> not clipping")
            return vertices, None, None, None, None
        else:
            # print("-> clipping")
            vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(sites, sites_sdf, d3d)
            if grad_interpol == "barycentric":
                # print("-> using barycentric weights for interpolation")
                # Use barycentric weights for interpolation
                vertices_sdf_grad, bary_w = interpolate_sdf_grad_of_vertices(
                    all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
                )
                sdf_verts = vertices_sdf[used_tet]
                grads = vertices_sdf_grad[used_tet]
                proj_vertices = newton_step_clipping(grads, sdf_verts, vertices)

                # tpc_proj_v, tet_probs = tet_plane_clipping(d3d[used_tet], sites, sites_sdf, sites_sdf_grad, vertices)
                # # replace proj_vertices with tpc_proj_v where bary_w has negative component
                # neg_row_mask = (bary_w[used_tet] < 0).any(dim=1)  # (K,)
                # print("bary_w", neg_row_mask.shape, "num bad:", neg_row_mask.sum().item())
                # proj_vertices[neg_row_mask] = tpc_proj_v[neg_row_mask]
            elif grad_interpol == "robust":
                proj_vertices, tet_probs = tet_plane_clipping(d3d[used_tet], sites, sites_sdf, sites_sdf_grad, vertices)
                # proj_vertices = tet_grads_clipping(vertices, vertices_sdf[used_tet], tets_sdf_grads[used_tet])
            elif grad_interpol == "hybrid":
                # print("-> using barycentric weights for interpolation")
                # Use barycentric weights for interpolation
                vertices_sdf_grad, bary_w = interpolate_sdf_grad_of_vertices(
                    all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
                )
                sdf_verts = vertices_sdf[used_tet]
                grads = vertices_sdf_grad[used_tet]
                proj_vertices = newton_step_clipping(grads, sdf_verts, vertices)

                tpc_proj_v, tet_probs = tet_plane_clipping(d3d[used_tet], sites, sites_sdf, sites_sdf_grad, vertices)
                # replace proj_vertices with tpc_proj_v where bary_w has negative component
                neg_row_mask = (bary_w[used_tet] < 0).any(dim=1)  # (K,)
                # print("bary_w", neg_row_mask.shape, "num bad:", neg_row_mask.sum().item())
                proj_vertices[neg_row_mask] = tpc_proj_v[neg_row_mask]

            # in paper this will be considered a regularisation
            if not no_mp:
                bisectors_sdf = (sites_sdf[bisectors_to_compute[:, 0]] + sites_sdf[bisectors_to_compute[:, 1]]) / 2
                bisectors_sdf_grad = (
                    sites_sdf_grad[bisectors_to_compute[:, 0]] + sites_sdf_grad[bisectors_to_compute[:, 1]]
                ) / 2

                proj_bisectors = newton_step_clipping(bisectors_sdf_grad, bisectors_sdf, bisectors)  # (M,3)

                proj_points = torch.cat((proj_vertices, proj_bisectors), 0)
            else:
                proj_points = proj_vertices

            vert_for_clipped_cvt = all_vor_vertices
            vert_for_clipped_cvt[used_tet] = proj_vertices
            # proj_points = proj_vertices
            return proj_points, vert_for_clipped_cvt, sites_sdf_grad, tet_probs, W


def compute_zero_crossing_vertices_3d(sites, vor=None, tri=None, simplices=None, model=None):
    """
    Computes the indices of the sites composing vertices where neighboring sites have opposite or zero SDF values.

    Args:
        sites (torch.Tensor): (N, D) tensor of site positions.
        model (callable): Function or neural network that computes SDF values.

    Returns:
        zero_crossing_vertices_index (list of triplets): List of sites indices (si, sj, sk) where atleast 2 sites have opposing SDF signs.
    """
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    # model might be a [sites, 1] tensor
    elif isinstance(model, torch.Tensor):
        sdf_values = model
    else:
        sdf_values = model(sites).detach()  # Assuming model outputs (N, 1) or (N,) tensor

    if tri is not None:
        all_tetrahedra = torch.tensor(np.array(tri.simplices), device=device)
    else:
        all_tetrahedra = torch.tensor(np.array(simplices), device=device)

    if vor is not None:
        neighbors = torch.tensor(np.array(vor.ridge_points), device=device)
    else:
        zero_crossing_pairs = compute_zero_crossing_sites_pairs(all_tetrahedra, sdf_values)

    # Check if vertices has a pair of zero crossing sites
    sdf_0 = sdf_values[all_tetrahedra[:, 0]]  # First site in each pair
    sdf_1 = sdf_values[all_tetrahedra[:, 1]]  # Second site in each pair
    sdf_2 = sdf_values[all_tetrahedra[:, 2]]  # Third site in each pair
    sdf_3 = sdf_values[all_tetrahedra[:, 3]]  # Fourth site in each pair
    mask_zero_crossing_faces = (
        (sdf_0 * sdf_1 <= 0).squeeze()
        | (sdf_0 * sdf_2 <= 0).squeeze()
        | (sdf_0 * sdf_3 <= 0).squeeze()
        | (sdf_1 * sdf_2 <= 0).squeeze()
        | (sdf_1 * sdf_3 <= 0).squeeze()
        | (sdf_2 * sdf_3 <= 0).squeeze()
    )
    zero_crossing_sites_making_verts = all_tetrahedra[mask_zero_crossing_faces]

    return (
        zero_crossing_sites_making_verts,
        zero_crossing_pairs,
        mask_zero_crossing_faces,
    )


def compute_zero_crossing_sites_pairs(all_tetrahedra, sdf_values):
    tetra_edges = torch.cat(
        [
            all_tetrahedra[:, [0, 1]],
            all_tetrahedra[:, [1, 2]],
            all_tetrahedra[:, [2, 3]],
            all_tetrahedra[:, [3, 0]],
            all_tetrahedra[:, [0, 2]],
            all_tetrahedra[:, [1, 3]],
        ],
        dim=0,
    ).to(device)
    # Sort each edge to ensure uniqueness (because (a, b) and (b, a) are the same)
    tetra_edges, _ = torch.sort(tetra_edges, dim=1)
    # neighbors = torch.unique(tetra_edges, dim=0)
    neighbors = tetra_edges

    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
    zero_crossing_pairs = neighbors[mask_zero_crossing_sites]

    return zero_crossing_pairs


def compute_all_bisectors_vectorized(sites, bisectors_to_compute):
    """
    Computes the bisector points for given pairs of sites in 3D.

    Args:
        sites (torch.Tensor): (N, 3) tensor of site positions.
        bisectors_to_compute (torch.Tensor): (M, 2) tensor of index pairs.

    Returns:
        torch.Tensor: (M, 3) tensor of computed bisector points.
    """
    # Extract all site pairs at once
    si = sites[bisectors_to_compute[:, 0]]  # Shape: (M, N)
    sj = sites[bisectors_to_compute[:, 1]]  # Shape: (M, N)

    # Compute bisectors in a single vectorized operation
    bisectors = (si + sj) / 2  # Shape: (M, N)

    return bisectors


def sdf_space_grad_pytorch_diego_sites_tets(sites, sdf, tets):
    """
    Compute the spatial gradient of the SDF at each site (vertex) and each tetrahedron.

    Args:
        sites: (N, 3) tensor of 3D vertex coordinates.
        sdf: (N,) tensor of SDF values at each vertex.
        tets: (M, 4) tensor of tetrahedral indices.

    Returns:
        grad_sdf: (N, 3) estimated SDF gradient at each site (vertex-wise average of surrounding tet gradients)
        grad_sdf_tet: (M, 3) estimated SDF gradient inside each tetrahedron (constant per tet)
    """
    M = tets.shape[0]
    tet_ids = tets
    a, b, c, d = (
        sites[tet_ids[:, 0]],
        sites[tet_ids[:, 1]],
        sites[tet_ids[:, 2]],
        sites[tet_ids[:, 3]],
    )
    sdf_a, sdf_b, sdf_c, sdf_d = (
        sdf[tet_ids[:, 0]],
        sdf[tet_ids[:, 1]],
        sdf[tet_ids[:, 2]],
        sdf[tet_ids[:, 3]],
    )

    center = (a + b + c + d) / 4
    sdf_center = (sdf_a + sdf_b + sdf_c + sdf_d) / 4

    volume = volume_tetrahedron(a, b, c, d)  # (M,)

    X = torch.stack([a, b, c, d], dim=1)  # (M, 4, 3)
    dX = X - center[:, None, :]  # (M, 4, 3)
    dX_T = dX.transpose(1, 2)  # (M, 3, 4)

    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    Ginv = torch.linalg.pinv(G)  # (M, 3, 3)

    W = torch.einsum("mij,mnj->mni", Ginv, dX)  # (M, 4, 3)

    sdf_stack = torch.stack([sdf_a, sdf_b, sdf_c, sdf_d], dim=1)  # (M, 4)
    sdf_diff = sdf_stack - sdf_center[:, None]  # (M, 4)

    grad_sdf_tet = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)

    for i in range(4):
        ids = tet_ids[:, i]  # (M,)
        grad_contrib = grad_sdf_tet * volume[:, None]  # (M, 3)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    grad_sdf /= weights_tot.clamp(min=1e-8).unsqueeze(1)

    return grad_sdf, grad_sdf_tet, W


def volume_tetrahedron(a, b, c, d):
    ad = a - d
    bd = b - d
    cd = c - d
    n = torch.linalg.cross(bd, cd, dim=-1)
    return torch.abs((ad * n).sum(dim=-1)) / 6.0


def sdf_space_grad_pytorch_diego(sites, sdf, tets):
    # sites: (N, 3)
    # sdf: (N,)
    # tets: (M, 4)

    M = tets.shape[0]
    tet_ids = tets
    a, b, c, d = (
        sites[tet_ids[:, 0]],
        sites[tet_ids[:, 1]],
        sites[tet_ids[:, 2]],
        sites[tet_ids[:, 3]],
    )
    sdf_a, sdf_b, sdf_c, sdf_d = (
        sdf[tet_ids[:, 0]],
        sdf[tet_ids[:, 1]],
        sdf[tet_ids[:, 2]],
        sdf[tet_ids[:, 3]],
    )

    center = (a + b + c + d) / 4
    sdf_center = (sdf_a + sdf_b + sdf_c + sdf_d) / 4

    volume = volume_tetrahedron(a, b, c, d)

    # Build dX: (M, 4, 3)
    X = torch.stack([a, b, c, d], dim=1)  # (M, 4, 3)
    dX = X - center[:, None, :]

    dX_T = dX.transpose(1, 2)  # (M, 3, 4)
    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    # Inverse G: (M, 3, 3)
    Ginv = torch.linalg.pinv(G)  # stable pseudo-inverse for singular cases
    # Ginv = torch.linalg.inv(G)  # faster, uses LU

    # Weights: Ginv @ dX^T -> (M, 3, 4)
    # W = torch.einsum('mij,mkj->mki', Ginv, dX)  # (M, 4, 3)
    W = torch.einsum("mij,mnj->mni", Ginv, dX)

    sdf_stack = torch.stack([sdf_a, sdf_b, sdf_c, sdf_d], dim=1)  # (M, 4)
    sdf_diff = sdf_stack - sdf_center[:, None]

    elem = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)

    for i in range(4):
        ids = tet_ids[:, i]
        grad_contrib = elem * volume[:, None]  # (M, 3)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    # Avoid division by zero (e.g., isolated vertices)
    weights_tot_clamped = weights_tot.clamp(min=1e-8).unsqueeze(1)  # (N, 1)
    grad_sdf /= weights_tot_clamped

    return grad_sdf


def interpolate_sdf_of_vertices(
    vertices: torch.Tensor,  # (M, 3)  positions of Voronoi vertices (e.g. circumcenters)
    tets: torch.LongTensor,  # (M, 4)  indices of the 4 sites per tetrahedron
    sites: torch.Tensor,  # (N, 3)  coordinates of the sites
    sdf: torch.Tensor,  # (N,)    scalar field value at each site
) -> torch.Tensor:
    """
    Interpolates the SDF at Voronoi vertices (e.g., circumcenters) using barycentric coordinates,
    without calling torch.linalg.solve.

    Returns
    -------
    phi_v : (M,) tensor of interpolated SDF values at Voronoi vertices
    """

    v_pos = sites[tets]  # (M, 4, 3)
    v_phi = sdf[tets]  # (M, 4)

    x0 = v_pos[:, 0]  # (M, 3)
    x1 = v_pos[:, 1]
    x2 = v_pos[:, 2]
    x3 = v_pos[:, 3]

    # Build D = [x1 - x0 | x2 - x0 | x3 - x0]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0

    D = torch.stack([e1, e2, e3], dim=2)  # (M,3,3)

    c1 = torch.cross(e2, e3, dim=1)  # cofactor for col 0
    c2 = torch.cross(e3, e1, dim=1)  # cofactor for col 1
    c3 = torch.cross(e1, e2, dim=1)  # cofactor for col 2

    adj_D = torch.stack([c1, c2, c3], dim=2)  # (M, 3, 3)

    # Determinant of D
    det_D = (e1 * c1).sum(dim=1, keepdim=True)  # (M, 1)

    # Right-hand side: x - x0
    rhs = vertices - x0  # (M, 3)

    # Inverse: D⁻¹ @ rhs = adj(D)^T @ rhs / det(D)
    w123 = torch.bmm(adj_D.transpose(1, 2), rhs.unsqueeze(-1)).squeeze(-1) / (det_D + 1e-12)  # (M, 3)
    w0 = 1.0 - w123.sum(dim=1, keepdim=True)  # (M, 1)
    W = torch.cat([w0, w123], dim=1)  # (M, 4)

    # Interpolate SDF
    phi_v = (W * v_phi).sum(dim=1)  # (M,)

    return phi_v


def get_faces(d3dsimplices, sites, vor_vertices, model=None, sites_sdf=None):
    with torch.no_grad():
        d3d = torch.tensor(d3dsimplices).to(device).detach()  # (M,4)
        # Generate all edges of each simplex
        #    torch.combinations gives the 6 index‐pairs within a 4‐long row
        comb = torch.combinations(torch.arange(d3d.shape[1], device=device), r=2)  # (6,2)
        # print("comb", comb.shape)
        edges = d3d[:, comb]  # (M,6,2)
        edges = edges.reshape(-1, 2)  # (M*6,2)
        edges, _ = torch.sort(edges, dim=1)  # sort each row so (a,b) == (b,a)

        # Unique ridges across all simplices
        # ridges, inverse = torch.unique(edges, dim=0, return_inverse=True) # (R,2)

        ridges = edges  # torch.unique(edges, dim=0, return_inverse=False) # (R,2)

        del comb, edges
        torch.cuda.empty_cache()

        # Evaluate SDF at each site
        if model is not None:
            sdf = model(sites).detach().view(-1)  # (N,)
        else:
            sdf = sites_sdf  # (N,)

        sdf_i = sdf[ridges[:, 0]]
        sdf_j = sdf[ridges[:, 1]]
        zero_cross = sdf_i * sdf_j <= 0  # (R,)
        # Keep only the zero-crossing ridges
        ridges = ridges[zero_cross]  # (R0,2)
        faces = faces_via_dict(d3dsimplices, ridges.detach().cpu().numpy())  # (R0, List of simplices)

        # Sort faces
        torch.cuda.empty_cache()
        R = len(faces)
        counts = np.array([len(face) for face in faces], dtype=np.int64)
        Kmax = counts.max()
        faces_np = np.full((R, Kmax), -1, dtype=np.int64)

        for i, face in enumerate(faces):
            faces_np[i, : len(face)] = face

        sorted_faces_np = np.full((R, Kmax), -1, dtype=np.int64)

        # print("-> sorting faces")
        batch_sort_numba(vor_vertices.detach().cpu().numpy(), faces_np, counts, sorted_faces_np)
        faces_sorted = [sorted_faces_np[i, : counts[i]].tolist() for i in range(R)]
        return faces_sorted


def faces_via_dict(d3dsimplices, ridges):
    # build dict of (a,b) → list of simplex-indices
    face_dict = defaultdict(list)
    for si, simplex in enumerate(d3dsimplices):
        # all 6 edges of a 4-vertex simplex
        a, b, c, d = simplex
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            key = (u, v) if u < v else (v, u)
            face_dict[key].append(si)

    # face dict creates a dictionnary of all the voronoi vertex that form voronoi faces

    # now for each ridge (a,b) grab its list
    out = []
    for a, b in ridges:
        key = (a, b) if a < b else (b, a)
        lst = face_dict.get(key, [])
        out.append(np.array(lst, dtype=np.int32))

    return np.array(out, dtype=object)


@njit(parallel=True)
def batch_sort_numba(vertices, faces_list, counts, output):
    R, Kmax = faces_list.shape
    for i in prange(R):
        length = counts[i]
        sorted_i = sort_face_loop_numba(vertices, faces_list[i, :length])
        for j in range(length):
            output[i, j] = sorted_i[j]


@njit
def sort_face_loop_numba(vertices, face):
    # face: 1D np.array of ints
    n = face.shape[0]
    # gather points and centroid
    ctr = np.zeros(3, dtype=np.float64)
    for i in range(n):
        ctr += vertices[face[i]]
    ctr /= n

    # make a normal from the first 3 points
    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    normal = _normalize(_compute_normal(a, b, c))

    # reference axis
    ref = vertices[face[0]] - ctr
    dot_nr = normal[0] * ref[0] + normal[1] * ref[1] + normal[2] * ref[2]
    ref = ref - normal * dot_nr
    ref = _normalize(ref)

    # compute all angles
    angs = np.empty(n, dtype=np.float64)
    for i in range(n):
        angs[i] = _angle(face[i], vertices, ctr, normal, ref)

    # now do an insertion‐sort by angle, carrying indices
    sorted_idxs = np.empty(n, dtype=face.dtype)
    sorted_angs = np.empty(n, dtype=np.float64)
    length = 0
    for i in range(n):
        a_i = angs[i]
        idx_i = face[i]
        # find insert position
        j = length
        while j > 0 and sorted_angs[j - 1] > a_i:
            sorted_angs[j] = sorted_angs[j - 1]
            sorted_idxs[j] = sorted_idxs[j - 1]
            j -= 1
        sorted_angs[j] = a_i
        sorted_idxs[j] = idx_i
        length += 1

    return sorted_idxs


@njit
def _compute_normal(a, b, c):
    # cross( b−a, c−a )
    ab = b - a
    ac = c - a
    # cross product
    return np.array(
        (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ),
        dtype=np.float64,
    )


@njit
def _normalize(v):
    norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / (norm + 1e-12)


@njit
def _angle(idx, vertices, ctr, normal, ref):
    p = vertices[idx]
    v = p - ctr
    # project into plane
    dot_nv = normal[0] * v[0] + normal[1] * v[1] + normal[2] * v[2]
    v = v - normal * dot_nv
    # compute angle = atan2(||ref×v||, ref·v)
    cr = np.empty(3, dtype=np.float64)
    cr[0] = ref[1] * v[2] - ref[2] * v[1]
    cr[1] = ref[2] * v[0] - ref[0] * v[2]
    cr[2] = ref[0] * v[1] - ref[1] * v[0]
    num = np.sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])
    den = ref[0] * v[0] + ref[1] * v[1] + ref[2] * v[2]
    ang = np.arctan2(num, den)
    # sign correction
    sign = (normal[0] * cr[0] + normal[1] * cr[1] + normal[2] * cr[2]) < 0
    return 2 * np.pi - ang if sign else ang


def compute_vertices_3d_vectorized(sites, vertices_to_compute):
    """
    Computes the circumcenters of multiple tetrahedra in a vectorized manner.

    Args:
        sites (torch.Tensor): (N, 3) tensor of site positions.
        vertices_to_compute (torch.Tensor): (M, 4) tensor of indices forming tetrahedra.

    Returns:
        torch.Tensor: (M, 3) tensor of computed Voronoi vertices.
    """
    # Extract tetrahedra site coordinates in a batched manner
    tetrahedra = sites[vertices_to_compute]  # Shape: (M, 4, 3)

    # Compute squared norms of each point
    squared_norms = (tetrahedra**2).sum(dim=2, keepdim=True)  # Shape: (M, 4, 1)

    # Construct the 4x4 matrices in batch
    ones_col = torch.ones_like(squared_norms)  # Column of ones for homogeneous coordinates

    A = torch.cat([tetrahedra, ones_col], dim=2)  # Shape: (M, 4, 4)
    Dx = torch.cat([squared_norms, tetrahedra[:, :, 1:], ones_col], dim=2)
    Dy = torch.cat([tetrahedra[:, :, :1], squared_norms, tetrahedra[:, :, 2:], ones_col], dim=2)
    Dz = torch.cat([tetrahedra[:, :, :2], squared_norms, ones_col], dim=2)

    # Compute determinants in batch
    detA = torch.linalg.det(A)  # Shape: (M,)
    detDx = torch.linalg.det(Dx)
    detDy = torch.linalg.det(Dy)  # todo, removed Negative due to orientation
    detDz = torch.linalg.det(Dz)

    # Compute circumcenters
    circumcenters = 0.5 * torch.stack([detDx / detA, detDy / detA, detDz / detA], dim=1)

    return circumcenters  # Shape: (M, 3)


def interpolate_sdf_grad_of_vertices(
    vertices: torch.Tensor,  # (M, 3) positions of Voronoi vertices
    tets: torch.LongTensor,  # (M, 4) indices of sites per tetrahedron
    sites: torch.Tensor,  # (N, 3) coordinates of the sites
    site_grads: torch.Tensor,  # (N, 3) spatial gradients ∇φ at each site
    quaternion_slerp: bool = False,  # use quaternion SLERP for interpolation
) -> torch.Tensor:
    """
    Interpolates the SDF gradient at Voronoi vertices using barycentric coordinates,
    without using torch.linalg.solve.

    Returns
    -------
    grad_v : (M, 3) tensor of interpolated SDF gradients at Voronoi vertices
    """

    v_pos = sites[tets]  # (M, 4, 3)
    v_grad = site_grads[tets]  # (M, 4, 3)

    x0, x1, x2, x3 = v_pos[:, 0], v_pos[:, 1], v_pos[:, 2], v_pos[:, 3]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0

    D = torch.stack([e1, e2, e3], dim=2)  # (M, 3, 3)

    # Cofactors of D
    c1 = torch.cross(e2, e3, dim=1)
    c2 = torch.cross(e3, e1, dim=1)
    c3 = torch.cross(e1, e2, dim=1)
    adj_D = torch.stack([c1, c2, c3], dim=2)  # (M, 3, 3)

    # Determinant
    det_D = (e1 * c1).sum(dim=1, keepdim=True)  # (M, 1)

    # Vector from x0 to each vertex
    rhs = vertices - x0  # (M, 3)

    # Solve D⁻¹ (x - x0)
    w123 = torch.bmm(adj_D.transpose(1, 2), rhs.unsqueeze(-1)).squeeze(-1) / (det_D + 1e-12)  # (M, 3)
    w0 = 1.0 - w123.sum(dim=1, keepdim=True)
    W = torch.cat([w0, w123], dim=1)  # (M, 4)

    # we = torch.abs(W).max(dim=1, keepdim=True)[0]  # (M, 1)

    if quaternion_slerp:
        # Use quaternion SLERP for interpolation
        grad_v = quaternion_slerp_barycentric(v_grad, W)
    else:
        # Weighted sum of gradients
        grad_v = (W.unsqueeze(-1) * v_grad).sum(dim=1)  # (M, 3)

    return grad_v, W


def quaternion_slerp_barycentric(
    v_grad: torch.Tensor,  # (M, 4, 3), SDF gradients at the tet corners
    weights: torch.Tensor,  # (M, 4), barycentric weights
) -> torch.Tensor:
    """
    Perform quaternion-based interpolation of gradients using SLERP.
    Args:
        v_grad: (M, 4, 3) per-tet gradients (assumed unit vectors)
        weights: (M, 4) barycentric weights (sum to 1)
    Returns:
        (M, 3) interpolated unit gradients
    """

    # Normalize gradients (quaternions must be unit length vectors)
    v_grad = torch.nn.functional.normalize(v_grad, dim=-1)  # (M, 4, 3)

    # Convert each gradient to quaternion representation using axis-angle [θ * n] → quaternion
    # We'll assume each 3D unit vector lies on the sphere and can be interpreted as a rotation from a canonical vector
    # We'll pick [1,0,0] as canonical; rotation from it to each gradient gives the rotation quaternion

    # Canonical vector
    canonical = torch.tensor([1.0, 0.0, 0.0], device=v_grad.device).expand(v_grad.shape[0], 1, 3)  # (M,1,3)
    q_rots = []

    for i in range(4):
        v_i = v_grad[:, i]  # (M, 3)
        axis = torch.cross(canonical.squeeze(1), v_i, dim=1)  # (M,3)
        axis = torch.nn.functional.normalize(axis, dim=1)
        dot = (canonical.squeeze(1) * v_i).sum(dim=1, keepdim=True).clamp(-1, 1)  # (M,1)
        angle = torch.acos(dot)  # (M,1)

        half_angle = angle / 2
        q = torch.cat(
            [
                torch.cos(half_angle),  # real part
                axis * torch.sin(half_angle),  # imag part
            ],
            dim=1,
        )  # (M,4)
        q_rots.append(q)

    # Slerp pairwise and combine
    q01 = quaternion_slerp(q_rots[0], q_rots[1], weights[:, 1:2] / (weights[:, 0:1] + weights[:, 1:2] + 1e-12))
    q012 = quaternion_slerp(q01, q_rots[2], weights[:, 2:3] / (weights[:, :3].sum(dim=1, keepdim=True) + 1e-12))
    q_final = quaternion_slerp(q012, q_rots[3], weights[:, 3:4] / (weights.sum(dim=1, keepdim=True) + 1e-12))

    # Convert quaternion to rotation matrix and rotate canonical vector
    R = quaternion_to_matrix(q_final)  # (M, 3, 3)
    grad_interp = torch.matmul(R, canonical.transpose(1, 2)).squeeze(-1)  # (M, 3)

    return grad_interp


def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.
    q1, q2: (..., 4) quaternions (w, x, y, z)
    t: (..., 1) interpolation factor in [0, 1]
    Returns:
        (..., 4) interpolated quaternion
    """
    # Normalize to ensure unit quaternions
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)  # (..., 1)

    # Ensure shortest path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta_0 = torch.acos(dot)  # angle between q1 and q2
    sin_theta_0 = torch.sin(theta_0)

    # Avoid division by 0
    small_angle = sin_theta_0 < 1e-6

    s1 = torch.where(small_angle, 1.0 - t, torch.sin((1.0 - t) * theta_0) / (sin_theta_0 + 1e-12))
    s2 = torch.where(small_angle, t, torch.sin(t * theta_0) / (sin_theta_0 + 1e-12))

    return s1 * q1 + s2 * q2  # (..., 4)


def tet_plane_clipping(
    tets: torch.Tensor,  # (M, 4)
    sites: torch.Tensor,  # (N, 3)
    sdf_values: torch.Tensor,  # (N,)
    sdf_grads: torch.Tensor,  # (N, 3)
    voronoi_vertices: torch.Tensor,  # (M, 3)
) -> torch.Tensor:
    eps = 1e-8
    # Gather tet-specific data
    tet_sites = sites[tets]  # (M, 4, 3)
    tet_sdf = sdf_values[tets]  # (M, 4)
    tet_grads = sdf_grads[tets]  # (M, 4, 3)
    # print(f"tet_sites shape: {tet_sites.shape}, tet_sdf shape: {tet_sdf.shape}, tet_grads shape: {tet_grads.shape}")

    # Project each site to its local zero level-set via Newton step
    grad_norm2 = torch.sqrt((tet_grads**2).sum(dim=-1, keepdim=True) + eps)  # (M, 4, 1)
    site_step_dir = tet_grads / grad_norm2
    steps = tet_sdf.unsqueeze(-1) * site_step_dir  # (M, 4, 3)
    projected_pts = tet_sites - steps  # (M, 4, 3)

    # Fit plane: subtract mean
    centroid = projected_pts.mean(dim=1, keepdim=True)  # (M, 1, 3)
    centered = projected_pts - centroid  # (M, 4, 3)

    # Compute covariance matrix
    cov = torch.einsum("mni,mnj->mij", centered, centered) / 4  # (M, 3, 3)
    # Compute eigenvectors — last one is normal
    _, eigvecs = torch.linalg.eigh(cov)  # (M, 3), (M, 3, 3)
    normal = eigvecs[:, :, 0]  # Smallest eigenvalue → normal direction
    normal = normal / (normal.norm(dim=1, keepdim=True) + eps)  # Normalize

    # Normalize the normal vector
    normal_norm2 = (normal**2).sum(dim=1, keepdim=True) + eps
    vert_step_dir = normal / torch.sqrt(normal_norm2)  # (M, 3)

    # Project voronoi vertices to plane
    v_to_c = voronoi_vertices - centroid.squeeze(1)  # (M, 3)
    normal_dot = (v_to_c * vert_step_dir).sum(dim=1, keepdim=True)  # (M, 1)

    steps_verts = normal_dot * vert_step_dir  # (M, 3)
    projected_verts = voronoi_vertices - steps_verts  # (M, 3)

    return projected_verts, (site_step_dir, steps_verts, tet_sites)


def newton_step_clipping(grads, sdf_verts, new_vertices):
    """
    Perform a single Newton step to clip vertices based on their SDF values and gradients.
    This function is used to refine the positions of Voronoi vertices after computing their SDFs.
    """
    # one Newton step https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    epsilon = 1e-12

    # grad_norm2 = torch.sqrt(((grads + epsilon)**2).sum(dim=1, keepdim=True))    # (M,1)
    grad_norm2 = torch.sqrt((grads**2).sum(dim=1, keepdim=True) + epsilon)  # (M,1)

    step = sdf_verts.unsqueeze(1) * grads / (grad_norm2)  # (M,3)
    proj_vertices = new_vertices - step

    return proj_vertices


def train_DCCVT(sites, sites_sdf, mnfld_points, hotspot_model, args):
    use_chamfer = args.w_chamfer > 0
    use_cvt = args.w_cvt > 0
    use_voroloss = args.w_voroloss > 0
    use_sdfsmooth = args.w_sdfsmooth > 0
    use_vertex_interp = args.w_vertex_sdf_interpolation > 0

    if use_chamfer:
        optimizer = torch.optim.Adam(
            [
                {"params": [sites], "lr": args.lr_sites},
                {"params": [sites_sdf], "lr": args.lr_sites},
            ],
            betas=(0.8, 0.95),
        )
    else:
        optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])
        # SDF at original sites
        sdf_values = resolve_sdf_values(sites_sdf, sites, verbose=False)
        sites_sdf = sdf_values.requires_grad_()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    #
    upsampled = 0.0
    epoch = 0
    t0 = time()
    cvt_loss = 0
    chamfer_loss_mesh = 0
    voroloss_loss = 0
    sdf_loss = 0
    d3dsimplices = None
    sites_sdf_grads = None
    voroloss = Voroloss_opt().to(device)

    for epoch in tqdm.tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        if use_cvt or use_chamfer:
            d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

        if use_chamfer:
            if args.marching_tetrahedra:
                d3dsimplices = torch.tensor(d3dsimplices, device=device)
                marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
                    sites.unsqueeze(0), d3dsimplices, sites_sdf.unsqueeze(0), return_tet_idx=False
                )
                vertices_list, faces_list = marching_tetrehedra_mesh
                v_vect = vertices_list[0]
                # f_or_clipped_v = faces_list[0]
                _, f_or_clipped_v, _, _, _ = get_clipped_mesh_numba(
                    sites,
                    None,
                    d3dsimplices.detach().cpu().numpy(),
                    args.clip,
                    sites_sdf,
                    args.build_mesh,
                    False,
                    args.grad_interpol,
                    args.no_mp,
                )
            else:
                if args.extract_optim:
                    v_vect, f_or_clipped_v = cvt_extraction(sites, sites_sdf, d3dsimplices, False)
                    sites_sdf_grads = None
                else:
                    v_vect, f_or_clipped_v, sites_sdf_grads, tet_probs, W = get_clipped_mesh_numba(
                        sites,
                        None,
                        d3dsimplices,
                        args.clip,
                        sites_sdf,
                        args.build_mesh,
                        False,
                        args.grad_interpol,
                        args.no_mp,
                    )

            if args.build_mesh:
                triangle_faces = [[f[0], f[i], f[i + 1]] for f in f_or_clipped_v for i in range(1, len(f) - 1)]
                triangle_faces = torch.tensor(triangle_faces, device=device)
                hs_p = sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=mnfld_points.shape[0])
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), hs_p.unsqueeze(0))
            else:
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), v_vect.unsqueeze(0))

        if use_voroloss:
            voroloss_loss = voroloss(mnfld_points.squeeze(0), sites).mean()

        if use_cvt:
            if use_voroloss:
                cvt_loss = compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
            else:
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay_volume(sites, None, d3dsimplices)
                if args.true_cvt:
                    cvt_loss = compute_cvt_loss_true(sites, d3dsimplices, f_or_clipped_v)
                else:
                    cvt_loss = compute_cvt_loss_CLIPPED_vertices(sites, d3dsimplices, f_or_clipped_v)

        sites_loss = args.w_cvt / 1 * cvt_loss + args.w_chamfer * chamfer_loss_mesh + args.w_voroloss * voroloss_loss

        if use_sdfsmooth:
            if sites_sdf_grads is None:
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach()
                )
            if epoch % 100 == 0 and epoch <= 500:
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            elif epoch % 100 == 0 and epoch <= 800:
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 2).detach()
                print("Estimated eps_H: ", eps_H)

            # eik_loss = args.w_sdfsmooth / 1000 * lf.tet_sdf_grad_eikonal_loss(sites, tets_sdf_grads, d3dsimplices)
            eik_loss = args.w_sdfsmooth / 10 * discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
            shl = args.w_sdfsmooth * tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, d3dsimplices, eps_H)
            sdf_loss = eik_loss + shl

        if use_vertex_interp:
            steps_verts = tet_probs[1]
            # all_vor_vertices = compute_vertices_3d_vectorized(sites, d3dsimplices)  # (M,3)
            # vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3dsimplices, sites, sites_sdf)
            # _, _, used_tet = compute_zero_crossing_vertices_3d(sites, None, None, d3dsimplices, sites_sdf)

            # projected_verts_sdf = vertices_sdf[used_tet] - steps_verts.norm(dim=1)

            step_len = (steps_verts**2).sum(dim=1).clamp_min(1e-12).sqrt()  # (M,)
            vertex_sdf_loss = args.w_vertex_sdf_interpolation * (step_len).mean()
            sdf_loss = sdf_loss + vertex_sdf_loss

        loss = sites_loss + sdf_loss
        # print(f"Epoch {epoch}: loss = {loss.item()}")
        loss.backward()
        # print("-----------------")

        optimizer.step()
        # scheduler.step()

        if upsampled < args.upsampling and epoch / (args.num_iterations * 0.80) > upsampled / args.upsampling:
            print("sites length BEFORE UPSAMPLING: ", len(sites))
            if len(sites) * 1.08 > args.target_size**3:
                print(
                    "Skipping upsampling, too many sites, sites length: ",
                    len(sites),
                    "target size: ",
                    args.target_size**3,
                )
                upsampled = args.upsampling
                sites = sites.detach().requires_grad_(True)

                if use_chamfer:
                    sites_sdf = sites_sdf.detach().requires_grad_(True)
                    optimizer = torch.optim.Adam(
                        [
                            {"params": [sites], "lr": args.lr_sites},
                            {"params": [sites_sdf], "lr": args.lr_sites},
                        ]
                    )
                    eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 3).detach()
                    print("Estimated eps_H: ", eps_H)
                else:
                    optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])

                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                continue

            if d3dsimplices is None:
                d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

            if sites_sdf_grads is None or sites_sdf_grads.shape[0] != sites_sdf.shape[0]:
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )

            if use_chamfer:
                sites, sites_sdf = upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads, ups_method=args.ups_method, score=args.score
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = sites_sdf.detach().requires_grad_(True)

                d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

                optimizer = torch.optim.Adam(
                    [
                        {"params": [sites], "lr": args.lr_sites},
                        {"params": [sites_sdf], "lr": args.lr_sites},
                    ]
                )
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            else:
                sites_sdf = hotspot_model(sites).squeeze(-1)
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )
                sites, sites_sdf = upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads, ups_method=args.ups_method, score=args.score
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = hotspot_model(sites)
                sites_sdf = sites_sdf.detach().squeeze(-1).requires_grad_()
                optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])

            if args.ups_extraction:
                with torch.no_grad():
                    extract_mesh(sites, sites_sdf, mnfld_points, 0, args, state=f"{int(upsampled)}ups")

            upsampled += 1.0
            print("sites length AFTER: ", len(sites))

        if args.video:
            extract_mesh(sites, sites_sdf, mnfld_points, 0, args, state=f"{int(epoch)}")

    return sites, sites_sdf


def compute_cvt_loss_vectorized_delaunay(sites, delaunay, simplices=None):
    centroids, _ = compute_voronoi_cell_centers_index_based_torch(sites, delaunay, simplices)
    centroids = centroids.to(device)
    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(abs(diff) < 0.1, diff, torch.tensor(0.0, device=sites.device))
    # cvt_loss = torch.mean(penalties**2)
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


def smoothed_heaviside(phi, eps_H):
    H = torch.zeros_like(phi)
    mask1 = phi < -eps_H
    mask2 = phi > eps_H
    mask3 = (~mask1) & (~mask2)
    phi_clip = phi[mask3]
    H[mask1] = 0
    H[mask2] = 1
    H[mask3] = 0.5 + phi_clip / (2 * eps_H) + (1 / (2 * np.pi)) * torch.sin(np.pi * phi_clip / eps_H)
    return H


def tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, tets, eps_H) -> torch.Tensor:
    if eps_H is None:
        eps_H = estimate_eps_H(sites, tets)  # adaptive bandwidth
    sdf_H = smoothed_heaviside(sites_sdf, eps_H)  # (M,)
    sdf_H_a = sdf_H[tets[:, 0]]
    sdf_H_b = sdf_H[tets[:, 1]]
    sdf_H_c = sdf_H[tets[:, 2]]
    sdf_H_d = sdf_H[tets[:, 3]]
    sdf_H_stack = torch.stack([sdf_H_a, sdf_H_b, sdf_H_c, sdf_H_d], dim=1)  # (M, 4)

    sdf_H_center = sdf_H_stack.mean(dim=1, keepdim=True)  # (M, 1)
    sdf_H_diff = sdf_H_stack - sdf_H_center  # (M, 4)

    grad_H_tet = torch.einsum("mi,mij->mj", sdf_H_diff, W)  # (M, 3)
    grad_norm = grad_H_tet.norm(dim=1)  # (M, 1)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]
    volume = volume_tetrahedron(a, b, c, d)  # (M,)
    # trim 5% biggest volumes
    volume = torch.where(volume > torch.quantile(volume, 0.95), torch.tensor(0.0, device=sites.device), volume)
    penalties = torch.mean(volume * grad_norm)
    # penalties = torch.mean(grad_norm)

    # return torch.mean(volume * grad_norm)
    return penalties


def discrete_tet_volume_eikonal_loss(sites, sites_sdf_grad, tets: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularization loss.

    Args:
        sites_sdf_grad: Tensor of shape (N, 3) containing ∇φ at each site.
        variant: 'a' for E1a: ½ mean((||∇φ|| - 1)²)
    Returns:
        A scalar tensor containing the eikonal loss.
    """
    grad_a = sites_sdf_grad[tets[:, 0]]  # (M,3)
    grad_b = sites_sdf_grad[tets[:, 1]]  # (M,3)
    grad_c = sites_sdf_grad[tets[:, 2]]  # (M,3)
    grad_d = sites_sdf_grad[tets[:, 3]]  # (M,3)

    grad_a_error = ((grad_a**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_b_error = ((grad_b**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_c_error = ((grad_c**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_d_error = ((grad_d**2).sum(dim=-1) - 1) ** 2  # (M,)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]

    volume = volume_tetrahedron(a, b, c, d)

    loss = 0.5 * torch.mean(volume * (grad_a_error + grad_b_error + grad_c_error + grad_d_error))  # (M,)

    return loss


def estimate_eps_H(sites, tets, multiplier=1.5):
    # Get all unique edges
    comb = torch.combinations(torch.arange(4), r=2)  # (6,2)
    edges = tets[:, comb]  # (M, 6, 2)
    edges = edges.reshape(-1, 2)  # (6M, 2)

    v0 = sites[edges[:, 0]]
    v1 = sites[edges[:, 1]]
    edge_lengths = torch.norm(v0 - v1, dim=1)

    # remove top 5% longest edges
    edge_lengths = torch.where(
        edge_lengths > torch.quantile(edge_lengths, 0.95), torch.tensor(0.0, device=sites.device), edge_lengths
    )

    avg_len = edge_lengths.mean()
    return multiplier * avg_len


def compute_voronoi_cell_centers_index_based_torch(sites, delau, simplices=None):
    """Compute Voronoi cell centers (circumcenters) for 2D or 3D Delaunay triangulation in PyTorch."""
    # simplices = torch.tensor(delaunay.simplices, dtype=torch.long)
    if simplices is None:
        simplices = delau.simplices

    # points = torch.tensor(delaunay.points, dtype=torch.float32)
    points = sites.detach().cpu().numpy()

    # Compute all circumcenters at once (supports both 2D & 3D)
    circumcenters_arr = circumcenter_torch(points, simplices)
    # Flatten simplices and repeat circumcenters to map them to the points
    indices = simplices.flatten()  # Flatten simplex indices
    indices = torch.tensor(indices, dtype=torch.int64, device=sites.device)  # Convert to tensor

    centers = circumcenters_arr.repeat_interleave(simplices.shape[1], dim=0).to(
        sites.device
    )  # Repeat for each vertex in simplex

    # Group circumcenters per point
    M = len(points)
    # Compute the sum of centers for each index
    centroids = torch.zeros(M, 3, dtype=torch.float32, device=sites.device)
    counts = torch.zeros(M, device=sites.device)

    centroids.index_add_(0, indices, centers)  # Sum centers per unique index
    counts.index_add_(0, indices, torch.ones(centers.shape[0], device=centers.device))  # Count occurrences
    centroids /= counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    distances = torch.norm(centroids[indices] - centers, dim=1)
    num_sites = centroids.shape[0]
    max_dist_per_site = torch.full((num_sites,), float("-inf"), device=sites.device)
    radius = max_dist_per_site.scatter_reduce(0, indices, distances, reduce="amax", include_self=True)

    return centroids, radius


def circumcenter_torch(points, simplices):
    """Compute the circumcenters for 2D triangles or 3D tetrahedra in a vectorized manner using PyTorch."""
    points = torch.tensor(points, dtype=torch.float32)
    simplices = torch.tensor(simplices, dtype=torch.long)

    if points.shape[1] == 2:  # **2D Case (Triangles)**
        p1, p2, p3 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]

        # Compute determinant (D)
        D = 2 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1]))

        # Compute circumcenter coordinates
        ux = (
            (p1[:, 0] ** 2 + p1[:, 1] ** 2) * (p2[:, 1] - p3[:, 1])
            + (p2[:, 0] ** 2 + p2[:, 1] ** 2) * (p3[:, 1] - p1[:, 1])
            + (p3[:, 0] ** 2 + p3[:, 1] ** 2) * (p1[:, 1] - p2[:, 1])
        ) / D

        uy = (
            (p1[:, 0] ** 2 + p1[:, 1] ** 2) * (p3[:, 0] - p2[:, 0])
            + (p2[:, 0] ** 2 + p2[:, 1] ** 2) * (p1[:, 0] - p3[:, 0])
            + (p3[:, 0] ** 2 + p3[:, 1] ** 2) * (p2[:, 0] - p1[:, 0])
        ) / D

        return torch.stack((ux, uy), dim=1)

    elif points.shape[1] == 3:  # **3D Case (Tetrahedra)**
        """
        Compute the circumcenters of multiple tetrahedra in a 3D Delaunay triangulation.

        Parameters:
        points : tensor of shape (N, 3)
            The 3D coordinates of all input points.
        simplices : tensor of shape (M, 4)
            Indices of tetrahedron vertices in `points`.

        Returns:
        circumcenters : tensor of shape (M, 3)
            The circumcenters of all tetrahedra.
        """
        # Extract tetrahedral vertices using broadcasting
        A = points[simplices[:, 0]]  # Shape: (M, 3)
        B = points[simplices[:, 1]]
        C = points[simplices[:, 2]]
        D = points[simplices[:, 3]]

        # Compute edge vectors relative to A
        BA = B - A  # Shape: (M, 3)
        CA = C - A
        DA = D - A

        # Compute squared edge lengths
        len_BA = torch.sum(BA**2, axis=1, keepdims=True)  # Shape: (M, 1)
        len_CA = torch.sum(CA**2, axis=1, keepdims=True)
        len_DA = torch.sum(DA**2, axis=1, keepdims=True)

        # Compute cross products
        cross_CD = torch.linalg.cross(CA, DA)  # Shape: (M, 3)
        cross_DB = torch.linalg.cross(DA, BA)
        cross_BC = torch.linalg.cross(BA, CA)

        # Compute denominator (scalar for each tetrahedron)
        denominator = 0.5 / torch.sum(BA * cross_CD, axis=1, keepdims=True)  # Shape: (M, 1)

        # Compute circumcenter offsets
        circ_offset = (len_BA * cross_CD + len_CA * cross_DB + len_DA * cross_BC) * denominator  # Shape: (M, 3)

        # Compute circumcenters
        circumcenters = A + circ_offset  # Shape: (M, 3)

        return circumcenters
    else:
        raise ValueError("Only 2D (triangles) and 3D (tetrahedra) are supported.")


def compute_cvt_loss_CLIPPED_vertices(sites, d3dsimplices, all_vor_vertices):
    d3dsimplices = torch.tensor(d3dsimplices, device=sites.device).detach()
    # compute centroids
    indices = d3dsimplices.flatten()  # Flatten simplex indices
    centers = all_vor_vertices.repeat_interleave(d3dsimplices.shape[1], dim=0).to(sites.device)
    M = len(sites)
    centroids = torch.zeros(M, 3, dtype=torch.float32, device=sites.device)
    counts = torch.zeros(M, device=sites.device)

    centroids.index_add_(0, indices, centers)  # Sum centers per unique index
    counts.index_add_(0, indices, torch.ones(centers.shape[0], device=centers.device))  # Count occurrences
    centroids /= counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(abs(diff) < 0.5, diff, torch.tensor(0.0, device=sites.device))
    # print number of zero in penalties
    # print("Number of zero in penalties: ", torch.sum(penalties == 0.0).item())
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


def compute_cvt_loss_true(sites, d3d, vertices=None):
    if vertices == None:
        vertices = compute_vertices_3d_vectorized(sites, d3d)

    # Concat sites and vertices to compute the Voronoi diagram
    points = torch.concatenate((sites, vertices), axis=0)
    # Avoid to get coplanar tet which create issue if the current algorithm
    points += (torch.rand_like(points) - 0.5) * 0.00001  # 0.001 % of the space ish
    d3dsimplices, _ = pygdel3d.triangulate(points.detach().cpu().numpy())
    # d3dsimplices = Delaunay(points.detach().cpu().numpy()).simplices
    d3dsimplices = torch.tensor(d3dsimplices, dtype=torch.int64, device=sites.device)

    ############ 2D Case (Triangles) ############
    # Compute the areas of all simplices (in 2D triangles)
    # a = points[d3dsimplices[:, 0]]
    # b = points[d3dsimplices[:, 1]]
    # c = points[d3dsimplices[:, 2]]
    # # areas_simplices = torch.linalg.norm(torch.cross(b - a, c - a), dim=1) / 2.0
    # triangle_areas = torch.linalg.norm(b - a, dim=1) * torch.linalg.norm(c - a, dim=1) / 2.0
    # triangle_center = (a + b + c) / 3.0
    # # print(triangle_areas.shape, triangle_center.shape)
    ############ 3D Case (Tetrahedra) ############
    a = points[d3dsimplices[:, 0]]
    b = points[d3dsimplices[:, 1]]
    c = points[d3dsimplices[:, 2]]
    d = points[d3dsimplices[:, 3]]

    tetrahedra_volume = volume_tetrahedron(a, b, c, d)
    tetrahedra_center = (a + b + c + d) / 4.0  # Shape: (M, 3)

    # Create a centroid for each sites
    centroids = torch.zeros_like(sites)
    volumes = torch.ones(sites.shape[0], dtype=torch.float32, device=sites.device) * 1e-8  # Avoid division by zero
    for i in range(4):
        # Filter simplices that are valid (i.e., not out of bounds)
        # We assume that the first N points are the sites
        mask = d3dsimplices[:, i] < sites.shape[0]
        # Uses index_add for atomic addition
        centroids.index_add_(0, d3dsimplices[mask, i], tetrahedra_center[mask] * tetrahedra_volume[mask].unsqueeze(1))
        volumes.index_add_(0, d3dsimplices[mask, i], tetrahedra_volume[mask])
    centroids /= volumes.unsqueeze(1)

    cvt_loss = torch.mean(torch.norm(sites - centroids, dim=1))
    # cvt_loss = torch.mean(torch.abs(sites - centroids))

    # print("Centroids shape:", centroids.shape)
    # print("Sites shape:", sites.shape)
    # return centroids, vertices
    return cvt_loss


def sample_mesh_points_heitz(vertices: torch.Tensor, faces: torch.LongTensor, num_samples: int) -> torch.Tensor:
    """
    Uniformly (area weighted) sample points on a triangular mesh
    using Heitz s low distortion square→triangle mapping.

    Args:
        vertices:    (V,3) float tensor of vertex positions.
        faces:       (F,3) long tensor of indices into `vertices`.
        num_samples: int, number of points to sample.

    Returns:
        samples: (num_samples, 3) float tensor of sampled points.
    """
    # 1) Gather triangle vertices
    v0 = vertices[faces[:, 0]]  # (F,3)
    v1 = vertices[faces[:, 1]]  # (F,3)
    v2 = vertices[faces[:, 2]]  # (F,3)

    # 2) Compute triangle areas for weighting
    e0 = v1 - v0  # (F,3)
    e1 = v2 - v0  # (F,3)
    cross = torch.cross(e0, e1, dim=1)  # (F,3)
    areas = 0.5 * cross.norm(dim=1)  # (F,)

    # 3) Sample faces proportional to area
    probs = areas / areas.sum()
    idx = torch.multinomial(probs, num_samples, replacement=True)  # (num_samples,)

    # 4) Draw uniform samples u,v in [0,1]^2
    u = torch.rand(num_samples, device=vertices.device)
    v = torch.rand(num_samples, device=vertices.device)

    # 5) Heitz–Talbot mapping to barycentric (b0,b1,b2):
    #    b0 = u/2, b1 = v/2, then shift one cell to compress diagonals
    b0 = 0.5 * u
    b1 = 0.5 * v
    offset = b1 - b0
    mask = offset > 0
    # if offset>0, push b1 out; else pull b0 back
    b1 = torch.where(mask, b1 + offset, b1)
    b0 = torch.where(mask, b0, b0 - offset)
    b2 = 1.0 - b0 - b1

    # 6) Assemble final sample positions
    #    pick the triangle vertices for each sample
    v0s = v0[idx]  # (num_samples,3)
    v1s = v1[idx]
    v2s = v2[idx]
    # reshape barycentric coords for broadcast
    b0 = b0.unsqueeze(1)  # (num_samples,1)
    b1 = b1.unsqueeze(1)
    b2 = b2.unsqueeze(1)
    samples = b0 * v0s + b1 * v1s + b2 * v2s  # (num_samples,3)

    return samples


def cvt_extraction(sites, sites_sdf, d3dsimplices, build_faces=False):
    """
    Extracts a mesh from the given sites and their SDF values.
    """
    d3d = torch.tensor(d3dsimplices, device=device)  # (M,4)
    all_vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)

    all_vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3d, sites, sites_sdf)

    # print("Voronoi vertices shape:", all_vor_vertices.shape, "SDF values shape:", all_vertices_sdf.shape)

    vertices_to_compute, zero_crossing_sites_pairs, used_tet = compute_zero_crossing_vertices_3d(
        sites, None, None, d3dsimplices, sites_sdf
    )
    vertices = compute_vertices_3d_vectorized(sites, vertices_to_compute)

    sdf_verts = interpolate_sdf_of_vertices(vertices, d3d[used_tet], sites, sites_sdf)

    # print("Vertices to compute:", vertices.shape, "SDF values shape:", sdf_verts.shape)

    tet_sites = sites[d3d[used_tet]]  # (M,4,3)
    tet_sdf = sites_sdf[d3d[used_tet]]  # (M,4)
    signs = torch.sign(tet_sdf)  # (M,4)
    sign_sum = torch.abs(signs.sum(dim=1))  # (M,)
    valid_mask = sign_sum < 4  # (M,) True if there's a sign change

    cross_mask = signs.unsqueeze(2) * signs.unsqueeze(1) < 0  # (M,4,4)
    site_mask = cross_mask.any(dim=2)  # (M,4) True if site has at least one sign change edge

    # -------------
    # Broadcast vertex to shape (M, 4, 3)
    v = vertices.unsqueeze(1)  # (M, 1, 3)
    phi_v = sdf_verts.unsqueeze(1)  # (M, 1)

    # Compute displacement vectors to sites
    delta = tet_sites - v  # (M, 4, 3)
    phi_i = tet_sdf  # (M, 4)

    # Compute interpolation weights (M, 4, 1)
    denom = (phi_v - phi_i).unsqueeze(-1)  # (M, 4, 1)
    numer = phi_v.unsqueeze(-1)  # (M, 1, 1)

    # Avoid division by zero
    eps = 1e-8
    denom = denom.clamp(min=-1e6, max=1e6)  # optional clamp for safety

    t = numer / denom  # (M, 4, 1)

    # Only keep valid projections: site must have opposite sign from vertex
    signs_diff = (phi_v * phi_i) < 0  # (M, 4)
    t[~signs_diff.unsqueeze(-1)] = 0.0  # zero out invalid

    # Interpolated positions per site
    p_i = v + t * delta  # (M, 4, 3)
    valid_mask = signs_diff.unsqueeze(-1)  # (M, 4, 1)

    # Average all valid interpolated positions
    num_valid = valid_mask.sum(dim=1).clamp(min=1)  # (M, 1, 1)
    projected = (p_i * valid_mask).sum(dim=1) / num_valid  # (M, 3)
    # ------------
    if build_faces:
        faces = get_faces(d3dsimplices, sites, all_vor_vertices, None, sites_sdf)  # (R0, List of simplices)
        # Compact the vertex list
        used = {idx for face in faces for idx in face}
        old2new = {old: new for new, old in enumerate(sorted(used))}
        new_vertices = all_vor_vertices[sorted(used)]
        new_faces = [[old2new[i] for i in face] for face in faces]
        return projected, new_faces

    vert_for_clipped_cvt = all_vor_vertices
    vert_for_clipped_cvt[used_tet] = projected
    return projected, vert_for_clipped_cvt


def extract_mesh(sites, model, target_pc, time, args, state="", d3dsimplices=None, t=time()):
    print(f"Extracting mesh at state: {state} with upsampling: {args.upsampling}")
    sdf_values = resolve_sdf_values(model, sites, verbose=True)  # (N,)

    sites_np = sites.detach().cpu().numpy()
    d3dsimplices = Delaunay(sites_np).simplices

    if args.w_chamfer > 0:
        v_vect, f_vect = cvt_extraction(sites, sdf_values, d3dsimplices, True)
        output_obj_file = build_dccvt_obj_path(args, state, "intDCCVT")
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        save_target_pc_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())

        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = get_clipped_mesh_numba(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        output_obj_file = build_dccvt_obj_path(args, state, "projDCCVT")
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        save_target_pc_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())

    if args.w_voroloss > 0:
        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = get_clipped_mesh_numba(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        output_obj_file = build_voromesh_obj_path(args, state)
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
    if args.w_mc > 0:
        # TODO:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        d3dsimplices = torch.tensor(d3dsimplices, device=device)
        marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
            sites.unsqueeze(0), d3dsimplices, sdf_values.unsqueeze(0), return_tet_idx=False
        )
        vertices_list, faces_list = marching_tetrehedra_mesh
        vertices = vertices_list[0]
        faces = faces_list[0]
        vertices_np = vertices.detach().cpu().numpy()  # Shape [N, 3]
        faces_np = faces.detach().cpu().numpy()  # Shape [M, 3] (triangles)
        output_obj_file = build_dccvt_obj_path(args, state, "MT")
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj(output_obj_file, vertices_np, faces_np)


def save_npz(sites, sites_sdf, time, args, output_file):
    np.savez(
        output_file,
        sites=sites.detach().cpu().numpy(),
        sites_sdf=sites_sdf.detach().cpu().numpy(),
        train_time=time,
        args=str(args),
    )


def save_obj(filename, vertices, faces):
    """
    Save a mesh to an OBJ file.

    Args:
        filename: str, path to output .obj file
        vertices: (N, 3) array of float vertex positions
        faces: (M, 3) or (M, K) array of int indices (0-based)
    """
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Offset indices by +1 for OBJ (1-based indexing)
        for face in faces:
            # Ensure all face entries are written, even if quads or ngons
            indices = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {indices}\n")


def save_target_pc_ply(filename, points):
    """
    Save a point cloud to a PLY file.

    Args:
        filename: str, path to output .ply file
        points: (N, 3) array of float point positions
    """
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def copy_script(arg_lists):
    script_path = os.path.abspath(__file__)
    script_copy_path = os.path.join(DEFAULTS["output"], script_path.split("/")[-1])
    os.makedirs(DEFAULTS["output"], exist_ok=True)
    with open(script_copy_path, "w") as f:
        f.write(open(script_path).read())

    # copy arg_lists in other file
    arg_list_file = os.path.join(DEFAULTS["output"], "arg_lists.txt")
    with open(arg_list_file, "w") as f:
        for arg_list in arg_lists:
            f.write(" ".join(arg_list) + "\n")
        print(f"Copied script to {script_copy_path} and arg lists to {arg_list_file}")


def process_single_mesh(arg_list):
    args = define_options_parser(arg_list)
    args.save_path = f"{args.output}" if args.save_path is None else args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    use_chamfer = args.w_chamfer > 0
    use_voroloss = args.w_voroloss > 0

    # if not os.path.exists(f"{args.save_path}/marching_tetrahedra_{args.upsampling}_final_MT.obj"):
    output_obj_file = check_if_already_processed(args)
    if os.path.exists(output_obj_file):
        print(f"Skipping already processed mesh: {output_obj_file}")
    else:
        print("args: ", args)
        try:
            model, mnfld_points = load_model(args.mesh, args.target_size, args.trained_HotSpot)
            sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

            if use_chamfer:
                if args.sdf_type == "bounding_sphere":
                    # bounding sphere of mnfld points
                    radius = torch.norm(mnfld_points, dim=-1).max().item() + 0.1
                    sdf = torch.norm(sites - torch.zeros(3).to(device), dim=-1) - radius
                    sdf = sdf.detach().squeeze(-1).requires_grad_()
                    print("sdf:", sdf.shape, sdf.dtype, sdf.is_leaf)
                elif args.sdf_type == "complex_alpha":
                    sdf = complex_alpha_sdf(mnfld_points, sites)
                    print("sdf:", sdf.shape, sdf.dtype, sdf.is_leaf)

                else:
                    sdf = init_sdf(model, sites)
            else:
                sdf = model

            # Extract the initial mesh
            extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

            if use_chamfer or use_voroloss:
                t0 = time()
                sites, sdf = train_DCCVT(sites, sdf, mnfld_points, model, args)
                ti = time() - t0

            # Extract the final mesh
            extract_mesh(sites, sdf, mnfld_points, ti, args, state="final")
        except Exception as e:
            print(f"Error processing mesh {args.mesh}: {e}")
        else:
            print(f"Finished processing mesh: {args.mesh}")
            torch.cuda.empty_cache()


def complex_alpha_sdf(mnfld_points, sites):
    def alpha_shape_3d(points: np.ndarray, alpha: float):
        """
        Build a 3D alpha shape mesh from points using Gudhi.
        alpha: radius parameter (not squared). Smaller -> tighter, more concave; too small -> holes/missing parts.
        Returns V,F for a triangle surface mesh.
        """
        ac = gudhi.AlphaComplex(points=points)
        st = ac.create_simplex_tree(max_alpha_square=alpha * alpha)

        # Collect tetrahedra (3-simplices) and triangles (2-simplices) in the complex
        tets = []
        tris = []
        for simplex, filt in st.get_skeleton(3):
            if len(simplex) == 4:
                tets.append(tuple(sorted(simplex)))
            elif len(simplex) == 3:
                tris.append(tuple(sorted(simplex)))

        # Count how many tetrahedra incident to each triangle; boundary triangles have <=1 incident tet
        tri_incidence = defaultdict(int)
        for tet in tets:
            a, b, c, d = tet
            faces = [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
            for f in faces:
                tri_incidence[tuple(sorted(f))] += 1

        boundary_tris = []
        for tri in tris:
            if tri_incidence.get(tri, 0) <= 1:
                boundary_tris.append(tri)

        V = points.copy()
        F = np.array(boundary_tris, dtype=int)

        # Clean up with trimesh (remove degenerates, unify winding, fill tiny holes if needed)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh, True)  # consistent winding + outward if possible
        trimesh.repair.fix_inversion(mesh, True)  # resolves inside-out components
        print(trimesh.repair.broken_faces(mesh))
        assert mesh.is_watertight, "Alpha mesh not watertight; tune alpha or repair."
        assert mesh.is_winding_consistent, "Inconsistent winding; fix_normals should help."
        return mesh

    def pick_alpha(points, k=8, quantile=0.9, magnitude=15.0):
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        dists, _ = nbrs.kneighbors(points)
        # ignore the zero distance to self at column 0 by slicing from 1:
        scale = np.quantile(dists[:, 1:].mean(axis=1), quantile)
        # for some reasons at 1.5 mesh is not watertight and trimesh cant fix it
        # so for safety we multiply by 15
        return magnitude * scale

    alpha = pick_alpha(mnfld_points.squeeze(0).detach().cpu().numpy())  # or set manually
    mesh = alpha_shape_3d(mnfld_points.squeeze(0).detach().cpu().numpy(), alpha)
    S = -trimesh.proximity.signed_distance(mesh, sites.detach().cpu().numpy())
    sdf0 = torch.from_numpy(S).to(device, dtype=torch.float32).requires_grad_()
    return sdf0


def upsampling_adaptive_vectorized_sites_sites_sdf(
    sites: torch.Tensor,  # (N,3)
    simplices=None,  # np.ndarray (M,4) if tri is None
    model=None,  # SDFGrid | nn.Module | Tensor (N,)
    sites_sdf_grads=None,  # (N,3) spatial gradients ∇φ at each site
    spacing_target: float = None,  # desired final spacing  (same units as sites)
    alpha_high: float = 1.5,  # regime switches   (α_high > α_low ≥ 1)
    alpha_low: float = 1.1,
    curv_pct: float = 0.75,  # percentile threshold for curvature pass
    growth_cap: float = 0.10,  # ≤ fraction of current sites allowed per iter
    eps: float = 1e-12,
    ups_method: str = "tet_frame",  # | "tet_random" | "random",
    score: str = "legacy",  # | "density" | "cosine" | "conservative"
):
    """
    # ------------------------------------------------------------------------------
    # Adaptive upsample: balances uniform coverage and high-curvature refinement
    # ------------------------------------------------------------------------------
    Returns:
        updated_sites      -- (N+4K,3)
        updated_sites_sdf  -- (N+4K,)
    """
    device = sites.device
    N = sites.shape[0]

    # SDF at original sites
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        sdf_values = model(sites).detach()
    sdf_values = sdf_values.squeeze()  # (N,)

    # Build edge list (ridge points)

    all_tets = torch.as_tensor(simplices, device=device).long()

    edges = torch.cat(
        [
            all_tets[:, [0, 1]],
            all_tets[:, [1, 2]],
            all_tets[:, [2, 3]],
            all_tets[:, [3, 0]],
            all_tets[:, [0, 2]],
            all_tets[:, [1, 3]],
        ],
        dim=0,
    )
    neighbors, _ = torch.sort(edges, dim=1)
    neighbors = torch.unique(neighbors, dim=0)  # (E,2)

    # Local spacing ρᵢ  (shortest incident edge)
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]  # (E,3)
    edge_len = torch.norm(edge_vec, dim=1)  # (E,)

    # Compute minimum distance to neighbors
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((N,), float("inf"), device=device)
    min_dists = min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")  # (N,)

    # # Gradient ∇φ and curvature proxy κᵢ  (1-ring normal variation)
    # # ∇φ estimate (scatter-add of finite-difference contributions)
    sdf_diff = sdf_values[neighbors[:, 1]] - sdf_values[neighbors[:, 0]]
    sdf_diff = sdf_diff.unsqueeze(1)  # (E,1)

    counts = torch.zeros((N, 1), device=device)
    ones = torch.ones_like(sdf_diff)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    # grad_est /= counts.clamp(min=1.0)

    grad_est = sites_sdf_grads

    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)

    if score == "density":
        # Disable curvature score, use uniform density
        curv_score = torch.ones(N, device=device)  # (N,)
    else:
        curv_score = torch.zeros(N, device=device)  # (N,)
        if score != "cosine":
            if score != "conservative":
                # Legacy curvature score: squared distance between normals
                dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1)
            else:
                dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1) * 0.8 + 0.2  # (E,)
        else:
            # Cosine similarity as curvature score from dot product
            # Make it conservartive
            dn2 = (1.0 - (unit_n[neighbors[:, 0]] * unit_n[neighbors[:, 1]]).sum(1)) * 0.8 + 0.2

        curv_score = curv_score.index_add(0, neighbors[:, 0], dn2)
        curv_score = curv_score.index_add(0, neighbors[:, 1], dn2)
        curv_score /= counts.squeeze()  # mean over 1-ring

    # Zero-crossing sites
    sdf_i, sdf_j = sdf_values[neighbors[:, 0]], sdf_values[neighbors[:, 1]]
    mask_zc = sdf_i * sdf_j <= 0
    zc_sites = torch.unique(neighbors[mask_zc].reshape(-1))

    # Decide regime (uniform / hybrid / curvature)
    median_min_dists = torch.median(min_dists)  # global spacing
    if spacing_target is None:
        spacing_target = median_min_dists * 0.8  # heuristic default

    score = (min_dists[zc_sites] / torch.median(min_dists[zc_sites])) * (
        curv_score[zc_sites] / (torch.median(curv_score[zc_sites]) + eps)
    )

    M = int(min(max(1, growth_cap * N), score.numel()))

    # # Construct cumsum of the scores WITHOUT sorting
    cumsum_scores = torch.cumsum(score, dim=0)
    total_score = cumsum_scores[-1].item()  # Last element is the total sum
    cumsum_scores /= total_score  # Normalize to [0, 1]

    # Sample M indices based on the cumulative distribution
    random_indices = torch.rand(M, device=device)  # (M,)
    sampled_indices = torch.searchsorted(cumsum_scores, random_indices)  # (M,

    # Remove duplicates
    sampled_indices = torch.unique(sampled_indices)
    sampled_indices = sampled_indices[sampled_indices < score.numel()]  # Ensure valid indices

    # Show the candidates percentage
    print(f"Sampled indices: {sampled_indices.numel()} out of {score.numel()} candidates (M={M})")

    cand = zc_sites[sampled_indices]  # (K,)

    K = cand.numel()
    if K == 0:
        return sites, sdf_values  # nothing selected

    if ups_method == "tet_frame":
        # -------------------------------------------------- #
        # Insert 4 off-spring per selected site (regular tetrahedron)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
            dtype=torch.float32,
            device=device,
        )  # (4,3)
        tetr_dirs = torch.nn.functional.normalize(tetr_dirs, dim=1)  # Normalize directions

        # Build local frame from ∇ϕ (surface normal)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)
        unit_grad = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # point outward
        frame = build_tangent_frame(unit_grad)  # (K,3,3)

        # Optional anisotropic weights for axes: (t1, t2, normal)
        anisotropy = torch.tensor([1.0, 1.0, 0.5], device=device)  # shrink in normal
        frame = frame * anisotropy.view(1, 1, 3)  # (K,3,3)

        # Rotate the 4 tetrahedral directions into local frame
        local_dirs = tetr_dirs.T.unsqueeze(0)  # (1,3,4)
        offs = torch.matmul(frame, local_dirs).permute(0, 2, 1)  # (K,4,3)
        # Scale by local spacing
        # used to be /4 but because anisotropy 0.5 its now /2
        scale = (min_dists[cand] / 2).unsqueeze(1).unsqueeze(2)  # (K,1,1)
        offs = offs * scale  # (K,4,3)

        # Translate from centroid
        centroids = sites[cand].unsqueeze(1)  # (K,1,3)
        new_sites = (centroids + offs).reshape(-1, 3)  # (4K,3)

        delta = new_sites.reshape(-1, 4, 3) - centroids  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    elif ups_method == "tet_frame_remove_parent":
        # -------------------------------------------------- #
        # Insert 4 off-spring per selected site (regular tetrahedron)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
            dtype=torch.float32,
            device=device,
        )  # (4,3)
        tetr_dirs = torch.nn.functional.normalize(tetr_dirs, dim=1)  # Normalize directions

        # Build local frame from ∇ϕ (surface normal)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)
        unit_grad = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # point outward
        frame = build_tangent_frame(unit_grad)  # (K,3,3)

        # Optional anisotropic weights for axes: (t1, t2, normal)
        anisotropy = torch.tensor([1.0, 1.0, 0.5], device=device)  # shrink in normal
        frame = frame * anisotropy.view(1, 1, 3)  # (K,3,3)

        # Rotate the 4 tetrahedral directions into local frame
        local_dirs = tetr_dirs.T.unsqueeze(0)  # (1,3,4)
        offs = torch.matmul(frame, local_dirs).permute(0, 2, 1)  # (K,4,3)
        # Scale by local spacing
        # used to be /4 but because anisotropy 0.5 its now /2
        scale = (min_dists[cand] / 2).unsqueeze(1).unsqueeze(2)  # (K,1,1)
        offs = offs * scale  # (K,4,3)

        # Translate from centroid
        centroids = sites[cand].unsqueeze(1)  # (K,1,3)
        new_sites = (centroids + offs).reshape(-1, 3)  # (4K,3)

        N = sites.shape[0]
        if cand.dtype == torch.bool:
            parent_mask = ~cand
        else:
            parent_mask = torch.ones(N, dtype=torch.bool, device=sites.device)
            parent_mask[cand] = False  # drop parents

        delta = new_sites.reshape(-1, 4, 3) - centroids  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)
        updated_sites = torch.cat([sites[parent_mask], new_sites], dim=0)
        updated_sites_sdf = torch.cat([sdf_values[parent_mask], new_sdf], dim=0)

    elif ups_method == "tet_random":
        # ---------------------------------------------------------------- #
        # Canonical regular-tetrahedron vertex directions (centered at origin)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=torch.float32, device=device
        )  # (4,3)

        K = cand.shape[0]
        centroids = sites[cand]  # (K,3)
        scale = (min_dists[cand] / 4).unsqueeze(1)  # (K,1)

        # --- Make a random rotation per centroid via random unit quaternions ---
        def quat_to_rotmat(q):  # q: (...,4) normalized
            w, x, y, z = q.unbind(-1)
            ww, xx, yy, zz = w * w, x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            R = torch.stack(
                [
                    ww + xx - yy - zz,
                    2 * (xy - wz),
                    2 * (xz + wy),
                    2 * (xy + wz),
                    ww - xx + yy - zz,
                    2 * (yz - wx),
                    2 * (xz - wy),
                    2 * (yz + wx),
                    ww - xx - yy + zz,
                ],
                dim=-1,
            ).reshape(q.shape[:-1] + (3, 3))
            return R

        # Random unit quaternions (K,4)
        q = torch.randn(K, 4, device=device, dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        R = quat_to_rotmat(q)  # (K,3,3)

        # Rotate the canonical tetra directions per centroid: (K,4,3)
        rotated_dirs = tetr_dirs.unsqueeze(0) @ R.transpose(-1, -2)

        # Build offspring: keep proportions via `scale`
        new_sites = (centroids.unsqueeze(1) + rotated_dirs * scale.unsqueeze(1)).reshape(-1, 3)  # (4K,3)

        print("Before tet random upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

        # First-order SDF interpolation φ(new) = φ(old) + ∇φ·δ
        cent_grad = grad_est[cand]  # (K,3)
        delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)

        # Concatenate & return
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    elif "random":
        eps = 1e-12
        # Inputs
        centroids = sites[cand]  # (K,3)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)  # (K,3)

        # Hemisphere axis = -∇φ direction (to match your previous "minus grad" step)
        axis = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # (K,3)

        # Build an orthonormal basis {v1, v2, axis} per centroid
        helper = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(axis).clone()
        near_pole = axis[:, 2].abs() > 0.99  # if axis ~ ±z, switch helper to avoid degeneracy
        helper[near_pole] = torch.tensor([0.0, 1.0, 0.0], device=device)

        v1 = torch.cross(helper, axis, dim=1)
        v1 = v1 / (v1.norm(dim=1, keepdim=True) + eps)  # (K,3)
        v2 = torch.cross(axis, v1, dim=1)
        v2 = v2 / (v2.norm(dim=1, keepdim=True) + eps)  # (K,3)

        # Uniform random direction on hemisphere around `axis`
        # cos(theta) ~ U[0,1], phi ~ U[0, 2π)
        K = centroids.shape[0]
        u = torch.rand(K, 1, device=device)  # cos(theta)
        phi = 2.0 * math.pi * torch.rand(K, 1, device=device)  # azimuth

        sin_theta = torch.sqrt((1.0 - u**2).clamp_min(0.0))  # (K,1)
        dir_hemi = (torch.cos(phi) * sin_theta) * v1 + (torch.sin(phi) * sin_theta) * v2 + u * axis  # (K,3)
        # dir_hemi is unit-length (up to numerical eps)

        # Step radius: keep your existing spacing-based step size
        step_size = (min_dists[cand] / 4.0).unsqueeze(1)  # (K,1)
        new_sites = centroids + step_size * dir_hemi  # (K,3)

        print("Before upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

        # First-order SDF interpolation: φ(new) = φ(old) + ∇φ · δ
        delta = new_sites - centroids  # (K,3)
        new_sdf = sdf_values[cand] + (cent_grad * delta).sum(dim=1)  # (K,)

        # Concatenate and return
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+K, 3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+K,)
    else:
        raise ValueError(f"Unknown upsampling method: {ups_method}")
    return updated_sites, updated_sites_sdf


def build_tangent_frame(normals):  # normals: (B, 3)
    B = normals.shape[0]
    device = normals.device

    # Choose a global "up" vector that is not parallel to most normals
    up = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
    alt = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3)

    dot = (normals * up).sum(dim=1, keepdim=True).abs()  # (B,1)
    fallback = dot > 0.9  # (B,1) --> mask to avoid colinearity
    base = torch.where(fallback, alt, up)  # (B,3)

    tangent1 = torch.nn.functional.normalize(torch.cross(normals, base, dim=1), dim=1)
    tangent2 = torch.nn.functional.normalize(torch.cross(normals, tangent1, dim=1), dim=1)

    return torch.stack([tangent1, tangent2, normals], dim=-1)  # (B, 3, 3)


def check_if_already_processed(args):
    state = "final"
    if args.w_chamfer > 0:
        output_obj_file = build_dccvt_obj_path(args, state, "projDCCVT")
    if args.w_voroloss > 0:
        output_obj_file = build_voromesh_obj_path(args, state)
    if args.w_mc > 0:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        output_obj_file = build_dccvt_obj_path(args, state, "MT")
    return output_obj_file


if __name__ == "__main__":
    root = argparse.ArgumentParser(add_help=True)
    root.add_argument(
        "--args-file",
        type=str,
        default=None,
        help="Text file: one experiment template per line. Use {mesh_id} to expand.",
    )
    root.add_argument("--mesh-ids", type=str, default=None, help="Override mesh list (comma/space separated).")
    root.add_argument("--timestamp", type=str, default=None, help="Timestamp for the experiment.")
    root.add_argument("--dry-run", action="store_true", help="Print experiments and exit.")
    root_args, _ = root.parse_known_args()

    # Build the mesh list override if provided
    mesh_ids_override = None
    if root_args.mesh_ids:
        mesh_ids_override = [s for s in re.split(r"[,\s]+", root_args.mesh_ids.strip()) if s]
        print(f"Using mesh IDs override: {mesh_ids_override}")

    if root_args.timestamp:
        timestamp = root_args.timestamp
        DEFAULTS["output"] = f"{ROOT_DIR}/outputs/{timestamp}/"

    if root_args.args_file:
        # Provide DEFAULTS + timestamp to formatting
        merged_defaults = DEFAULTS | {"timestamp": timestamp, "ROOT_DIR": ROOT_DIR}
        arg_lists = load_arg_lists_from_file(root_args.args_file, defaults=merged_defaults, mesh_ids=mesh_ids_override)
        if root_args.dry_run:
            for i, a in enumerate(arg_lists):
                print(f"[{i}] {a}")
            sys.exit(0)
    else:
        raise ValueError("Please provide an --args-file with experiment templates.")

    copy_script(arg_lists)

    for arg_list in arg_lists:
        process_single_mesh(arg_list)
