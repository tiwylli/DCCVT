# this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
import os
import sys
import argparse
import tqdm as tqdm
from time import time
import torch
import numpy as np
import polyscope as ps
import kaolin
import shlex
import re
import gudhi
import trimesh
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import pygdel3d
from scipy.spatial import Delaunay
import sdfpred_utils.sdfpred_utils as su
import sdfpred_utils.loss_functions as lf
from pytorch3d.loss import chamfer_distance
import argparse

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
import datetime

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
    "visualize": False,
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

_PS_INITIALIZED = False


def ensure_polyscope_initialized():
    global _PS_INITIALIZED
    if not _PS_INITIALIZED:
        ps.init()
        _PS_INITIALIZED = True

import os, re, shlex


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
    parser.add_argument(
        "--clip", action=argparse.BooleanOptionalAction, default=DEFAULTS["clip"], help="Enable/disable clipping"
    )
    parser.add_argument(
        "--grad_interpol",
        type=str,
        default=DEFAULTS["grad_interpol"],
        help="Gradient interpolation method: robust, hybrid, barycentric",
    )
    parser.add_argument(
        "--marching_tetrahedra",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["marching_tetrahedra"],
        help="Enable/disable marching_tetrahedra",
    )
    parser.add_argument(
        "--true_cvt",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["true_cvt"],
        help="Enable/disable true CVT loss",
    )
    parser.add_argument(
        "--extract_optim",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["extract_optim"],
        help="Enable/disable extraction optimization",
    )
    parser.add_argument(
        "--sdf_type",
        type=str,
        default=DEFAULTS["sdf_type"],
        help="SDF type: hotspot, sphere, complex_alpha",
    )
    parser.add_argument(
        "--no_mp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["no_mp"],
        help="Enable/disable multiprocessing",
    )
    parser.add_argument(
        "--ups_extraction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["ups_extraction"],
        help="Enable/disable upsampling extraction",
    )
    parser.add_argument(
        "--build_mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable build mesh",
    )
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable video output"
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["visualize"],
        help="Enable/disable Polyscope visualization",
    )
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


def train_DCCVT(sites, sites_sdf, mnfld_points, hotspot_model, args):
    if args.w_chamfer > 0:
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
        if sites_sdf is None:
            raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
        if sites_sdf.__class__.__name__ == "SDFGrid":
            sdf_values = sites_sdf.sdf(sites)
        elif isinstance(sites_sdf, torch.Tensor):
            sdf_values = sites_sdf.to(device)
        else:  # nn.Module / callable
            sdf_values = sites_sdf(sites).detach()
        sdf_values = sdf_values.squeeze()  # (N,)
        sites_sdf = sdf_values.requires_grad_()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

    upsampled = 0.0
    epoch = 0
    t0 = time()
    cvt_loss = 0
    chamfer_loss_mesh = 0
    voroloss_loss = 0
    sdf_loss = 0
    d3dsimplices = None
    sites_sdf_grads = None
    voroloss = lf.Voroloss_opt().to(device)

    from tqdm import tqdm

    for epoch in tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        if args.w_cvt > 0 or args.w_chamfer > 0:
            sites_np = sites.detach().cpu().numpy()
            if args.marching_tetrahedra:
                d3dsimplices = Delaunay(sites_np).simplices

            else:
                d3dsimplices, _ = pygdel3d.triangulate(sites_np)
                d3dsimplices = np.array(d3dsimplices)

        if args.w_chamfer > 0:
            if args.marching_tetrahedra:
                d3dsimplices = torch.tensor(d3dsimplices, device=device)
                marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
                    sites.unsqueeze(0), d3dsimplices, sites_sdf.unsqueeze(0), return_tet_idx=False
                )
                vertices_list, faces_list = marching_tetrehedra_mesh
                v_vect = vertices_list[0]
                # f_or_clipped_v = faces_list[0]
                _, f_or_clipped_v, _, _, _ = su.get_clipped_mesh_numba(
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
                    v_vect, f_or_clipped_v = su.cvt_extraction(sites, sites_sdf, d3dsimplices, False)
                    sites_sdf_grads = None
                else:
                    v_vect, f_or_clipped_v, sites_sdf_grads, tet_probs, W = su.get_clipped_mesh_numba(
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
                hs_p = su.sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=mnfld_points.shape[0])
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), hs_p.unsqueeze(0))
            else:
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), v_vect.unsqueeze(0))

        if args.w_voroloss > 0:
            voroloss_loss = voroloss(mnfld_points.squeeze(0), sites).mean()

        if args.w_cvt > 0:
            if args.w_voroloss > 0:
                cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
            else:
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay_volume(sites, None, d3dsimplices)
                if args.true_cvt:
                    cvt_loss = lf.compute_cvt_loss_true(sites, d3dsimplices, f_or_clipped_v)
                else:
                    cvt_loss = lf.compute_cvt_loss_CLIPPED_vertices(sites, None, None, d3dsimplices, f_or_clipped_v)

        sites_loss = args.w_cvt / 1 * cvt_loss + args.w_chamfer * chamfer_loss_mesh + args.w_voroloss * voroloss_loss

        if args.w_sdfsmooth > 0:
            if sites_sdf_grads is None:
                sites_sdf_grads, tets_sdf_grads, W = su.sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach()
                )
            if epoch % 100 == 0 and epoch <= 500:
                eps_H = lf.estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            elif epoch % 100 == 0 and epoch <= 800:
                eps_H = lf.estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 2).detach()
                print("Estimated eps_H: ", eps_H)

            # eik_loss = args.w_sdfsmooth / 1000 * lf.tet_sdf_grad_eikonal_loss(sites, tets_sdf_grads, d3dsimplices)
            eik_loss = args.w_sdfsmooth / 10 * lf.discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
            shl = args.w_sdfsmooth * lf.tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, d3dsimplices, eps_H)
            sdf_loss = eik_loss + shl

        if args.w_vertex_sdf_interpolation > 0:
            steps_verts = tet_probs[1]
            # all_vor_vertices = su.compute_vertices_3d_vectorized(sites, d3dsimplices)  # (M,3)
            # vertices_sdf = su.interpolate_sdf_of_vertices(all_vor_vertices, d3dsimplices, sites, sites_sdf)
            # _, _, used_tet = su.compute_zero_crossing_vertices_3d(sites, None, None, d3dsimplices, sites_sdf)

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

                if args.w_chamfer > 0:
                    sites_sdf = sites_sdf.detach().requires_grad_(True)
                    optimizer = torch.optim.Adam(
                        [
                            {"params": [sites], "lr": args.lr_sites},
                            {"params": [sites_sdf], "lr": args.lr_sites},
                        ]
                    )
                    eps_H = lf.estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 3).detach()
                    print("Estimated eps_H: ", eps_H)
                else:
                    optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])

                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                continue

            if d3dsimplices is None:
                if args.marching_tetrahedra:
                    d3dsimplices = Delaunay(sites_np).simplices
                else:
                    d3dsimplices, _ = pygdel3d.triangulate(sites_np)
                    d3dsimplices = np.array(d3dsimplices)

            if sites_sdf_grads is None or sites_sdf_grads.shape[0] != sites_sdf.shape[0]:
                sites_sdf_grads, tets_sdf_grads, W = su.sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )

            if args.w_chamfer > 0:
                sites, sites_sdf = su.upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads, ups_method=args.ups_method, score=args.score
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = sites_sdf.detach().requires_grad_(True)

                if args.marching_tetrahedra:
                    d3dsimplices = Delaunay(sites.detach().cpu().numpy()).simplices
                else:
                    d3dsimplices, _ = pygdel3d.triangulate(sites.detach().cpu().numpy())
                    d3dsimplices = np.array(d3dsimplices)

                optimizer = torch.optim.Adam(
                    [
                        {"params": [sites], "lr": args.lr_sites},
                        {"params": [sites_sdf], "lr": args.lr_sites},
                    ]
                )
                eps_H = lf.estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            else:
                sites_sdf = hotspot_model(sites).squeeze(-1)
                sites_sdf_grads, tets_sdf_grads, W = su.sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )
                sites, sites_sdf = su.upsampling_adaptive_vectorized_sites_sites_sdf(
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


def extract_mesh(sites, model, target_pc, time, args, state="", d3dsimplices=None, t=time()):
    print(f"Extracting mesh at state: {state} with upsampling: {args.upsampling}")
    # SDF at original sites
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        print("Using SDFGrid")
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        print("Using Tensor")
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        print("Using nn.Module / callable model")
        sdf_values = model(sites).detach()

    sdf_values = sdf_values.squeeze()  # (N,)

    # if d3dsimplices is None:
    #     sites_np = sites.detach().cpu().numpy()
    #     if args.marching_tetrahedra:
    #         d3dsimplices = Delaunay(sites_np).simplices
    #     else:
    #         d3dsimplices, _ = pygdel3d.triangulate(sites_np)
    #         d3dsimplices = np.array(d3dsimplices)

    sites_np = sites.detach().cpu().numpy()
    d3dsimplices = Delaunay(sites_np).simplices

    if args.w_chamfer > 0:
        # v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = su.get_clipped_mesh_numba(
        #     sites, None, d3dsimplices, args.clip, sdf_values, True
        # )
        v_vect, f_vect = su.cvt_extraction(sites, sdf_values, d3dsimplices, True)
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_intDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_intDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        su.save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        su.save_target_pc_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())

        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = su.get_clipped_mesh_numba(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_projDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_projDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        su.save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        su.save_target_pc_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())
        if args.visualize:
            try:
                print("printing")
                ensure_polyscope_initialized()
                ps.register_surface_mesh("mesh", v_vect.detach().cpu().numpy(), f_vect)
                ps.show()
            except Exception as exc:
                print(f"Polyscope visualization skipped: {exc}")

    if args.w_voroloss > 0:
        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = su.get_clipped_mesh_numba(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        output_obj_file = f"{args.save_path}/voromesh_{args.num_centroids}_{state}_DCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        su.save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
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
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_MT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_MT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        su.save_obj(output_obj_file, vertices_np, faces_np)


def save_npz(sites, sites_sdf, time, args, output_file):
    np.savez(
        output_file,
        sites=sites.detach().cpu().numpy(),
        sites_sdf=sites_sdf.detach().cpu().numpy(),
        train_time=time,
        args=str(args),
    )


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

    # if not os.path.exists(f"{args.save_path}/marching_tetrahedra_{args.upsampling}_final_MT.obj"):
    output_obj_file = check_if_already_processed(args)
    if os.path.exists(output_obj_file):
        print(f"Skipping already processed mesh: {output_obj_file}")
    else:
        print("args: ", args)
        try:
            model, mnfld_points = load_model(args.mesh, args.target_size, args.trained_HotSpot)
            sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

            if args.w_chamfer > 0:
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

            if args.w_chamfer > 0 or args.w_voroloss > 0:
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


def check_if_already_processed(args):
    state = "final"
    if args.w_chamfer > 0:
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_intDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_intDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_projDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_projDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    if args.w_voroloss > 0:
        output_obj_file = f"{args.save_path}/voromesh_{args.num_centroids}_{state}_DCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    if args.w_mc > 0:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_MT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_MT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    return output_obj_file


if __name__ == "__main__":
    ps.init()
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
