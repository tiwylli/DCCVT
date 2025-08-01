# this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
import os
import sys
import argparse
import tqdm as tqdm
from time import time
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import kaolin


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
torch.manual_seed(69)


# Generate a timestamp string for unique output folders
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# timestamp = "20250730_125341"
# Default parameters for the DCCVT experiments
ROOT_DIR = "/home/wylliam/dev/Kyushu_experiments"
DEFAULTS = {
    "output": f"{ROOT_DIR}/outputs/{timestamp}/",
    "mesh": f"{ROOT_DIR}/mesh/thingi32/",
    "trained_HotSpot": f"{ROOT_DIR}/hotspots_model/",
    "input_dims": 3,
    "num_iterations": 1000,
    "num_centroids": 16,  # ** input_dims
    "sample_near": 0,  # 32 # ** input_dims
    "target_size": 32,  # 32 # ** input_dims
    "clip": False,
    "marching_tetrahedra": False,  # True
    # "build_mesh": False,
    "w_cvt": 0,
    "w_sdfsmooth": 0,
    "w_voroloss": 0,  # 1000
    "w_chamfer": 0,  # 1000
    "w_mt": 0,  # 1000
    "w_mc": 0,  # 1000
    # "w_bpa": 0,  # 1000
    "upsampling": 0,  # 0
    "lr_sites": 0.0005,
    "mesh_ids": [
        "252119",
        "313444",
        "316358",
        "354371",
        "398259",
        "441708",
        "44234",
        "47984",
        "527631",
        "53159",
        "58168",
        "64444",
        "64764",
        "68380",
        "68381",
        "72870",
        "72960",
        "73075",
        "75496",
        "75655",
        "75656",
        "75662",
        "75665",
        "76277",
        "77245",
        "78671",
        "79241",
        "90889",
        "92763",
        "92880",
        "95444",
        "96481",
    ],
}


# m_list=["gargoyle", "chair", "bunny"]
# ["gargoyle", "gargoyle_unconverged", "bunny", "chair"]
def build_arg_list(m_list=DEFAULTS["mesh_ids"]):
    arg_list = []
    for m in m_list:
        # # Voroloss vs DCCVT vs MT : baseline
        # # Voroloss
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_voroloss",
        #         "1000",
        #         "--num_centroids",
        #         "32",
        #     ]
        # )
        # DCCVT+cvt+sdfreg
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--w_sdfsmooth",
                "100",
                "--num_centroids",
                "32",
                "--clip",
                "--w_mt",
                "1",
                "--w_mc",
                "1",
            ]
        )
        # DCCVT + cvt + sdfreg + upsampling
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--w_sdfsmooth",
                "100",
                "--upsampling",
                "10",
                "--clip",
                "--w_mt",
                "1",
                "--w_mc",
                "1",
            ]
        )
        # # MT
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--upsampling",
        #         "0",
        #         "--clip",
        #         "--marching_tetrahedra",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # # MT + upsampling
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--upsampling",
        #         "10",
        #         "--clip",
        #         "--marching_tetrahedra",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )

        # # Voroloss vs DCCVT vs Marching Tetrahedra : unconverged
        # # Voroloss
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_voroloss",
        #         "1000",
        #         "--num_centroids",
        #         "32",
        #     ]
        # )
        # DCCVT+cvt+sdfreg
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
                "--output",
                f"{DEFAULTS['output']}unconverged_{m}",
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--w_sdfsmooth",
                "100",
                "--num_centroids",
                "32",
                "--clip",
                "--w_mt",
                "1",
                "--w_mc",
                "1",
            ]
        )

        # DCCVT+cvt+sdfreg+upsampling
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
                "--output",
                f"{DEFAULTS['output']}unconverged_{m}",
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--w_sdfsmooth",
                "100",
                "--upsampling",
                "10",
                "--clip",
                "--w_mt",
                "1",
                "--w_mc",
                "1",
            ]
        )
        # # MT
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--upsampling",
        #         "0",
        #         "--clip",
        #         "--marching_tetrahedra",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # # MT+upsampling
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--upsampling",
        #         "10",
        #         "--clip",
        #         "--marching_tetrahedra",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )

        # Ablation study:
        # DCCVT + cvt + sdfreg, already done above
        # DCCVT + cvt
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--w_cvt",
        #         "100",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--w_cvt",
        #         "100",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # # DCCVT + sdfreg
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--w_sdfsmooth",
        #         "100",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--w_sdfsmooth",
        #         "100",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # # DCCVT + nothing
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32/{m}.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )
        # arg_list.append(
        #     [
        #         "--mesh",
        #         f"{DEFAULTS['mesh']}{m}",
        #         "--trained_HotSpot",
        #         f"{DEFAULTS['trained_HotSpot']}thingi32_unconverged/{m}_500.pth",
        #         "--output",
        #         f"{DEFAULTS['output']}unconverged_{m}",
        #         "--w_chamfer",
        #         "1000",
        #         "--num_centroids",
        #         "32",
        #         "--clip",
        #         "--w_mt",
        #         "1",
        #         "--w_mc",
        #         "1",
        #     ]
        # )

    return arg_list


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
        "--marching_tetrahedra",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["marching_tetrahedra"],
        help="Enable/disable marching_tetrahedra",
    )

    parser.add_argument(
        "--build_mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable build mesh",
    )
    parser.add_argument("--w_cvt", type=float, default=DEFAULTS["w_cvt"], help="Weight for CVT regularization")
    parser.add_argument("--w_sdfsmooth", type=float, default=DEFAULTS["w_sdfsmooth"], help="Weight for SDF smoothing")
    parser.add_argument("--w_voroloss", type=float, default=DEFAULTS["w_voroloss"], help="Weight for Voronoi loss")
    parser.add_argument(
        "--w_chamfer", type=float, default=DEFAULTS["w_chamfer"], help="Weight for Chamfer distance on points"
    )
    # parser.add_argument("--w_bpa", type=float, default=DEFAULTS.get("w_bpa", 0), help="flag to use BPA instead of DCCVT")
    parser.add_argument("--w_mc", type=float, default=DEFAULTS["w_mc"], help="Weight for MC loss")
    parser.add_argument("--w_mt", type=float, default=DEFAULTS["w_mt"], help="Weight for MT loss")
    parser.add_argument("--upsampling", type=int, default=DEFAULTS["upsampling"], help="Upsampling factor")
    parser.add_argument("--lr_sites", type=float, default=DEFAULTS["lr_sites"], help="Learning rate for sites")
    parser.add_argument(
        "--save_path", type=str, default=None, help="(optional) full save path; if omitted, computed from other flags"
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
        num_samples = sample_near**input_dims - num_centroids**input_dims
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


def train_DCCVT(sites, sites_sdf, target_pc, args):
    if args.w_chamfer > 0:
        optimizer = torch.optim.Adam(
            [
                {"params": [sites], "lr": args.lr_sites},
                {"params": [sites_sdf], "lr": args.lr_sites},
            ]
        )
    else:
        optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])
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
                f_vect = faces_list[0]
            else:
                v_vect, f_vect, sites_sdf_grads, _, _ = su.get_clipped_mesh_numba(
                    sites, None, d3dsimplices, args.clip, sites_sdf, args.build_mesh
                )
            if args.build_mesh:
                triangle_faces = [[f[0], f[i], f[i + 1]] for f in f_vect for i in range(1, len(f) - 1)]
                triangle_faces = torch.tensor(triangle_faces, device=device)
                hs_p = su.sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=mnfld_points.shape[0])
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), hs_p.unsqueeze(0))
            else:
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), v_vect.unsqueeze(0))

        if args.w_voroloss > 0:
            voroloss_loss = voroloss(target_pc.squeeze(0), sites).mean()

        if args.w_cvt > 0:
            cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)

        sites_loss = args.w_cvt * cvt_loss + args.w_chamfer * chamfer_loss_mesh + args.w_voroloss * voroloss_loss

        if args.w_sdfsmooth > 0:
            if sites_sdf_grads is None:
                sites_sdf_grads = su.sdf_space_grad_pytorch_diego(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach()
                )
            eik_loss = args.w_sdfsmooth / 10 * lf.discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
            shl = args.w_sdfsmooth / 0.1 * lf.smoothed_heaviside_loss(sites, sites_sdf, sites_sdf_grads, d3dsimplices)
            sdf_loss = eik_loss + shl

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

            if args.w_chamfer > 0:
                sites, sites_sdf = su.upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = sites_sdf.detach().requires_grad_(True)
                optimizer = torch.optim.Adam(
                    [
                        {"params": [sites], "lr": args.lr_sites},
                        {"params": [sites_sdf], "lr": args.lr_sites},
                    ]
                )
            else:
                sites, _ = su.upsampling_adaptive_vectorized_sites_sites_sdf(sites, d3dsimplices, sites_sdf)
                sites = sites.detach().requires_grad_(True)
                optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])
            upsampled += 1.0
            print("sites length AFTER: ", len(sites))
    return sites, sites_sdf


def extract_mesh(sites, model, target_pc, time, args, state="", d3dsimplices=None, t=time()):
    print(f"Extracting mesh at state: {state} with upsampling: {args.upsampling}")
    # SDF at original sites
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        print("Using SDFGrid model")
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        print("Using Tensor model")
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        print("Using nn.Module / callable model")
        sdf_values = model(sites).detach()

    sdf_values = sdf_values.squeeze()  # (N,)

    if d3dsimplices is None:
        sites_np = sites.detach().cpu().numpy()
        if args.marching_tetrahedra:
            d3dsimplices = Delaunay(sites_np).simplices
        else:
            d3dsimplices, _ = pygdel3d.triangulate(sites_np)
            d3dsimplices = np.array(d3dsimplices)

    if args.w_chamfer > 0:
        v_vect, f_vect, _, _, _ = su.get_clipped_mesh_numba(sites, None, d3dsimplices, args.clip, sdf_values, True)
        if args.marching_tetrahedra:
            output_obj_file = f"{args.save_path}/marching_tetrahedra_{args.upsampling}_{state}_DCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        else:
            output_obj_file = f"{args.save_path}/DCCVT_{args.upsampling}_{state}_DCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
        save_npz(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        su.save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        su.save_target_pc_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())
    if args.w_voroloss > 0:
        v_vect, f_vect, _, _, _ = su.get_clipped_mesh_numba(sites, None, d3dsimplices, args.clip, sdf_values, True)
        output_obj_file = f"{args.save_path}/voromesh_{args.upsampling}_{state}_DCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
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
        args=args,
    )


if __name__ == "__main__":
    arg_lists = build_arg_list()
    start_time = time()
    for arg_list in arg_lists:
        args = define_options_parser(arg_list)
        args.save_path = f"{args.output}" if args.save_path is None else args.save_path
        os.makedirs(args.save_path, exist_ok=True)

        if not os.path.exists(f"{args.save_path}/marching_tetrahedra_{args.upsampling}_final_MT.obj"):
            print("args: ", args)
            model, mnfld_points = load_model(args.mesh, args.target_size, args.trained_HotSpot)
            sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

            if args.w_chamfer > 0:
                sdf = init_sdf(model, sites)
            else:
                sdf = model

            # Extract the initial mesh

            extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

            if args.w_chamfer > 0 or args.w_voroloss > 0:
                t0 = time() - start_time
                sites, sdf = train_DCCVT(sites, sdf, mnfld_points, args)
                ti = time() - t0 - start_time

            # Extract the final mesh
            extract_mesh(sites, sdf, mnfld_points, ti, args, state="final")

        # reset everything for the next iteration
        for var_name in ["sites", "sites_sdf", "model", "mnfld_points"]:
            if var_name in locals() and locals()[var_name] is not None:
                del locals()[var_name]
        torch.cuda.empty_cache()
