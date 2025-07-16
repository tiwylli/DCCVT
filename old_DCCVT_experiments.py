# this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
import os
import sys
import argparse
import tqdm as tqdm
from time import time
import kaolin
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import diffvoronoi
import sdfpred_utils.sdfpred_utils as su
import sdfpred_utils.loss_functions as lf
from pytorch3d.loss import chamfer_distance

import sys

sys.path.append("3rdparty/HotSpot")
from dataset import shape_3d
import models.Net as Net


# cuda devices
device = torch.device("cuda:0")
print("Using device: ", torch.cuda.get_device_name(device))


DEFAULTS = {
    "output": "/home/wylliam/dev/Kyushu_experiments/outputs/",
    "mesh": "/home/wylliam/dev/Kyushu_experiments/mesh/",
    "trained_HotSpot": "/home/wylliam/dev/Kyushu_experiments/hotspots_model/",
    "input_dims": 3,
    "num_iterations": 100,
    "num_centroids": 8,  # ** input_dims
    "sample_near": 32,  # 32 # ** input_dims
    "clip": False,
    "triangulate": True,
    "w_cvt": 0,  # 100
    "w_sdf_pull": 0,  # 1
    "w_voroloss": 0,  # 1000
    "w_cd_points": 0,  # 1000
    "w_cd_mesh": 0,  # 1000
    "upsampling": 0,  # 0
    "lr_sites": 0.005,
}

import argparse


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
    parser.add_argument(
        "--clip", action=argparse.BooleanOptionalAction, default=DEFAULTS["clip"], help="Enable/disable clipping"
    )
    parser.add_argument(
        "--triangulate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["triangulate"],
        help="Enable/disable triangulation",
    )
    parser.add_argument("--w_cvt", type=float, default=DEFAULTS["w_cvt"], help="Weight for CVT regularization")
    parser.add_argument("--w_sdf_pull", type=float, default=DEFAULTS["w_sdf_pull"], help="Weight for SDF pull loss")
    parser.add_argument("--w_voroloss", type=float, default=DEFAULTS["w_voroloss"], help="Weight for Voronoi loss")
    parser.add_argument(
        "--w_cd_points", type=float, default=DEFAULTS["w_cd_points"], help="Weight for Chamfer distance on points"
    )
    parser.add_argument(
        "--w_cd_mesh", type=float, default=DEFAULTS["w_cd_mesh"], help="Weight for Chamfer distance on mesh"
    )
    parser.add_argument("--upsampling", type=int, default=DEFAULTS["upsampling"], help="Upsampling factor")
    parser.add_argument("--lr_sites", type=float, default=DEFAULTS["lr_sites"], help="Learning rate for sites")
    parser.add_argument(
        "--save_path", type=str, default=None, help="(optional) full save path; if omitted, computed from other flags"
    )
    return parser.parse_args(arg_list)


def load_model(mesh, grid, trained_HotSpot):
    # LOAD MODEL WITH HOTSPOT
    loss_type = "igr_w_heat"
    loss_weights = [350, 0, 0, 1, 0, 0, 20]
    train_set = shape_3d.ReconDataset(
        file_path=mesh + ".ply",
        n_points=grid * grid * 150,  # 15000, #args.n_points,
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
    ######
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
    noise_scale = 0.05
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

    sites = meshgrid.to(device, dtype=torch.float32).requires_grad_(True)
    # add mnfld points with random noise to sites
    N = mnfld_points.squeeze(0).shape[0]
    num_samples = sample_near**input_dims - num_centroids**input_dims
    idx = torch.randint(0, N, (num_samples,))
    sampled = mnfld_points.squeeze(0)[idx]
    perturbed = sampled + (torch.rand_like(sampled) - 0.5) * noise_scale
    sites = torch.cat((sites, perturbed), dim=0)
    # make sites a leaf tensor
    sites = sites.detach().requires_grad_()
    return sites


def train_DCCVT(sites, model, target_pc, args):
    optimizer = torch.optim.Adam(
        [
            {"params": [sites], "lr": args.lr_sites},
            # {'params': model.parameters(), 'lr': lr_model}
        ],
        betas=(0.9, 0.999),
    )
    upsampled = 0.0
    epoch = 0
    t0 = time()
    cvt_loss = 0
    chamfer_loss_points = 0
    chamfer_loss_mesh = 0
    voroloss_loss = 0
    d3dsimplices = None
    voroloss = lf.Voroloss_opt().to(device)

    from tqdm import tqdm

    for epoch in tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        if args.w_sdf_pull > 0:
            for param in model.parameters():
                param.requires_grad = False
            # s1 = torch.mean(model(points)**2)
            # s2 = torch.maximum((model(sites).abs() - 0.05), torch.tensor(0.0)).mean()
            # sdf_loss = 0*s1+s2
            sdf_loss = torch.maximum((model(sites).abs() - 0.05), torch.tensor(0.0)).mean()
            sdf_loss.backward(retain_graph=True)
            for param in model.parameters():
                param.requires_grad = True

        if args.w_cvt > 0 or args.w_cd_points > 0 or args.w_cd_mesh > 0:
            sites_np = sites.detach().cpu().numpy()
            d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(args.input_dims * sites_np.shape[0]))
            d3dsimplices = np.array(d3dsimplices)

            if args.w_cd_points > 0:
                vertices_to_compute, bisectors_to_compute = su.compute_zero_crossing_vertices_3d(
                    sites, None, None, d3dsimplices, model
                )
                vertices = su.compute_vertices_3d_vectorized(sites, vertices_to_compute)
                bisectors = su.compute_all_bisectors_vectorized(sites, bisectors_to_compute)
                points = torch.cat((vertices, bisectors), 0)

        if args.w_cvt > 0:
            cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)

        if args.w_cd_points > 0:
            chamfer_loss_points, _ = chamfer_distance(target_pc.detach(), points.unsqueeze(0))

        if args.w_cd_mesh > 0:
            v_vect, f_vect = su.get_clipped_mesh_numba(sites, model, d3dsimplices, args.clip)

            if args.triangulate:
                triangle_faces = [[f[0], f[i], f[i + 1]] for f in f_vect for i in range(1, len(f) - 1)]
                triangle_faces = torch.tensor(triangle_faces, device=device)
                hs_p = su.sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=2 * args.sample_near * 150)
            else:
                hs_p = su.sample_mesh_points_heitz(v_vect, f_vect, num_samples=2 * args.sample_near * 150)

            chamfer_loss_mesh, _ = chamfer_distance(target_pc.detach(), hs_p.unsqueeze(0))

        if args.w_voroloss > 0:
            voroloss_loss = voroloss(target_pc.squeeze(0), sites).mean()

        sites_loss = (
            args.w_cvt * cvt_loss
            + args.w_cd_mesh * chamfer_loss_mesh
            + args.w_cd_points * chamfer_loss_points
            + args.w_voroloss * voroloss_loss
        )

        loss = sites_loss
        # print(f"Epoch {epoch}: loss = {loss.item()}")
        loss.backward()
        # print("-----------------")

        optimizer.step()

        # if epoch>100 and (epoch // 100) == upsampled+1 and loss.item() < 0.5 and upsampled < upsampling:
        if epoch / args.num_iterations > (upsampled + 1) / (args.upsampling + 1) and upsampled < args.upsampling:
            print("sites length BEFORE UPSAMPLING: ", len(sites))
            sites = su.upsampling_vectorized(sites, tri=None, vor=None, simplices=d3dsimplices, model=model)
            sites = sites.detach().requires_grad_(True)
            optimizer = torch.optim.Adam(
                [
                    {"params": [sites], "lr": args.lr_sites},
                    # {'params': model.parameters(), 'lr': lr_model}
                ]
            )
            upsampled += 1.0
            print("sites length AFTER: ", len(sites))
    return sites


def output_npz(sites, model, target_pc, args, state="", d3dsimplices=None, t0=time()):
    train_time = time() - t0
    # #Export the sites, their sdf values, the gradients of the sdf values and the hessian
    sdf_values = model(sites)
    # sdf_gradients = torch.autograd.grad(outputs=sdf_values, inputs=sites, grad_outputs=torch.ones_like(sdf_values), create_graph=True, retain_graph=True,)[0] # (N, 3)
    # N, D = sites.shape
    # hess_sdf = torch.zeros(N, D, D, device=sites.device)
    # for i in range(D):
    #     grad2 = torch.autograd.grad(outputs=sdf_gradients[:, i], inputs=sites, grad_outputs=torch.ones_like(sdf_gradients[:, i]), create_graph=False, retain_graph=True,)[0] # (N, 3)
    #     hess_sdf[:, i, :] = grad2 # fill row i of each 3×3 Hessian

    v_vect, f_vect = su.get_clipped_mesh_numba(sites, model, d3dsimplices, args.clip)

    if args.triangulate:
        f_vect = [[f[0], f[i], f[i + 1]] for f in f_vect for i in range(1, len(f) - 1)]

    # Compute metrics on mesh : CD and F1
    hs_p = su.sample_mesh_points_heitz(v_vect, torch.tensor(f_vect), num_samples=2 * args.sample_near * 150)
    chamfer_loss_mesh, _ = chamfer_distance(target_pc.detach(), hs_p.unsqueeze(0))
    print("Chamfer loss mesh: ", chamfer_loss_mesh.item())
    # Compute F1 score
    dmat = torch.cdist(hs_p, target_pc.squeeze(0))  # (N, M)
    d_rec2gt, idx_rec2gt = dmat.min(dim=1)  # for precision
    d_gt2rec, _ = dmat.min(dim=0)  # for recall
    tau = 0.003
    P = (d_rec2gt <= tau).float().mean()
    R = (d_gt2rec <= tau).float().mean()
    F1 = 2 * P * R / (P + R + 1e-8)
    # Compute normal consistency

    s = f"{args.save_path}_{state}.npz"
    print("Saving to: ", s)
    np.savez(
        s,
        sites=sites.detach().cpu().numpy(),
        sdf_values=sdf_values.detach().cpu().numpy(),
        # sdf_gradients=sdf_gradients.detach().cpu().numpy(),
        # sdf_hessians=hess_sdf.detach().cpu().numpy(),
        v_vect=v_vect.detach().cpu().numpy(),
        f_vect=f_vect,
        train_time=train_time,
        grads_mesh_extraction_time=time() - t0 - train_time,
        chamfer_loss_mesh=chamfer_loss_mesh.item(),
        F1=F1.item(),
    )


def output_image_polyscope(state=""):
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")

    # Load mesh from args.save_path
    data = np.load(f"{args.save_path}_{state}.npz", allow_pickle=True)
    v_vect = data["v_vect"]
    f_vect = data["f_vect"]

    ps.register_surface_mesh("mesh", v_vect, f_vect, back_face_policy="identical")
    ps.screenshot(args.save_path + f"_{state}.png", transparent_bg=True)
    ps.unshow()


def build_arg_list(m_list=["gargoyle", "chair", "bunny"]):
    arg_list = []
    for m in m_list:
        # Voroloss, Voroloss + CVT, Voroloss + CVT + Clipping,
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_voroloss",
                "1000",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cvt",
                "100",
                "--w_voroloss",
                "1000",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cvt",
                "100",
                "--w_voroloss",
                "1000",
            ]
        )
        # Vertex + Bisect, Vertex + Bisect + Clip, Vertex + Bisect + CVT, Vertex + Bisect + CVT + Clip
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cvt",
                "100",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cvt",
                "100",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        # Sampling, Sampling + CVT, Sampling + Clip, Sampling + CVT + Clip
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cd_mesh",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cvt",
                "100",
                "--w_cd_mesh",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cd_mesh",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cvt",
                "100",
                "--w_cd_mesh",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        # cdp + cdm, cdp + cdm + CVT, cdp + cdm + Clip, cdp + cdm + CVT + Clip
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cd_mesh",
                "1000",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_cvt",
                "100",
                "--w_cd_mesh",
                "1000",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cd_mesh",
                "1000",
                "--w_cd_points",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--clip",
                "--w_cvt",
                "100",
                "--w_cd_points",
                "1000",
                "--w_cd_mesh",
                "1000",
                "--w_sdf_pull",
                "1",
            ]
        )
    return arg_list


def crop_transparent(img, tol=0.0):
    """
    Crop off outer rows/cols where the alpha channel is nearly zero.
    img: H×W×4 array (RGBA) with alpha in [0–1] or [0–255].
    tol: minimum alpha to treat as “foreground” (default 0).
    """
    # make sure alpha is float in [0–1]
    alpha = img[..., 3]
    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float32) / 255.0

    # mask of non-transparent pixels
    mask = alpha > tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # nothing non-transparent

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def output_results_figure():
    # collect and group .npz files by experiment prefix
    print(args.output)
    files = [f for f in os.listdir(args.output) if f.endswith(".npz")]
    groups = {}
    for f in files:
        if f.endswith("_init.npz"):
            prefix = f[:-9]  # strip "_init.npz"
            groups.setdefault(prefix, {})["init"] = f
        elif f.endswith("_final.npz"):
            prefix = f[:-10]  # strip "_final.npz"
            groups.setdefault(prefix, {})["final"] = f

    prefixes = sorted(groups.keys())
    n = len(prefixes)
    total_rows = n + 1  # +1 for our header

    fig, axs = plt.subplots(
        total_rows,
        3,
        figsize=(15, 5 * total_rows),
        gridspec_kw={"height_ratios": [0.5] + [1] * n},  # make header a bit shorter
        constrained_layout=True,
    )

    # If there's only one data‐row, axs will be 2D but with shape (2,3); no change needed.
    # -------------------------------------------------------------------------
    # 1) Header row: turn off all three axes, then place text in the center one
    header_text = (
        f"All results are for {args.num_iterations} iterations and {args.sample_near**args.input_dims} sites\n"
        "cdp = chamfer distance vertices+bisectors to target point\n"
        "cdm = Chamfer distance sampled mesh to target point\n"
        "v = voroloss"
    )
    for col in range(3):
        axs[0, col].axis("off")
    axs[0, 1].text(0.5, 0.5, header_text, ha="center", va="center", fontsize=16, family="monospace")

    # -------------------------------------------------------------------------
    # 2) Data rows
    for i, prefix in enumerate(prefixes, start=1):
        grp = groups[prefix]

        # --- init image ---
        ax = axs[i, 0]
        init_png = grp.get("init", "").replace(".npz", ".png")
        img_init = plt.imread(os.path.join(args.output, init_png))
        trimmed_init = crop_transparent(img_init, tol=1e-3)
        ax.imshow(trimmed_init)
        ax.set_title("Init")
        ax.axis("off")

        # --- final image ---
        ax = axs[i, 1]
        final_png = grp.get("final", "").replace(".npz", ".png")
        img_final = plt.imread(os.path.join(args.output, final_png))
        trimmed_final = crop_transparent(img_final, tol=1e-3)
        ax.imshow(trimmed_final)
        ax.set_title("Final")
        ax.axis("off")

        # --- metrics ---
        ax = axs[i, 2]
        data = np.load(os.path.join(args.output, grp["final"]))
        text = (
            f"Chamfer loss mesh 10⁻⁵: {data['chamfer_loss_mesh'] * 1e4:.4f}\n"
            f"F1 score:               {data['F1']:.4f}\n"
            f"Train time (s):         {data['train_time']:.4f}\n"
            f"Mesh extraction (s):    {data['grads_mesh_extraction_time']:.4f}"
        )
        ax.text(0, 0.5, text, va="center", fontsize=18, family="monospace")
        ax.set_title(prefix, fontsize=20)
        ax.axis("off")

    # -------------------------------------------------------------------------
    plt.savefig(
        os.path.join(args.output, f"results{args.num_iterations}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    print(os.path.join(args.output, f"results{args.num_iterations}.png"))


def open_everything_polyscope():
    # open polyscope with all the meshes
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    # ps.set_background_color((1, 1, 1))
    # remove groud plane
    ps.set_ground_plane_mode("none")

    # Load mesh from args.save_path
    files = os.listdir(args.output)
    files = [f for f in files if f.endswith(".npz")]

    for f in files:
        data = np.load(args.output + "/" + f, allow_pickle=True)
        v_vect = data["v_vect"]
        f_vect = data["f_vect"]
        ps_mesh = ps.register_surface_mesh(f[:-4], v_vect, f_vect, back_face_policy="identical")

    ps.show()


if __name__ == "__main__":
    arg_lists = build_arg_list()
    for arg_list in arg_lists:
        args = define_options_parser(arg_list)
        args.save_path = (
            args.output
            + f"/cdp{int(args.w_cd_points)}_cdm{int(args.w_cd_mesh)}_v{int(args.w_voroloss)}_cvt{int(args.w_cvt)}_clip{args.clip}"
        )
        if os.path.exists(args.save_path + "_final" + ".npz"):
            print("File already exists, skipping...")
            continue
        print("args: ", args)
        model, mnfld_points = load_model(args.mesh, args.sample_near, args.trained_HotSpot)
        sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

        output_npz(sites, model, mnfld_points, args, "init")
        output_image_polyscope("init")

        sites = train_DCCVT(sites, model, mnfld_points, args)

        output_npz(sites, model, mnfld_points, args, "final")
        output_image_polyscope("final")

        # reset everything for the next iteration
        del sites
        del model
        del mnfld_points
        torch.cuda.empty_cache()

    output_results_figure()
    # open_everything_polyscope()
