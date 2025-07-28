# this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
import os
import sys
import argparse
import tqdm as tqdm
from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import pygdel3d
import sdfpred_utils.sdfpred_utils as su
import sdfpred_utils.loss_functions as lf
from pytorch3d.loss import chamfer_distance


sys.path.append("3rdparty/HotSpot")
from dataset import shape_3d
import models.Net as Net


# cuda devices
device = torch.device("cuda:0")
print("Using device: ", torch.cuda.get_device_name(device))
torch.manual_seed(69)

DEFAULTS = {
    "output": "/home/wylliam/dev/Kyushu_experiments/outputs/",
    "mesh": "/home/wylliam/dev/Kyushu_experiments/mesh/",
    "trained_HotSpot": "/home/wylliam/dev/Kyushu_experiments/hotspots_model/",
    "input_dims": 3,
    "num_iterations": 1000,
    "num_centroids": 16,  # ** input_dims
    "sample_near": 0,  # 32 # ** input_dims
    "target_size": 32,  # 32 # ** input_dims
    "clip": False,
    "build_mesh": False,
    "w_cvt": 0,  # 10
    "w_voroloss": 0,  # 1000
    "w_chamfer": 0,  # 1000
    "w_bpa": 0,  # 1000
    "upsampling": 0,  # 0
    "lr_sites": 0.0005,
}

# if user name is beltegeuse, use the defaults
if os.getenv("USER") == "beltegeuse":
    DEFAULTS["output"] = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments/outputs/"
    DEFAULTS["mesh"] = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments/mesh/"
    DEFAULTS["trained_HotSpot"] = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments/hotspots_model/"


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
        "--build_mesh",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["build_mesh"],
        help="Enable/disable build mesh",
    )
    parser.add_argument("--w_cvt", type=float, default=DEFAULTS["w_cvt"], help="Weight for CVT regularization")
    parser.add_argument("--w_voroloss", type=float, default=DEFAULTS["w_voroloss"], help="Weight for Voronoi loss")
    parser.add_argument(
        "--w_chamfer", type=float, default=DEFAULTS["w_chamfer"], help="Weight for Chamfer distance on points"
    )
    parser.add_argument("--w_bpa", type=float, default=DEFAULTS["w_bpa"], help="flag to use BPA instead of DCCVT")
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
    voroloss = lf.Voroloss_opt().to(device)

    for epoch in tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        if args.w_cvt > 0 or args.w_chamfer > 0:
            sites_np = sites.detach().cpu().numpy()
            # d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(args.input_dims * sites_np.shape[0]))
            d3dsimplices, _ = pygdel3d.triangulate(sites_np)
            d3dsimplices = np.array(d3dsimplices)

            # d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(args.input_dims * sites_np.shape[0]))
            # d3dsimplices = np.array(d3dsimplices)

        if args.w_cvt > 0:
            cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
            sites_sdf_grads = su.sdf_space_grad_pytorch_diego(
                sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach()
            )
            eik_loss = args.w_cvt / 10 * lf.discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
            shl = args.w_cvt / 0.1 * lf.smoothed_heaviside_loss(sites, sites_sdf, sites_sdf_grads, d3dsimplices)
            sdf_loss = eik_loss + shl

        if args.w_chamfer > 0:
            v_vect, f_vect, sdf_verts, sdf_verts_grads, _ = su.get_clipped_mesh_numba(
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

        sites_loss = args.w_cvt * cvt_loss + args.w_chamfer * chamfer_loss_mesh + args.w_voroloss * voroloss_loss

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
                # d3dsimplices = diffvoronoi.get_delaunay_simplices(sites.detach().cpu().numpy().reshape(-1))
                d3dsimplices, _ = pygdel3d.triangulate(sites_np)
                d3dsimplices = np.array(d3dsimplices)
                # Convert to int64
                # d3dsimplices = d3dsimplices.astype(np.int64)

            if args.w_chamfer > 0:
                sites, sites_sdf = su.upsampling_adaptive_vectorized_sites_sites_sdf(sites, d3dsimplices, sites_sdf)
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


def output_npz(sites, model, target_pc, args, state="", d3dsimplices=None, t=time()):
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

    v_vect, f_vect, _, _, _ = su.get_clipped_mesh_numba(sites, None, d3dsimplices, args.clip, sdf_values, True)
    # Compute metrics on mesh : CD and F1
    # hs_p = su.sample_mesh_points_heitz(v_vect, torch.tensor(f_vect), num_samples=target_pc.shape[0])
    # chamfer_loss_mesh, _ = chamfer_distance(target_pc.detach(), hs_p.unsqueeze(0))

    output_obj_file = f"{args.save_path}_{state}.obj"

    su.save_obj(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
    su.save_target_pc_ply(f"{args.save_path}_target.ply", target_pc.squeeze(0).detach().cpu().numpy())

    ours_pts, _ = su.sample_points_on_mesh(output_obj_file, n_points=args.target_size * args.target_size * 150)
    gt_pts, _ = su.sample_points_on_mesh(args.mesh + ".obj", n_points=args.target_size * args.target_size * 150)

    accuracy, completeness, chamfer, precision, recall, f1 = su.chamfer_accuracy_completeness_f1(ours_pts, gt_pts)

    print(f"Chamfer Accuracy (Ours → GT): {accuracy:.6f}")
    print(f"Chamfer Completeness (GT → Ours): {completeness:.6f}")
    print(f"Chamfer Distance (symmetric): {chamfer:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    f_vect = [[f[0], f[i], f[i + 1]] for f in f_vect for i in range(1, len(f) - 1)]

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
        train_time=t,
        # grads_mesh_extraction_time=time() - t0 - train_time,
        accuracy=accuracy,
        completeness=completeness,
        chamfer=chamfer,
        precision=precision,
        recall=recall,
        f1=f1,
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


# m_list=["gargoyle", "chair", "bunny"]
def build_arg_list(m_list=["gargoyle", "gargoyle_unconverged", "bunny", "chair"]):
    arg_list = []
    for m in m_list:
        # Voroloss vs DCCVT : baseline
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
                "--num_centroids",
                "32",
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
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--num_centroids",
                "32",
                "--clip",
            ]
        )
        # Voroloss vs DCCVT : upsampling
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
                "--upsampling",
                "10",
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
                "--w_chamfer",
                "1000",
                "--w_cvt",
                "100",
                "--upsampling",
                "10",
                "--clip",
            ]
        )

        # BPA
        arg_list.append(
            [
                "--mesh",
                f"{DEFAULTS['mesh']}{m}",
                "--trained_HotSpot",
                f"{DEFAULTS['trained_HotSpot']}{m}.pth",
                "--output",
                f"{DEFAULTS['output']}{m}",
                "--w_bpa",
                "1000",
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
    # for each mesh in outputs
    # collect and group .npz files by experiment prefix

    for m in os.listdir(DEFAULTS["output"]):
        current_mesh_path = f"{DEFAULTS['output']}{m}/"
        files = [f for f in os.listdir(current_mesh_path) if f.endswith(".npz")]
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
            f"All results are for {args.num_iterations} iterations and from a uniform grid init\n"
            "cdp = chamfer distance vertices+bisectors to target point\n"
            "v = voroloss\n"
            "all metrics are multiplied by 1e4 \n"
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
            img_init = plt.imread(os.path.join(current_mesh_path, init_png))
            trimmed_init = crop_transparent(img_init, tol=1e-3)
            ax.imshow(trimmed_init)
            ax.set_title("Init")
            ax.axis("off")

            # --- final image ---
            ax = axs[i, 1]
            final_png = grp.get("final", "").replace(".npz", ".png")
            img_final = plt.imread(os.path.join(current_mesh_path, final_png))
            trimmed_final = crop_transparent(img_final, tol=1e-3)
            ax.imshow(trimmed_final)
            ax.set_title("Final")
            ax.axis("off")

            # --- metrics ---
            ax = axs[i, 2]
            data = np.load(os.path.join(current_mesh_path, grp["final"]))
            text = (
                f"Chamfer: {data['chamfer'] * 1e4:.4f}\n"
                f"Accuracy: {data['accuracy'] * 1e4:.4f}\n"
                f"Completeness: {data['completeness'] * 1e4:.4f}\n"
                f"Precision: {data['precision'] * 1e4:.4f}\n"
                f"Recall: {data['recall'] * 1e4:.4f}\n"
                f"F1: {data['f1'] * 1e4:.4f}\n"
                f"Train time (s):         {data['train_time']:.4f}\n"
                f"Nber of sites: {len(data['sites'])}\n"
            )
            ax.text(0, 0.5, text, va="center", fontsize=18, family="monospace")
            ax.set_title(prefix, fontsize=20)
            ax.axis("off")

        # -------------------------------------------------------------------------
        plt.savefig(
            os.path.join(current_mesh_path, f"results{args.num_iterations}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        print(os.path.join(current_mesh_path, f"results{args.num_iterations}.png"))


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


def generate_latex_table_from_npz(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    methods = {}

    for f in files:
        prefix = f.replace(".npz", "")
        data = np.load(os.path.join(folder, f))
        methods[prefix] = {
            "CD": data["chamfer"] * 1e4,
            "F1": data["f1"],
            # "NC": data.get("normal_consistency", np.nan),
            "Time": data["train_time"],
        }

    # Order by method name
    method_names = sorted(methods.keys())

    # Build LaTeX string
    latex = []
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Method & CD ($\\times10^{-4}$) & F1 & Time (s) \\\\")
    latex.append("\\midrule")

    for name in method_names:
        m = methods[name]
        latex.append(f"{name} & {m['CD']:.3f} & {m['F1']:.3f} & {m['Time']:.1f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    latex_table = "\n".join(latex)

    return latex_table


def BPA_metrics(args):
    bpa_pts, mesh = su.sample_points_on_mesh(
        args.save_path + ".obj", n_points=args.target_size * args.target_size * 150
    )
    gt_pts, _ = su.sample_points_on_mesh(args.mesh + ".obj", n_points=args.target_size * args.target_size * 150)
    accuracy, completeness, chamfer, precision, recall, f1 = su.chamfer_accuracy_completeness_f1(bpa_pts, gt_pts)
    print(f"Chamfer Accuracy (Ours → GT): {accuracy:.6f}")
    print(f"Chamfer Completeness (GT → Ours): {completeness:.6f}")
    print(f"Chamfer Distance (symmetric): {chamfer:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    s = f"{args.save_path}_init.npz"
    np.savez(
        s,
        sites=np.zeros(args.target_size * args.target_size * 150),
        v_vect=mesh.vertices,
        f_vect=mesh.faces,
        train_time=99999,
        accuracy=None,
        completeness=None,
        chamfer=None,
        precision=None,
        recall=None,
        f1=None,
    )
    s = f"{args.save_path}_final.npz"
    print("Saving to: ", s)

    np.savez(
        s,
        sites=np.zeros(args.target_size * args.target_size * 150),
        v_vect=mesh.vertices,
        f_vect=mesh.faces,
        train_time=99999,
        accuracy=accuracy,
        completeness=completeness,
        chamfer=chamfer,
        precision=precision,
        recall=recall,
        f1=f1,
    )


if __name__ == "__main__":
    arg_lists = build_arg_list()
    start_time = time()
    print(start_time)
    for arg_list in arg_lists:
        args = define_options_parser(arg_list)

        if args.w_chamfer > 0 or args.w_voroloss > 0:
            args.save_path = (
                args.output
                + f"/cdp{int(args.w_chamfer)}_v{int(args.w_voroloss)}_cvt{int(args.w_cvt)}_clip{args.clip}_build{args.build_mesh}_upsampling{args.upsampling}_num_centroids{args.num_centroids}_target_size{args.target_size}"
            )
        else:
            args.save_path = args.output + "/BPA"

        if os.path.exists(args.save_path + "_final" + ".npz"):
            print("File already exists, skipping...")
            continue

        print("args: ", args)
        model, mnfld_points = load_model(args.mesh, args.target_size, args.trained_HotSpot)
        sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

        if args.w_chamfer > 0:
            sdf = init_sdf(model, sites)
        else:
            sdf = model

        if args.w_chamfer > 0 or args.w_voroloss > 0:
            output_npz(sites, sdf, mnfld_points, args, "init")
            output_image_polyscope("init")
            t0 = time() - start_time
            sites, sites_sdf = train_DCCVT(sites, sdf, mnfld_points, args)
            ti = time() - t0 - start_time
            output_npz(sites, sites_sdf, mnfld_points, args, "final", None, ti)

        else:
            BPA_metrics(args)
            output_image_polyscope("init")

        output_image_polyscope("final")

        # reset everything for the next iteration
        for var_name in ["sites", "sites_sdf", "model", "mnfld_points"]:
            if var_name in locals() and locals()[var_name] is not None:
                del locals()[var_name]
        torch.cuda.empty_cache()

    output_results_figure()
    # print(generate_latex_table_from_npz(args.output))

    # open_everything_polyscope()
