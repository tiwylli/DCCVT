#this file should be run to generate results comparison between DCCVT, Voromesh and the different methods of optimisation
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

from dataset import shape_3d
import models.Net as Net
sys.path.append("3rdparty/HotSpot")




#cuda devices
device = torch.device("cuda:0")
print("Using device: ", torch.cuda.get_device_name(device))


DEFAULTS = {
    "input_dims" : 3,
    "output" : "/home/wylliam/dev/Kyushu_experiments/outputs/",
    "mesh" : "/home/wylliam/dev/Kyushu_experiments/mesh/",
    "trained_HotSpot" : "/home/wylliam/dev/Kyushu_experiments/hotspots_model/",
    #"trained_HotSpot" : "/home/wylliam/dev/HotSpot/log/3D/pc/HotSpot-all-2025-04-24-18-16-03/gargoyle/gargoyle/trained_models/model.pth",
    "num_iterations" : 100,
    "num_centroids" : 4, # ** input_dims 
    "sample_near" : 32, # ** input_dims
    "clip" : True,
    "triangulate" : True,
    "w_cvt" : 100,
    "w_sdf_pull" : 1,    
    "w_voroloss" : 1000,
    "w_cd_points" : 1000,
    "w_cd_mesh" : 1000,
    "upsampling" : 0,
    
}

def define_options_parser():
    parser = argparse.ArgumentParser(description="DCCVT experiments")
    parser.add_argument('--dataset', type=str, default='dataset_name', help='Dataset name')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points in the dataset')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations for optimization')
    return parser.parse_args()

def load_model(mesh, grid, trained_HotSpot):
    #LOAD MODEL WITH HOTSPOT
    loss_type = "igr_w_heat"
    loss_weights = [350, 0, 0, 1, 0, 0, 20]
    train_set = shape_3d.ReconDataset(
        file_path = mesh+".ply",
        n_points=grid*grid*150,#15000, #args.n_points,
        n_samples=10001, #args.n_iterations,
        grid_res=256, #args.grid_res,
        grid_range=1.1, #args.grid_range,
        sample_type="uniform_central_gaussian", #args.nonmnfld_sample_type,
        sampling_std=0.5, #args.nonmnfld_sample_std,
        n_random_samples=7500, #args.n_random_samples,
        resample=True,
        compute_sal_dist_gt=(
            True if "sal" in loss_type and loss_weights[5] > 0 else False
        ),
        scale_method="mean"#"mean" #args.pcd_scale_method,
    )
    model = Net.Network(
        latent_size=0,#args.latent_size,
        in_dim=3,
        decoder_hidden_dim=128,#args.decoder_hidden_dim,
        nl="sine",#args.nl,
        encoder_type="none",#args.encoder_type,
        decoder_n_hidden_layers=5,#args.decoder_n_hidden_layers,
        neuron_type="quadratic",#args.neuron_type,
        init_type="mfgi",#args.init_type,
        sphere_init_params=[1.6, 0.1],#args.sphere_init_params,
        n_repeat_period=30#args.n_repeat_period,
    )
    model.to(device)
    ######       
    test_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)   
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
        #throw error not yet implemented
        raise NotImplementedError("2D not yet implemented")
    elif input_dims == 3:
        x = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        y = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        z = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        meshgrid = torch.meshgrid(x, y, z)
        meshgrid = torch.stack(meshgrid, dim=3).view(-1, 3)

    sites = meshgrid.to(device, dtype=torch.float32).requires_grad_(True)
    #add mnfld points with random noise to sites 
    N = mnfld_points.squeeze(0).shape[0]
    num_samples = sample_near**input_dims - num_centroids**input_dims
    idx = torch.randint(0, N, (num_samples,))
    sampled = mnfld_points.squeeze(0)[idx]
    perturbed = sampled + (torch.rand_like(sampled)-0.5)*noise_scale
    sites = torch.cat((sites, perturbed), dim=0)
    # make sites a leaf tensor
    sites = sites.detach().requires_grad_()
    return sites
        
        
   
def train_DCCVT(sites, model, target_pc, args):
    optimizer = torch.optim.Adam([
    {'params': [sites], 'lr': args.lr_sites},
    #{'params': model.parameters(), 'lr': lr_model}
], betas=(0.9, 0.999))
    upsampled = 0.0
    epoch = 0
    t0 = time()
    
    while epoch <= args.max_iter:
        optimizer.zero_grad()
        
        if args.w_sdf_pull > 0:
            for param in model.parameters():
                param.requires_grad = False
            #s1 = torch.mean(model(points)**2)
            #s2 = torch.maximum((model(sites).abs() - 0.05), torch.tensor(0.0)).mean()
            #sdf_loss = 0*s1+s2
            sdf_loss = torch.maximum((model(sites).abs() - 0.05), torch.tensor(0.0)).mean()
            sdf_loss.backward(retain_graph=True)
            for param in model.parameters():
                param.requires_grad = True
        
        if args.w_cvt > 0 or args.w_cd_points > 0 or args.w_cd_mesh > 0:
            sites_np = sites.detach().cpu().numpy()
            d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(args.input_dims*sites_np.shape[0]))
            d3dsimplices = np.array(d3dsimplices)
            
            if args.w_cd_points > 0:
                vertices_to_compute, bisectors_to_compute = su.compute_zero_crossing_vertices_3d(sites, None, None, d3dsimplices, model)
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
                triangle_faces = [[f[0], f[i], f[i+1]] for f in f_vect for i in range(1, len(f)-1)]
                triangle_faces = torch.tensor(triangle_faces, device=device)
                hs_p = su.sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=2*args.sample_near*150)
            else:
                hs_p = su.sample_mesh_points_heitz(v_vect, f_vect, num_samples=2*args.sample_near*150)
            
            chamfer_loss_mesh, _ = chamfer_distance(target_pc.detach(), hs_p.unsqueeze(0))

        if args.w_voroloss > 0:
            voroloss = lf.Voroloss_opt().to(device)
            voroloss_loss = voroloss(target_pc.squeeze(0), sites).mean()
        
        sites_loss = (
            args.w_cvt * cvt_loss +
            args.w_cd_mesh * chamfer_loss_mesh +
            args.w_cd_points * chamfer_loss_points +
            args.w_voroloss * voroloss_loss
        )
            
        loss = sites_loss
        print(f"Epoch {epoch}: loss = {loss.item()}")
        loss.backward()
        print("-----------------")
        
        optimizer.step()
        
        # if epoch>100 and (epoch // 100) == upsampled+1 and loss.item() < 0.5 and upsampled < upsampling:
        if epoch/args.max_iter > (upsampled+1)/(args.upsampling+1) and upsampled < args.upsampling:
            print("sites length BEFORE UPSAMPLING: ",len(sites))
            sites = su.upsampling_vectorized(sites, tri=None, vor=None, simplices=d3dsimplices, model=model)
            sites = sites.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([{'params': [sites], 'lr': args.lr_sites}, 
                                          #{'params': model.parameters(), 'lr': lr_model}
                                          ])
            upsampled += 1.0
            print("sites length AFTER: ",len(sites))
            
        epoch += 1           
    
    #Export the sites, their sdf values, the gradients of the sdf values and the hessian
    sdf_values = model(sites)
    sdf_gradients = torch.autograd.grad(outputs=sdf_values, inputs=sites, grad_outputs=torch.ones_like(sdf_values), create_graph=True, retain_graph=True,)[0] # (N, 3)
    N, D = sites.shape
    hess_sdf = torch.zeros(N, D, D, device=sites.device)
    for i in range(D):
        grad2 = torch.autograd.grad(outputs=sdf_gradients[:, i], inputs=sites, grad_outputs=torch.ones_like(sdf_gradients[:, i]), create_graph=False, retain_graph=True,)[0] # (N, 3)
        hess_sdf[:, i, :] = grad2 # fill row i of each 3×3 Hessian
    
    save_path = args.output + os.path.basename(args.mesh) + f"{args.w_cvt}_{args.w_cd_points}_{args.w_cd_mesh}_{args.w_voroloss}_{args.w_sdf_pull}"
    np.savez(f'{save_path}.npz', 
             sites=sites.detach().cpu().numpy(), 
             sdf_values=sdf_values.detach().cpu().numpy(), 
             sdf_gradients=sdf_gradients.detach().cpu().numpy(), 
             sdf_hessians=hess_sdf.detach().cpu().numpy(),
             train_time = time()-t0
             )
    return sites     


if __name__ == "__main__":
    parser = define_options_parser()
    args = parser.parse_args()
    
    model, mnfld_points = load_model(args.mesh, args.grid, args.trained_HotSpot)
    sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)
    sites = train_DCCVT(sites, model, mnfld_points, args)