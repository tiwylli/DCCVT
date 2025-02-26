#DMTET network with small adjustments

import torch
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
from tqdm import tqdm

device = torch.device("cuda:0")

# MLP + Positional Encoding
class Decoder(torch.nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 1, hidden = 5, multires = 2):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=input_dims)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter, radius = 2.0):
        print ("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            p = torch.rand((128*128,3), device='cuda') - 0.5
            p = p*20
            ref_value  = torch.sqrt((p**2).sum(-1)) - radius
            output = self(p) # sdf 0 , deform 1-3
            loss = loss_fn(output[...,0], ref_value)
                                                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())
        
    # def pre_train_target_pc(self, iter, target_points):
    #     print ("Initialize SDF to target point cloud")
    #     loss_fn = torch.nn.MSELoss()
    #     optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

    #     for i in tqdm(range(iter)):
    #         p = torch.rand((100000, 3), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
    #         p = p*20
    #         ref_value  = ((abs(target_points) - abs(p))**2).sum(-1)
    #         output = self(p) # sdf 0 , deform 1-3
    #         loss = loss_fn(output[...,0], ref_value)
      
    #         # p2 = target_points
    #         # ref_value2  = torch.zeros((len(target_points)), device=device)
    #         # output2 = self(p2) # sdf 0 , deform 1-3
    #         # loss += loss_fn(output2[...,0], ref_value2)
  

    #         # Compute gradients for Eikonal loss
    #         output3 = self(p)
    #         grads = torch.autograd.grad(
    #             outputs = output3[..., 0],  # Network output
    #             inputs=p,                # Input coordinates
    #             grad_outputs=torch.ones_like(output3[..., 0]),  # Gradient w.r.t. output
    #             create_graph=True,
    #             retain_graph=True
    #         )[0]

    #         # Eikonal loss: Enforce gradient norm to be 1
    #         loss += ((grads.norm(2, dim=1) - 1) ** 2).mean()
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print("Pre-trained MLP", loss.item())
    
    def cleanup(self, iter):
        print ("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)
        scale = 5.0
        for i in tqdm(range(iter)):
            #domain loss: everything outside of -4 and 4 mesh should be positive                
            # p2 = torch.rand((300000, 3), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
            # p2 = p2*scale*2
            # #filter all the points outside -4 and 4 in all dimensions
            # p2 = p2[(p2 < -scale).any(dim=1) | (p2 > scale).any(dim=1)]
            # output2 = self(p2)
            # loss += torch.relu(-output2[:, 0]).sum()


            domain_min, domain_max = -5, 5  # Domain limits in all dimensions
            buffer_scale = 0.2  # Scale for buffer zones
            num_points = 300000  # Number of points to sample
            # Compute buffer zones
            buffer_x = buffer_scale * (domain_max - domain_min)
            buffer_y = buffer_scale * (domain_max - domain_min)
            buffer_z = buffer_scale * (domain_max - domain_min)

            num_per_region = num_points // 6  # Distribute points across six outer regions

            # Sample outside the domain in each dimension
            left_x = torch.empty(num_per_region, device=device).uniform_(domain_min - 2 * buffer_x, domain_min)
            left_y = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_y, domain_max + buffer_y)
            left_z = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_z, domain_max + buffer_z)

            right_x = torch.empty(num_per_region, device=device).uniform_(domain_max, domain_max + 2 * buffer_x)
            right_y = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_y, domain_max + buffer_y)
            right_z = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_z, domain_max + buffer_z)

            top_x = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_x, domain_max + buffer_x)
            top_y = torch.empty(num_per_region, device=device).uniform_(domain_max, domain_max + 2 * buffer_y)
            top_z = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_z, domain_max + buffer_z)

            bottom_x = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_x, domain_max + buffer_x)
            bottom_y = torch.empty(num_per_region, device=device).uniform_(domain_min - 2 * buffer_y, domain_min)
            bottom_z = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_z, domain_max + buffer_z)

            front_x = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_x, domain_max + buffer_x)
            front_y = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_y, domain_max + buffer_y)
            front_z = torch.empty(num_per_region, device=device).uniform_(domain_max, domain_max + 2 * buffer_z)

            back_x = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_x, domain_max + buffer_x)
            back_y = torch.empty(num_per_region, device=device).uniform_(domain_min - buffer_y, domain_max + buffer_y)
            back_z = torch.empty(num_per_region, device=device).uniform_(domain_min - 2 * buffer_z, domain_min)

            # Combine all points
            points_x = torch.cat([left_x, right_x, top_x, bottom_x, front_x, back_x])
            points_y = torch.cat([left_y, right_y, top_y, bottom_y, front_y, back_y])
            points_z = torch.cat([left_z, right_z, top_z, bottom_z, front_z, back_z])

            # Stack into (num_points, 3)
            points = torch.stack([points_x, points_y, points_z], dim=1)

            # Compute the SDF values and loss
            sdf_values = self(points)[:, 0]
            
            loss = torch.relu(-sdf_values).mean()  # Penalize negative SDF (inside the shape)
            
                                                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("cleanup MLP", loss.item())
        
    def train_GT_mesh(self, iter, mesh, target_points):
        #TODO: pretrain sphere first ?
        #self.pre_train_sphere(iter*2)
        print("Train MLP with GT mesh")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)
        
        bunny = trimesh.load(mesh[1])
        
        
        min_target = target_points.min(0)[0]
        max_target = target_points.max(0)[0]
        grid_points, grid_sdf = sample_sdf_near_surface(bunny, number_of_points=10000)
        grid_points = torch.tensor(grid_points, device=device, dtype=torch.float64)
        grid_sdf = torch.tensor(grid_sdf, device=device, dtype=torch.float64)
        min_grid = grid_points.min(0)[0]
        max_grid = grid_points.max(0)[0]

        scale = torch.max((max_target - min_target) / (max_grid - min_grid))
        print("scale", scale)
        grid_points = grid_points * scale
        grid_sdf = grid_sdf * scale
        
        #minmax grids
        print("minmax grid", grid_points.min(0)[0], grid_points.max(0)[0])
        print("minmax sdf", grid_sdf.min(), grid_sdf.max())

                    
        for i in tqdm(range(iter)):
            loss = 0
            p1 = grid_points
            ref_value1 = grid_sdf
            output1 = self(p1) # sdf 0 , deform 1-3
            loss = loss_fn(output1[...,0], ref_value1)
                    
                    
            # # domain loss: everything outside of -4 and 4 mesh should be positive                
            # p2 = torch.rand((10000, 3), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
            # p2 = p2*scale*2
            # #filter all the points outside -4 and 4 in all dimensions
            # p2 = p2[(p2 < -scale).any(dim=1) | (p2 > scale).any(dim=1)]
            # output2 = self(p2)
            # loss += torch.relu(-output2[:, 0]).sum()
            
            
            
            # p3 = target_points
            # ref_value3  = torch.zeros((len(target_points)), device=device)
            # output3 = self(p3) # sdf 0 , deform 1-3
            # loss += loss_fn(output3[...,0], ref_value3)
  
            # # # Compute gradients for Eikonal loss
            # #TODO: sample uniformly 
            # p3 = torch.rand((10000, 3), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
            # p3 = p3*scale*10
            # output3 = self(p3)
            # grads = torch.autograd.grad(
            #     outputs = output3[..., 0],  # Network output
            #     inputs=p3,                # Input coordinates
            #     grad_outputs=torch.ones_like(output3[..., 0]),  # Gradient w.r.t. output
            #     create_graph=True,
            #     retain_graph=True
            # )[0]

            # # Eikonal loss: Enforce gradient norm to be 1
            # loss += ((grads.norm(2, dim=1) - 1) ** 2).mean()
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("GT Trained MLP", loss.item())
        
    # def train_GT_grid(self, iter):
    #     print("Train MLP with GT grid")
    #     loss_fn = torch.nn.MSELoss()
    #     optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)
    #     import numpy as np
    #     filename = "/home/wylliam/dev/Kyushu_experiments/Resources/stanford-bunny.obj"
    #     sdf = np.load(filename[:-4] + '.npy')
    #     #sdf is 128x128x128
    #     #sdf is linked to stanford bunny fixed mesh with min max vertices
    #     grid_sdf = torch.tensor(sdf, device=device, dtype=torch.float64)
    #     grid_sdf = grid_sdf.reshape(-1)
        
    #     for i in tqdm(range(iter)):
    #         #create a unifrorm grid of res
    #         res = 32
    #         x = torch.linspace(-1, 1, res, device=device, dtype=torch.float64)
    #         y = torch.linspace(-1, 1, res, device=device, dtype=torch.float64)
    #         z = torch.linspace(-1, 1, res, device=device, dtype=torch.float64)
    #         xv, yv, zv = torch.meshgrid(x,y,z, indexing='ij')
    #         grid_points = torch.stack((xv, yv, zv), dim=-1).reshape(-1, 3)  # Shape: [res³, 3]

    #         p1 = grid_points
    #         ref_value1 = grid_sdf
    #         output1 = self(p1) # sdf 0 , deform 1-3
    #         loss = loss_fn(output1[...,0], ref_value1)
            
    #         # # Compute gradients for Eikonal loss
    #         p3 = torch.rand((1000, 3), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
    #         p3 = p3*2
    #         output3 = self(p3)
    #         grads = torch.autograd.grad(
    #             outputs = output3[..., 0],  # Network output
    #             inputs=p3,                # Input coordinates
    #             grad_outputs=torch.ones_like(output3[..., 0]),  # Gradient w.r.t. output
    #             create_graph=True,
    #             retain_graph=True
    #         )[0]
            
    #         # Eikonal loss: Enforce gradient norm to be 1
    #         loss += ((grads.norm(2, dim=1) - 1) ** 2).mean()
            
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print("GT Trained MLP", loss.item())

    def pre_train_circle(self, iter, radius=2.0):
        print("Initialize SDF to circle")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            # Generate random points in the 2D plane (x, y)
            p = torch.rand((128*128, 2), device='cuda', requires_grad=True) - 0.5  # x and y values in the range [-0.5, 0.5]
            p = p*20
            # Calculate the reference value (SDF for circle)
            ref_value = torch.sqrt((p**2).sum(-1)) - radius  # Distance from origin (0, 0) minus the circle radius (2)

            # Get the network output
            output = self(p)
            
            # Compute the loss
            sdf_loss = loss_fn(output[..., 0], ref_value)
            
            
            
            # Compute gradients for Eikonal loss
            grads = torch.autograd.grad(
                outputs=output[..., 0],  # Network output
                inputs=p,                # Input coordinates
                grad_outputs=torch.ones_like(output[..., 0]),  # Gradient w.r.t. output
                create_graph=True,
                retain_graph=True
            )[0]

            # Eikonal loss: Enforce gradient norm to be 1
            eikonal_loss = ((grads.norm(2, dim=1) - 1) ** 2).mean()
            

            # Total loss: Combine SDF and Eikonal losses
            total_loss = sdf_loss #+ 1.0 * eikonal_loss  # Adjust the weight (0.1) as necessary

            # Perform backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print("Pre-trained MLP", total_loss.item())
            
        


# Positional Encoding from https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims):
    embed_kwargs = {
                'include_input' : True,
                #'input_dims' : 3,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim






