import os
import cv2
import torch
import argparse
import numpy as np
import src.Geometry.sampling as sampler
from src.Geometry.CVT2D import CVT_2D
import src.IO.ply as ply
from src.Confs.VisHull import Load_Visual_Hull
from src.IO.dataset import Dataset
from pyhocon import ConfigFactory
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer 
import cv2 as cv
from src.models.fields import ColorGEONetwork
import scipy.spatial


from torch.utils.cpp_extension import load
cvt_march_cuda = load(
    'cvt_march_cuda', ['src/Geometry/CVT_sampling.cpp', 'src/Geometry/CVT_sampling.cu'], verbose=True)

class Runner:
    def __init__(self, conf_path, data_name, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.base_exp_dir  = self.base_exp_dir .replace('DATA_NAME', data_name)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        os.makedirs(self.base_exp_dir + "/validations_fine", exist_ok=True)
       
        self.end_iter = 100

    ###############################################
    ###############################################
    ### Main function to optimize the sites with Lloyd algorithm
    def Lloyd(self, verbose = True):
        print("Start Lloyd")
        ###############################################
        ############ Prepare CVT ######################
        ###############################################
        
        if not hasattr(self, 'cvt'):
            ##### 2. Load initial sites
            visual_hull = [150, 150, 400, 400]
            import src.Geometry.sampling as sampler
            sites = sampler.sample_2D_Bbox(visual_hull[0:2], visual_hull[2:4], 8) #, perturb_f =  10) 

            outside = np.zeros(sites.shape[0], np.int32)
            outside[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/16] = 1
            outside[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/16] = 1
            outside[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/16] = 1
            outside[sites[:,1] > visual_hull[3] - (visual_hull[2]-visual_hull[0])/16] = 1

            from numpy import random
            sites[outside == 0] = sites[outside == 0] + 20*random.rand(sites.shape[0]-outside.sum(), 2)

            self.cvt = CVT_2D(sites, np.array([]))

        for iter_step in tqdm(range(self.end_iter)):         
            ## sample points along the rays
            start = timer()

            ## Compute Voronoi graph from current sites
            voro = scipy.spatial.Voronoi(self.cvt.sites[:self.cvt.nb_sites].cpu().numpy())
            
            #print(voro.points.ptp)
            #print(voro.regions)

            ## Compute center of each cell
            centers = torch.zeros(self.cvt.sites.shape).cuda()
            centers[:] = self.cvt.sites[:]
            for i in range(self.cvt.nb_sites):
                if voro.point_region[i] == -1 or outside[i]: ## no cell for the site
                    continue
                
                #print(voro.point_region[i])
                #print(voro.regions[voro.point_region[i]])
                #print(sum([x == -1 for x in voro.regions[voro.point_region[i]]]))
                    
                infinite_reg = sum([x == -1 for x in voro.regions[voro.point_region[i]]])

                if not infinite_reg and len(voro.regions[voro.point_region[i]]) > 3: ## infinite cell
                    points = voro.vertices[voro.regions[voro.point_region[i]]]
                    hull = scipy.spatial.ConvexHull(points)
                    ## compute center of cell as the barycenter of border
                    sx = sy = sL = 0
                    for k in range(len(hull.vertices)):   # counts from 0 to len(points)-1
                        x0, y0 = points[hull.vertices[k - 1]]     # in Python points[-1] is last element of points
                        x1, y1 = points[hull.vertices[k]]
                        L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
                        sx += (x0 + x1)/2 * L
                        sy += (y0 + y1)/2 * L
                        sL += L
                    xc = sx / sL
                    yc = sy / sL
                    centers[i,0] = xc
                    centers[i,1] = yc
                    #print("centers[i] => ", centers[i])
                    #print("self.cvt.sites[i] => ", self.cvt.sites[i])

            if True:
                self.cvt.draw("Exp/Lloyd/2D_cvt_"+str(iter_step).zfill(5)+".png", centers.cpu().numpy())
                print("Done")

            ## Move sites to center points 
            self.cvt.sites[:self.cvt.nb_sites, :] = centers[:]
            #print("centers[i] => ", centers[16])
            #print("self.cvt.sites[i] => ", self.cvt.sites[16])

            input()

    def ImplicitCVT(self, K_NN = 9, verbose = True):
        print("Start ImplicitCVT")
        ###############################################
        ############ Prepare CVT ######################
        ###############################################
        
        if not hasattr(self, 'cvt'):
            ##### 2. Load initial sites
            visual_hull = [150, 150, 400, 400]
            import src.Geometry.sampling as sampler
            sites = sampler.sample_2D_Bbox(visual_hull[0:2], visual_hull[2:4], 8) #, perturb_f =  10) 

            outside = np.zeros(sites.shape[0], np.int32)
            outside[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/16] = 1
            outside[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/16] = 1
            outside[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/16] = 1
            outside[sites[:,1] > visual_hull[3] - (visual_hull[2]-visual_hull[0])/16] = 1

            from numpy import random
            sites[outside == 0] = sites[outside == 0] + 20*random.rand(sites.shape[0]-outside.sum(), 2)

            self.cvt = CVT_2D(sites, np.array([]))

        for iter_step in tqdm(range(self.end_iter)):         
            ## sample points along the rays
            start = timer()

            ## Compute Voronoi graph from current sites
            voro = scipy.spatial.Voronoi(self.cvt.sites[:self.cvt.nb_sites].cpu().numpy())
            
            ## Compute center of each cell
            centers = torch.zeros(self.cvt.sites.shape).cuda()
            centers[:] = self.cvt.sites[:]
            for i in range(self.cvt.nb_sites):
                if voro.point_region[i] == -1 or outside[i]: ## no cell for the site
                    continue
                    
                infinite_reg = sum([x == -1 for x in voro.regions[voro.point_region[i]]])

                if not infinite_reg and len(voro.regions[voro.point_region[i]]) > 3: ## infinite cell
                    points = voro.vertices[voro.regions[voro.point_region[i]]]
                    hull = scipy.spatial.ConvexHull(points)
                    ## compute center of cell as the barycenter of border
                    sx = sy = sL = 0
                    for k in range(len(hull.vertices)):   # counts from 0 to len(points)-1
                        x0, y0 = points[hull.vertices[k - 1]]     # in Python points[-1] is last element of points
                        x1, y1 = points[hull.vertices[k]]
                        L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
                        sx += (x0 + x1)/2 * L
                        sy += (y0 + y1)/2 * L
                        sL += L
                    xc = sx / sL
                    yc = sy / sL
                    centers[i,0] = xc
                    centers[i,1] = yc

            if True:
                self.cvt.draw("Exp/Lloyd/2D_cvt_"+str(iter_step).zfill(5)+".png", centers.cpu().numpy())
                print("Done")

            ## Move sites
            loss = 0
            grad = torch.zeros(self.cvt.sites.shape).cuda()
            for i in range(self.cvt.nb_sites):
                if outside[i]:
                    continue

                curr_site = self.cvt.sites[i]
                idx = self.cvt.knn_sites[i+1]

                for k in range(K_NN-1):
                    # Compute bisector normal vector
                    nmle = (self.cvt.sites[idx[k]] - curr_site) / (torch.linalg.norm(self.cvt.sites[idx[k]] - curr_site))
                    
                    # Compute middle point
                    #b_point = (self.cvt.sites[idx[i]] + curr_site) / 2.0
                    dist = torch.linalg.norm(self.cvt.sites[idx[k]] - curr_site) / 2.0

                    ### Test  for all other bisectors if it is opposite
                    min_dist = np.inf
                    min_id = k
                    for j in range(K_NN-1):
                        if k == j:
                            continue

                        # Compute bisector normal vector
                        nmle_curr = (self.cvt.sites[idx[j]] - curr_site) / (torch.linalg.norm(self.cvt.sites[idx[j]] - curr_site))
                        # Compute middle point
                        b_point_curr = (self.cvt.sites[idx[j]] + curr_site) / 2.0

                        # Compute ray - plane intersection point
                        denom = torch.sum(nmle_curr * nmle, dim=-1)
                        if denom > 0.0:
                            dist_curr = torch.sum((b_point_curr - curr_site) * nmle_curr, dim=-1) / denom
                            if dist_curr < min_dist:
                                min_dist = dist_curr
                                min_id = j
                    
                    if min_dist >= dist: # no other bisector that hides the current bisector
                        min_id = k
                        min_dist = dist

                    loss = loss + min_dist
                    # compute gradient
                    #grad[i] = grad[i] - (self.cvt.sites[idx[k]] - curr_site)
                    grad[i] = grad[i] - min_dist*nmle
                    

            print(loss)
            self.cvt.sites[outside == 0] = self.cvt.sites[outside == 0] - 0.001*grad[outside == 0]

            #input()

    def ImplicitCVT2(self, K_NN = 9, verbose = True):
        print("Start ImplicitCVT")
        ###############################################
        ############ Prepare CVT ######################
        ###############################################
        
        if not hasattr(self, 'cvt'):
            ##### 2. Load initial sites
            visual_hull = [100, 100, 400, 400]
            from numpy import random
            if True:
                res = 16
                import src.Geometry.sampling as sampler
                sites = sampler.sample_2D_Bbox(visual_hull[0:2], visual_hull[2:4], res) #, perturb_f =  10) 

                outside = np.zeros(sites.shape[0], np.int32)
                outside[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/(2*res)] = 1
                outside[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/(2*res)] = 1
                outside[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/(2*res)] = 1
                outside[sites[:,1] > visual_hull[3] - (visual_hull[3]-visual_hull[1])/(2*res)] = 1

                """sites[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/16, 0] = sites[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/16, 0] - 50
                sites[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/16, 1] = sites[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/16, 1] - 50
                sites[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/16, 0] = sites[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/16, 0] + 50
                sites[sites[:,1] > visual_hull[3] - (visual_hull[3]-visual_hull[1])/16, 1] = sites[sites[:,1] > visual_hull[3] - (visual_hull[3]-visual_hull[1])/16, 1] + 50"""

                sites[outside == 0] = sites[outside == 0] + 15*random.rand(sites.shape[0]-outside.sum(), 2)

                np.save("Exp/sites_init_16.npy", sites)

            else:
                res = 16
                sites = np.load("Exp/sites_init_16.npy")
                outside = np.zeros(sites.shape[0], np.int32)
                outside[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/(2*res)] = 1
                outside[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/(2*res)] = 1
                outside[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/(2*res)] = 1
                outside[sites[:,1] > visual_hull[3] - (visual_hull[3]-visual_hull[1])/(2*res)] = 1

            self.cvt = CVT_2D(sites, np.array([]))

            
        ##### 2. Initialize SDF field    
        if not hasattr(self, 'sdf'):
            self.sdf = torch.linalg.norm(self.cvt.sites - torch.from_numpy(np.array([256,256])).cuda(), ord=2, axis=-1, keepdims=True)[:,0] - 50.0       
            self.sdf = self.sdf.contiguous()
            self.grad_sdf = torch.div(2.0*(self.cvt.sites - torch.from_numpy(np.array([256,256])).cuda()), self.sdf.reshape(-1,1).expand(self.cvt.sites.shape[0], 2))
            """self.grad_sdf = torch.zeros((self.cvt.sites.shape[0], 2))
            self.grad_sdf[:,0] = (1.0/self.sdf[:])*2.0*(self.cvt.sites[:,0] - 256)
            self.grad_sdf[:,1] = (1.0/self.sdf[:])*2.0*(self.cvt.sites[:,1] - 256)"""

        self.cvt.draw("Exp/Implicit2/2D_cvt.png", np.array([]), self.sdf.cpu().numpy())

        sites = torch.from_numpy(sites).cuda()
        sites.requires_grad_(True)
        self.optimizer = torch.optim.Adam([sites], lr=1.0)

        self.lr = 1.0e-5

        for iter_step in tqdm(range(self.end_iter)):         
            norm_sites = torch.linalg.norm(self.cvt.sites - torch.from_numpy(np.array([256,256])).cuda(), ord=2, axis=-1, keepdims=True)
            #print(self.cvt.sites[18])
            #print(norm_sites[18])
            self.sdf = norm_sites[:,0]**2 - 75.0**2  
            #print(self.sdf[18])
            self.grad_sdf = 2.0 * (self.cvt.sites - torch.from_numpy(np.array([256,256])).cuda())
            #self.grad_sdf = torch.div((self.cvt.sites - torch.from_numpy(np.array([256,256])).cuda()), norm_sites.reshape(-1,1).expand(self.cvt.sites.shape[0], 2))

            self.grad_grad = torch.zeros([self.cvt.sites.shape[0], 4]).cuda()
            self.grad_grad[:,0] = 2.0
            self.grad_grad[:,3] = 2.0
            ## sample points along the rays
            start = timer()

            ## Compute Voronoi graph from current sites
            voro = scipy.spatial.Voronoi(self.cvt.sites[:self.cvt.nb_sites].cpu().numpy())
            
            ## Compute center of each cell
            centers = torch.zeros(self.cvt.sites.shape).cuda()
            centers[:] = self.cvt.sites[:]
            for i in range(self.cvt.nb_sites):
                if voro.point_region[i] == -1 or outside[i]: ## no cell for the site
                    continue
                    
                infinite_reg = sum([x == -1 for x in voro.regions[voro.point_region[i]]])

                if not infinite_reg and len(voro.regions[voro.point_region[i]]) > 3: ## infinite cell
                    points = voro.vertices[voro.regions[voro.point_region[i]]]
                    hull = scipy.spatial.ConvexHull(points)
                    ## compute center of cell as the barycenter of border
                    sx = sy = sL = 0
                    for k in range(len(hull.vertices)):   # counts from 0 to len(points)-1
                        x0, y0 = points[hull.vertices[k - 1]]     # in Python points[-1] is last element of points
                        x1, y1 = points[hull.vertices[k]]
                        L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
                        sx += (x0 + x1)/2 * L
                        sy += (y0 + y1)/2 * L
                        sL += L
                    xc = sx / sL
                    yc = sy / sL
                    centers[i,0] = xc
                    centers[i,1] = yc


            ## Move sites
            loss = 0
            grad = torch.zeros(self.cvt.sites.shape).cuda()
            ## Shoot 4 rays with opposite directions ( = 8  rays in total)
            ray = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [-1.0, 1.0]]
            samples = torch.zeros((9,2)).cuda()
            for i in range(self.cvt.nb_sites):
                if outside[i]:
                    continue

                curr_site = self.cvt.sites[i]
                idx = self.cvt.knn_sites[i+1]

                theta = 2.0 * np.pi * random.rand(1) # random rotation angle
                rot = torch.from_numpy(np.array([[np.cos(theta[0]), -np.sin(theta[0])], [np.sin(theta[0]), np.cos(theta[0])]])).cuda()

                cross_site = False
                tot_dist = 0.0
                for r_id in range(4):
                    curr_ray = torch.matmul(rot, torch.from_numpy(np.array(ray[r_id])).cuda())
                    min_dist = np.inf
                    min_id = -1
                    max_dist = -np.inf
                    max_id = -1
                    for k in range(K_NN-1):
                        if idx[k] == -1:
                            break
                        # Compute bisector normal vector
                        nmle = (self.cvt.sites[idx[k]] - curr_site) / (torch.linalg.norm(self.cvt.sites[idx[k]] - curr_site))
                        
                        # Compute middle point
                        b_point = (self.cvt.sites[idx[k]] + curr_site) / 2.0

                        # Compute ray - plane intersection point
                        denom = torch.sum(curr_ray * nmle, dim=-1)
                        if abs(denom) > 1.0e-6:
                            dist = torch.sum((b_point - curr_site) * nmle, dim=-1) / denom
                            if dist >= 0.0 and dist < min_dist:
                                min_dist = dist
                                min_id = idx[k]
                            if dist <= 0.0 and dist > max_dist:
                                max_dist = dist
                                max_id = idx[k]
                    
                    if abs(min_dist) == np.inf or abs(max_dist) == np.inf:
                        print("NO INTERSECTION??")
                        continue

                    denom1 = torch.sum(curr_ray * (self.cvt.sites[min_id] - curr_site), dim=-1) 
                    num1 = torch.linalg.norm(self.cvt.sites[min_id] - curr_site)**2
                    denom2 = torch.sum(curr_ray * (self.cvt.sites[max_id] - curr_site), dim=-1) 
                    num2 = torch.linalg.norm(self.cvt.sites[max_id] - curr_site)**2

                    d_min_dist = 0.5 * (-(self.cvt.sites[min_id] - curr_site)/denom1 + num1 * curr_ray/(denom1**2))
                    d_max_dist = 0.5 * (-(self.cvt.sites[max_id] - curr_site)/denom2 + num2 * curr_ray/(denom2**2))

                    grad[i] = grad[i] - 1.0e4*(min_dist * d_min_dist + max_dist * d_max_dist)                                        
                    #grad[min_id] = grad[min_id] + min_dist * d_min_dist 
                    #grad[max_id] = grad[max_id] + max_dist * d_max_dist
                    
                    #grad[i] = grad[i] - (min_dist + max_dist)*curr_ray  ## <====
                    #grad[min_id] = grad[min_id] + min_dist*curr_ray  
                    #grad[max_id] = grad[max_id] + max_dist*curr_ray  

                    if not outside[i]:
                        tot_dist = tot_dist + min_dist
                        #loss = loss + 0.5*(min_dist**2 + max_dist**2)

                    #if i == 12:
                    #    samples[r_id] = curr_site + min_dist*curr_ray

                    if self.sdf[i]*self.sdf[min_id] < 0.0:
                        #sdf_mid = 0.5*((self.sdf[i] + 0.5*torch.sum(self.grad_sdf[i] * (self.cvt.sites[min_id] - curr_site), dim=-1)) +\
                        #               (self.sdf[min_id] - 0.5*torch.sum(self.grad_sdf[min_id] * (self.cvt.sites[min_id] - curr_site), dim=-1)))

                        sdf_mid = self.sdf[i] + min_dist * torch.sum(self.grad_sdf[i] * curr_ray, dim=-1)

                        ##### TRUE GRADIENT #####
                        grad[i] = grad[i] + sdf_mid*(self.grad_sdf[i] + d_min_dist*torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) +\
                                                     (min_dist * torch.tensor([curr_ray[0] * self.grad_grad[i,0] + curr_ray[1] * self.grad_grad[i,1], curr_ray[0] * self.grad_grad[i,2] + curr_ray[1] * self.grad_grad[i,3]]).cuda()))
                        
                        #grad[min_id] = grad[min_id] + sdf_mid*(d_min_dist*torch.sum(self.grad_sdf[i] * curr_ray, dim=-1))
                        
                        """if self.sdf[i] < 0.0:
                            grad[i] = grad[i] + sdf_mid*curr_ray #torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) * d_min_dist
                        else:
                            grad[i] = grad[i] - sdf_mid*curr_ray #torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) * d_min_dist
                        #grad[min_id] = grad[min_id] + 0.5*sdf_mid*(self.grad_sdf[i] - self.grad_sdf[min_id])"""
                        loss = loss + 0.5*(sdf_mid**2)
                        #print(grad[i])
                        
                    if self.sdf[i]*self.sdf[max_id] < 0.0:
                        sdf_mid = self.sdf[i] + max_dist * torch.sum(self.grad_sdf[i] * curr_ray, dim=-1)
                        ##### TRUE GRADIENT #####
                        grad[i] = grad[i] + sdf_mid*(self.grad_sdf[i] + d_max_dist*torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) +\
                                                     (max_dist * torch.tensor([curr_ray[0] * self.grad_grad[i,0] + curr_ray[1] * self.grad_grad[i,1], curr_ray[0] * self.grad_grad[i,2] + curr_ray[1] * self.grad_grad[i,3]]).cuda()))
                        
                        #grad[max_id] = grad[max_id] + sdf_mid*(d_max_dist*torch.sum(self.grad_sdf[i] * curr_ray, dim=-1))

                        """if self.sdf[i] < 0.0:
                            grad[i] = grad[i] - sdf_mid*curr_ray #torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) * d_max_dist
                        else:
                            grad[i] = grad[i] + sdf_mid*curr_ray #torch.sum(self.grad_sdf[i] * curr_ray, dim=-1) * d_max_dist
                        """
                        #sdf_mid = 0.5*((self.sdf[i] + 0.5*torch.sum(self.grad_sdf[i] * (self.cvt.sites[max_id] - curr_site), dim=-1)) +\
                        #               (self.sdf[max_id] - 0.5*torch.sum(self.grad_sdf[max_id] * (self.cvt.sites[max_id] - curr_site), dim=-1)))
                        #grad[i] = grad[i] + 0.5*sdf_mid*(self.grad_sdf[max_id] - self.grad_sdf[i])
                        #grad[max_id] = grad[max_id] + 0.5*sdf_mid*(self.grad_sdf[i] - self.grad_sdf[max_id])
                        loss = loss + 0.5*(sdf_mid**2)

                        #grad[i] = grad[i] + 0.5*(self.grad_sdf[max_id] - self.grad_sdf[i])
                        #loss = loss + 0.5*(self.sdf[i] + 1.0*torch.sum(self.grad_sdf[i] * (self.cvt.sites[max_id] - curr_site), dim=-1))**2
                        #print(grad[i])"""
                        
                    if self.sdf[i]*self.sdf[min_id] < 0.0 or self.sdf[i]*self.sdf[max_id] < 0.0:
                        cross_site = True
                
                """if abs(self.sdf[i]) < 10.0: #cross_site:
                    grad[i] = self.sdf[i]*self.grad_sdf[i]
                    loss = loss + 0.5*(self.sdf[i]**2)"""
                
                #grad[i] = grad[i] / 4#tot_dist
                #if i == 12:
                #    print(grad[i])
                #    samples[8] = curr_site + grad[i]
                    
            if iter_step % 1 == 0:
                #self.cvt.draw("Exp/Implicit2/2D_cvt_"+str(iter_step).zfill(5)+".png", np.array([]), self.sdf.cpu().numpy())
                #self.cvt.draw("Exp/Implicit2/2D_cvt_"+str(iter_step).zfill(5)+".png", centers.cpu().numpy(), self.sdf.cpu().numpy())
                #self.cvt.draw("Exp/Implicit2/2D_cvt_"+str(iter_step).zfill(5)+".png", samples.cpu().numpy())
                self.cvt.draw("Exp/Implicit2/2D_cvt.png", centers.cpu().numpy(), self.sdf.cpu().numpy())
                print("Done")
                    

            print(loss)

            with torch.no_grad():
                sites[outside == 0] = sites[outside == 0] - self.lr*grad[outside == 0]

            if iter_step == 10:                
                self.lr = 1.0e-5

                
            """self.optimizer.zero_grad()
            sites.grad = grad
            sites.grad[outside == 1,:] = 0.0
        
            self.optimizer.step()

            with torch.no_grad():
                self.cvt.sites[:] = sites[:]

            if iter_step == 10:                
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.1"""

            #self.cvt.sites[:self.cvt.nb_sites, :] = centers[:]

            self.cvt = CVT_2D(sites.detach().cpu().numpy(), np.array([]))

            #input()

if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./src/Confs/test_bmvs.conf')
    parser.add_argument('--data_name', type=str, default='bmvs_man')
    parser.add_argument('--mode', type=str, default='Lloyd')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    ## Initialise CUDA device for torch computations
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.data_name, args.mode)

    if args.mode == 'Lloyd':
        runner.Lloyd()
    if args.mode == 'Implicit':
        runner.ImplicitCVT2()
        