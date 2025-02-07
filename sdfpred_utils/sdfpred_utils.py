# Python code for creating a CVT
# Vassilis Vassiliades - Inria, Nancy - April 2018

import numpy as np
from sklearn.cluster import KMeans
import torch

def createCVTgrid():
    # Default values
    num_centroids = 128
    dimensionality = 2
    num_samples = 100000
    num_replicates = 1
    max_iterations = 100000
    tolerance = 0.00001
    verbose = True

    X = np.random.rand(num_samples,dimensionality)

    kmeans = KMeans(
        init='k-means++', 
        n_clusters=num_centroids, 
        n_init=num_replicates, 
        #n_jobs=-1, 
        max_iter=max_iterations, 
        tol=tolerance,
        verbose=0)

    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    centroids = (np.array(centroids- 0.5))* 10.0
    #make centroids double
    centroids = centroids.astype(np.double)

    #sites = torch.from_numpy(centroids).to(device).requires_grad_(True)

    sites = torch.from_numpy(centroids).to(device, dtype=torch.double).requires_grad_(True)
    print(sites.shape, sites.dtype)
    return sites