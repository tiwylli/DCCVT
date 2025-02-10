import polyscope as ps
import torch
import sdfpred_utils.sdfpred_utils as su


sites = su.createCVTgrid(dimensionality=3)
print(sites)

# Initialize polyscope
ps.init()
# Convert tensor to NumPy
points_numpy = sites.numpy()

# Register and show in Polyscope
ps.register_point_cloud("my_tensor_cloud", points_numpy)
ps.show()
