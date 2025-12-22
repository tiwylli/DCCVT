"""Runtime initialization: device selection and deterministic seeds."""

import numpy as np
import torch

# cuda devices
device = torch.device("cuda:0")
print("Using device: ", torch.cuda.get_device_name(device))
# Improve reproducibility
torch.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)
