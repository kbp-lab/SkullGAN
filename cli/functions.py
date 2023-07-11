## -------------------------------- ##
## IMPORT NECESSARY PACKAGES
## -------------------------------- ##

## ----- SYSTEM ----- ##
import os
import random
import numpy as np

## ----- TORCH ----- ##
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

## ----- DISPLAY ----- ##
from IPython import display
from tqdm.notebook import tqdm 

## ----- MISC ----- ##
from datetime import datetime

## ----- REPRODUCIBILITY ----- ##
seed = 111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True


## -------------------------------- ##
## FUNCTIONS
## -------------------------------- ##

## ----- NUMPY -> TENSOR HELPERS ----- ## 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
to_t = lambda array: torch.tensor(array.astype('float32'), device=device)
from_t = lambda tensor: tensor.to('cpu').detach().numpy()

## ----- WEIGHT INITIALIZATION ----- ##
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.constant_(m.bias.data, 0)
        
## ----- LATENT VECTOR GENERATOR ----- ##
latent_vec_gen = 'uniform'
def latent_vec(device, size, nz = 200, gen = latent_vec_gen):
    if gen == "normal":
        return torch.randn(size, nz, 1, 1, device=0)
    elif gen == "uniform":
        return torch.distributions.Uniform(-1.0, 1.0).\
               sample(sample_shape=torch.Size([size, nz, 1, 1])).to(device)

## ----- COUNT MODEL PARAMETERS ----- ##        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## ----- STANDARDIZE SKULLS ----- ##
def standardize(slices, scalings, direction = 1):
    if direction == 0: # FORWARDS
        scalings.append(np.min(slices))
        slices = slices - np.min(slices)     # 0 - 2500

        scalings.append(np.max(slices))
        slices = slices / np.max(slices)             # 0 - 1

        scalings.extend([0.5, 0.5])
        slices = (slices - 0.5) / 0.5                # µ = 0.5, σ = 0.5
        return slices, scalings

    elif direction == 1: # BACKWARDS
        slices = ((slices.squeeze() * 0.5) + 0.5) * scalings[1] + scalings[0]
        return slices
