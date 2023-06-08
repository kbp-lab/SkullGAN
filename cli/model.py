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
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
## DATASET
## -------------------------------- ##

## ----- CUSTOM DATASET ----- ##
class CustomDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        if self.transform:
            image = self.transform(image)
        return image.float()
    

## -------------------------------- ##
## GENERATOR ARCHITECTURE
## -------------------------------- ##

## ----- GENERATOR NOISE ----- ##
class GeneratorNoise(nn.Module):
    def __init__(self, device, upper_bound, lower_bound, std):
        super(GeneratorNoise, self).__init__()
        self.device = device
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.std = std
    
    def forward(self, x):
        clone = torch.clone(x)
        pixel_max = torch.amax(clone, (0, 1), True)
        channel_max = torch.amax(clone, (2, 3), True)
        channel_min = torch.amin(clone, (2, 3), True)        
        channel_std = torch.std(x, dim=(2, 3), keepdims=True)
        pixel_std = torch.std(x, dim=(0, 1), keepdims=True)        
        channel_mean = torch.mean(x, dim=(2, 3), keepdims=True)
        pixel_mean = torch.mean(x, dim=(2, 3), keepdims=True)  
        noise = torch.empty_like(x).normal_(mean=0.0, std=1.0).to(self.device)

        if self.std == 'fixed':
            channel_noise = noise
        elif self.std == 'dynamic':
            channel_noise = noise * channel_std

        clone[clone < - channel_max / self.lower_bound] = 0        
        clone[clone > channel_max / self.upper_bound] = 0        
        return x + clone * channel_noise           
    
## ----- GENERATOR ----- ##
class Generator(nn.Module):
    def __init__(self, device, nc, nz, ngf):
        super(Generator, self).__init__()
        
        self.device = device
        
        # input is Z, going into a convolution
        self.main1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True))
            
        # state size. 4096 x 4 x 4
        self.main2 = nn.Sequential(            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),            
            nn.ReLU(True))

        # state size. 2048 x 8 x 8
        self.main3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
            
        # state size. 1024 x 16 x 16 
        self.main4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),            
            nn.ReLU(True))
            
        # state size. 512 x 32 x 32
        self.main5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True))

        # state size. 256 x 64 x 64        
        self.main6 = nn.Sequential(
            nn.ConvTranspose2d(ngf, round(ngf / 2), 4, 2, 1, bias=False))
            
        # state size. 1 x 128 x 128
        self.conv1 = nn.Conv2d(round(ngf / 2), round(ngf / 4), 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(round(ngf / 4), round(ngf / 8), 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(round(ngf / 8), round(ngf / 16), 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(round(ngf / 16), nc, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()
 
    def forward(self, input):
        channel_noise_1 = GeneratorNoise(self.device, upper_bound=4.0, lower_bound=2.0, std='dynamic')
        x = self.main1(input)
        x = self.main2(x)
        x = self.main3(x)
        x = self.main4(x)
        x = self.main5(x)
        x = self.main6(x)
        x = channel_noise_1(x)
        x = self.conv1(x)                                    
        x = self.conv2(x)  
        x = channel_noise_1(x)             
        x = self.conv3(x)      
        x = self.conv4(x)      
        x = self.tanh(x)
        return x
    
    
## -------------------------------- ##
## DISCRIMINATOR ARCHITECTURE
## -------------------------------- ##

## ----- DISCRIMINATOR ----- ##
class Discriminator(nn.Module):
    def __init__(self, device, nc, ndf):
        super(Discriminator, self).__init__()
        
        self.device = device
        
        self.main = nn.Sequential(
            # input is 1 x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=True), 
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 32 x 64 x 64
            #GaussianNoise(),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 64 x 32 x 32
            #GaussianNoise(),            
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 128 x 16 x 16 
            #GaussianNoise(),            
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. 256 x 8 x 8
            #GaussianNoise(),            
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 512 x 4 x 4
            #GaussianNoise(),            
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)
