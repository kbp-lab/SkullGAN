## -------------------------------- ##
## IMPORT NECESSARY PACKAGES
## -------------------------------- ##

## ----- SYSTEM ----- ##
import os
import glob
import random
from datetime import datetime

## ----- FUNCTIONS ----- ##
from functions import *
from model import *

## ----- STATS ----- ##
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

## ----- TORCH ----- ##
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

## ----- DISPLAY ----- ##
from tqdm import tqdm 
import matplotlib.pyplot as plt

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

def main(data, fig_path, params, pretrain = False, pretrained_path = ""):

    ## -------------------------------- ##
    ## UNPACK DATA AND HYPERPARAMETERS
    ## -------------------------------- ##
    
    slices = data[0]
    scalings = data[1]
    
    batch_size = params[0]
    image_size = params[1]
    nc = params[2]
    nz = params[3]
    ngf = params[4]
    ndf = params[5]
    num_epochs = params[6]
    lrD = params[7]
    lrG = params[8]
    beta = params[9]
    
    if device == "cuda":
        ngpu = torch.cuda.device_count()
        
    
    ## -------------------------------- ##
    ## INITIALIZE DATALOADER AND MODELS
    ## -------------------------------- ##
    
    # Dataloader
    dataset = CustomDataset(to_t(slices))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0, 
                            worker_init_fn=np.random.seed(seed))


    # Define generator
    netG = Generator(device, nc, nz, ngf).to(device)
    if torch.cuda.device_count() > 1: netG = nn.DataParallel(netG, list(range(ngpu)))

    # Initialize generator weights
    netG.apply(weights_init)
    netG = netG.float()

    # Define discriminator
    netD = Discriminator(device, nc, ndf).to(device)
    if torch.cuda.device_count() > 1: netG = nn.DataParallel(netG, list(range(ngpu)))

    # Initialize discriminator weights
    netD.apply(weights_init)
    netD = netD.float()

    # Print model parameter counts
    print("")
    print("Discriminator Parameter Count:", f"{count_parameters(netD):,}")
    print("Generator Parameter Count:", f"{count_parameters(netG):,}")
    print("")
    
    
    ## -------------------------------- ##
    ## OPTIMIZATION
    ## -------------------------------- ##
 
    # Define loss as binary cross-entropy
    criterion = nn.BCELoss()

    # Create fixed batch of latent vectors
    fixed_noise = latent_vec(device, batch_size, nz, gen = latent_vec_gen)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta, 0.999)) #, weight_decay=1e-5)
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta, 0.999)) #, weight_decay=1e-5)

    # Establish loss threshold for training
    threshold = 0.9

    # Add dynamic learning rate scheduler
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=1000)
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.8, patience=1000)

    now = str(datetime.now())
    os.makedirs(fig_path + "SkullGAN_" + now + "/", exist_ok=True)  
    os.makedirs(fig_path + "SkullGAN_" + now + "/models/", exist_ok=True) 
    os.makedirs(fig_path + "SkullGAN_" + now + "/images/", exist_ok=True)      
    

    ## -------------------------------- ##
    ## OPTIONAL PRETRAINING
    ## -------------------------------- ##
                
    # Load pretrained model weights
    if pretrain == True:

        # Get list of model filenames
        netG_pretrained = pretrained_path + "netG.pth"
        netD_pretrained = pretrained_path + "netD.pth"

        # Load state dictionaries
        netD.load_state_dict(torch.load(netD_pretrained))
        netG.load_state_dict(torch.load(netG_pretrained))

    ## ----- PROGRESS LISTS ----- ##
    img_list = []
    corr_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
                
    ## -------------------------------- ##
    ## TRAIN MODEL
    ## -------------------------------- ##

    ## ----- DISPLAY ----- ##
    epoch_fig = plt.figure(figsize = (12, 4))

    pbar = tqdm(range(num_epochs))
    pbar.set_description("Epoch")
    inner_pbar = tqdm(range(len(slices)))
    inner_pbar.set_description("Batch")

    normalize = True

    ## ----- FOR EACH EPOCH ----- ##
    for epoch in pbar:

        inner_pbar.reset()

        ## ----- FOR EACH BATCH ----- ##
        for i, data in enumerate(dataloader, 0):

            ## -------------------------------------- ##
            ## UPDATE DISCRIMINATOR:
            ## maximize log(D(x)) + log(1 - D(G(z)))
            ## -------------------------------------- ##

            ## ---- TRAIN WITH ALL REALS ----- ##

            # Zero gradients
            netD.zero_grad()

            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # Labels
            label_smoothing = 0.90
            label = label_smoothing * torch.full((b_size,), real_label, dtype=torch.float32, device=device)

            # Inference discriminator
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label.float())

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## ----- TRAIN WITH FAKES & SMOOTHED REALS ----- ##

            # Generate batch of latent vectors
            noise = latent_vec(device, b_size, nz, gen=latent_vec_gen)

            # Generate fake image batch with G
            brt = 15  # blurred reals threshold
            if epoch <= brt:
                fake = netG(noise)
            elif epoch > brt and epoch <= b_size:
                # Generate fake image batch with G
                noise_std = max(0.5, 1.0 * brt / epoch) 
                Gauss_blur = transforms.GaussianBlur(kernel_size=3, sigma=noise_std)
                real_blurred = Gauss_blur(next(iter(dataloader)))            

                fake = torch.cat((netG(noise)[:int(b_size-(epoch - brt)), :, :], 
                               real_blurred[:(epoch - brt), :, :]), dim=0) 
            elif epoch >= b_size:
                fake = torch.cat((netG(noise)[:int(b_size / 2), :, :], 
                               real_blurred[:int(b_size / 2), :, :]), dim=0) 

            # Labels
            label = torch.full((fake.size()[0],), fake_label, dtype=torch.float32, device=device)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label.float())

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            ## ----- UPDATE D ----- ##

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Step optimizer
            if epoch < 10 or errD.item() > threshold:
                optimizerD.step()

            # Update learning rate
            schedulerD.step(errD)

            ## -------------------------------------- ##
            ## UPDATE GENERATOR: 
            ## maximize log(D(G(z)))
            ## -------------------------------------- ##

            # Zero gradients
            netG.zero_grad()

            # fake labels are real for generator cost
            label = torch.full((fake.size()[0],), real_label, dtype=torch.float32, device=device)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label.float())

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            ## ----- UPDATE G ----- ##

            # Update G
            if epoch < 10 or errG.item() > threshold:            
                optimizerG.step() 

            # Update learning rate
            schedulerG.step(errG)

            ## ----- UPDATE LISTS ----- ##

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i == len(dataloader) - 1:
                pbar.set_description("Epoch {:03} Loss_D: {:03} Loss_G: {:03} lrG {:.6} lrD {:.6}" \
                    .format(epoch, round(errD.item(),3), round(errG.item(),3), \
                    schedulerG.optimizer.param_groups[0]['lr'], schedulerD.optimizer.param_groups[0]['lr']))     

            inner_pbar.update(batch_size)

            ## ----- UPDATE DISPLAY ----- ##

            if epoch > 0 and epoch % 5 == 0:  

                # Generate a batch of fake images
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake[0:16], padding=2, normalize=normalize))

                slices_fake_scaled = standardize(fake, scalings)

                ## ----- FAKE IMAGES ----- ##
                plt.figure(epoch_fig.number)
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.axis("off")
                plt.title("Fake Images: Epoch " + str(epoch))
                plt.imshow(np.transpose(vutils.make_grid(fake[0:9], padding=2, 
                                                         nrow=3, normalize=normalize), (1,2,0)))
                ## ----- LOSS ----- ##           
                plt.subplot(1, 2, 2)
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(np.abs(G_losses),label="G")
                plt.plot(np.abs(D_losses),label="D")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(fig_path + "SkullGAN_" + now + "/images/" + "progress_epoch_" + str(epoch) + "_.png", dpi=300)
                plt.close()

            iters += 1

        pbar.update(1)

    pbar.close()

    print("Final Generator Loss: " + str(G_losses[-1]))
    print("Final Discriminator Loss: " + str(D_losses[-1]))

    ## ----- SAVE MODEL ----- ##
    torch.save(netG.state_dict(), fig_path + "SkullGAN_" + now + "/models/Generator.pth")
    torch.save(netD.state_dict(), fig_path + "SkullGAN_" + now + "/models/Discriminator.pth")

    ## ----- SAVE GENERATOR PROGRESS ----- ##
    plt.figure(figsize=(8, 6))
    for i in range(9):
        plt.subplot(3, 3, i+1)    
        ifactor = int(np.ceil(len(img_list) / 9))
        plt.imshow(np.transpose(from_t(img_list[-1 if i == 8 else i * ifactor]), (1, 2, 0))[1:384, 1:384])
        title = str(i * ifactor) if i != 8 else str(epoch + 1)
        plt.title('Epoch ' + title + " / " + str(epoch + 1))
        plt.axis('off')

    plt.suptitle('Generator Progress')
    plt.tight_layout()
    plt.savefig(fig_path + "SkullGAN_" + now + "/generator_progress.png", dpi=300)
    plt.close()

    ## ----- EXPORT SAMPLES ----- ##

    # Generate fake image batch with G
    with torch.no_grad():
        test_samples = 100
        test_noise = latent_vec(device, test_samples, nz, gen = latent_vec_gen)
        fake_slices = from_t(standardize(netG(test_noise), scalings)).squeeze()

        # Repeat as desired (loop for memory management purposes)
        for i in range(9):
            test_noise = latent_vec(test_samples, gen=latent_vec_gen)
            fake_slices = np.concatenate((fake_slices, from_t(standardize(netG(test_noise), scalings)).squeeze()))

    skulls = {'slices' : fake_slices}
    np.save(fig_path + "SkullGAN_" + now + "/fake_skulls.npy", fake_slices)
    
def prepare_data(path, fig_path = "", display = False):
    
    ## ----------- LOAD SAMPLES -----------
    ## SLICES               : 456    SAMPLES 
    ## SLICES_2K            : 2414   SAMPLES 
    ## SLICES_RESCALED      : 12074  SAMPLES 
    ## CELEB_50K            : 50000  SAMPLES
    ## CELEB_100k           : 100000 SAMPLES
    ## ------------------------------------

    slices_org = np.load(path)[:200]
    nsamples = len(slices_org)

    ## ----- STANDARDIZE DATA ----- #####
    slices, scalings = standardize(slices_org, [], 0)

    print("Original Range:", round(np.min(slices_org), 3), "-", round(np.max(slices_org), 3))
    print("Standardized Range:", round(np.min(slices), 3), "-", round(np.max(slices), 3))

    if display == True:
                
        ## ----- DISPLAY SKULLS ----- ##
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Max of Real Skulls')
        plt.imshow(np.max(slices.squeeze(), axis=0), cmap='gray')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.title('Average of Real Skulls')
        plt.imshow(np.mean(slices.squeeze(), axis=0), cmap='gray')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.title('Real Skulls Sample')
        plt.imshow(slices[0].squeeze(), cmap='gray')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.savefig(fig_path + "ex_inputs.png", dpi=300)
        plt.close()
    
    return slices, scalings


if __name__ == "__main__":
    
    ## -------------------------------- ##
    ## HYPERPARAMETERS
    ## -------------------------------- ##

    # Figure location
    fig_path = "figures/"
    
    # Training data location
    data_path = "data/slices_2k.npy"

    # Batch size
    batch_size = 16

    # Input image size
    image_size = 128

    # Number of channels (grayscale)
    nc = 1

    # Size of generator input latent vector
    nz = 200

    # Size of feature maps in generator
    ngf = 256

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 1000

    # Learning rate for optimizers
    lrD = 3e-4 
    lrG = 3e-4

    # beta hyperparam for Adam optimizers
    beta = 0.5
                
    # Toggle if using pretrained model
    # pretrain = False
        
    # Pretrained model path
    # pretrained_path = "/home/shared/SkullGAN/models/"
    
    ## -------------------------------- ##
    ## TRAIN SKULLGAN
    ## -------------------------------- ##
                
    data = prepare_data(data_path, fig_path, display = True)
                
    params = [batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lrD, lrG, beta]
                
    main(data, fig_path, params) #, pretrain, pretrained_path)
                
    
