# -*- coding: utf-8 -*-
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt



torch.manual_seed(11)
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=4, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1) 
        
        self.end_layer_1=nn.Linear(8192,512)
        self.end_layer_2=nn.Linear(512,1)


        
        self.dropout=nn.Dropout(p=0.2)
        
    def forward(self,x):
        x=F.leaky_relu((self.conv1(x)))
        x=F.leaky_relu((self.conv2(x)))
        x=F.leaky_relu((self.conv3(x)))
        x=F.leaky_relu((self.conv4(x)))

        x = x.view(x.size(0), -1)
        x=F.leaky_relu(self.dropout(self.end_layer_1(x)))
        x=self.end_layer_2(x)
        return x
    
# =============================================================================
# =============================================================================
# # # dimensions test zone
# # discrimineur=Discriminator()
# # discrimineur.to(device)
# # img=np.zeros([1,256,256])
# # img=torch.Tensor(img)
# # img=img.view(-1,1,256,256).to(device)
# # a=discrimineur(img)
# # print(np.shape(a))
# =============================================================================
# =============================================================================

class Generator(nn.Module):
    def __init__(self, latent_space=256):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(latent_space, latent_space*8)
        self.fc2 = nn.Linear(latent_space*8, 16 * 16 * 128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,bias=False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1,bias=False)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1,bias=False)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1,bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(-1, 128, 16, 16)
        x = F.leaky_relu(self.deconv1(x))  # Output size: 64x64
        x = F.leaky_relu(self.deconv2(x)) # Output size: 128x128
        x = F.leaky_relu(self.deconv3(x)) # Output size: 256x256
        x = F.tanh(self.deconv4(x))  # Output size: 256x256
        return x
# note: to improve generation quality one may replace the ConvTranspose2d layers
# https://stackoverflow.com/questions/59101333/how-to-get-rid-of-checkerboard-artifacts
    
# =============================================================================
# =============================================================================
# # # dimensions test zone
# # latent_dim=256
# # batch_size=1
# # generateur = Generator(latent_space=latent_dim)
# # generateur.to(device)
# # latent_img=torch.randn(batch_size, latent_dim)
# # latent_img=torch.Tensor(latent_img)
# # latent_img=latent_img.view(-1,256).to(device)
# # a=generateur(latent_img)
# # print(np.shape(a))
# # ===========================================================================
# 
# =============================================================================



def gradient_penalty(netD, real_img, fake_img, LAMDA=10, cuda=True):
    """
    Parameters
    ----------
    netD : 
        model discriminator of the gan.
    real_img : 
        real image.
    fake_img : 
        generated image.
    LAMDA : float, optional
        coefficient of the gradient penalty. The default is 10.
    cuda : bool, optional
        if true use gpu. The default is True.

    Returns
    -------
    float
        gradient penalty value.

    """
    batch_size = real_img.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha= alpha.expand(batch_size, real_img.nelement()//batch_size).reshape(real_img.shape)
    if cuda:
        alpha = alpha.cuda()
    x = (alpha * real_img + (1-alpha) * fake_img).requires_grad_(True)
    if cuda:
        x = x.cuda()
    out = netD(x)
    
    grad_outputs = torch.ones(out.shape)
    if cuda:
        grad_outputs = grad_outputs.cuda()
        
    gradients = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    
    return LAMDA * ((gradients.norm(2, dim=1)-1)**2).mean()



        
def weights_init(m):
    """
    Parameters
    ----------
    m : 
        gan model.

    Returns
    -------
    None.
        modify the initial weights of the model

    """
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        # Initialize the weights with a normal distribution (mean=0, std=0.02)
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        # Initialize the biases to zero if they exist
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif 'BatchNorm' in classname:
        # Initialize the BatchNorm weights with mean=1 and std=0.02
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        # Initialize the BatchNorm biases to zero
        nn.init.constant_(m.bias.data, 0.0)


def train_wgangp(generator, discriminator, train, latent_dim, device, BATCH_SIZE=32, EPOCHS=50,
                    noise_std = 0.1,LAMBDA=10,n_critic_training=5,
                    model_name=None, LR_D=1e-4,LR_G=1e-4, decay=1e-6, 
                    step_gamma=5, GAMMA=0.89,
                    amsgrad=False,logs=False,save=False):
    # note they are reference to the fid score, but the fid score is not implemented
    # fid score will then always value 0
    """
    Parameters
    ----------
    generator : 
        generator model.
    discriminator : 
        discriminator model.
    train : list
        training data.
    latent_dim : int
        size of the latent dim.
    device : 
        cuda device.
    BATCH_SIZE : int, optional
        batch size. The default is 32.
    EPOCHS : int, optional
        number of epochs. The default is 50.
    noise_std : float, optional
        standard deviation of the noise. The default is 0.1.
    LAMBDA : float, optional
        weight of the gradient penalty. The default is 10.
    n_critic_training : int, optional
        number of train of the discriminator for 1 train of the generator. The default is 5.
    model_name : str, optional
        model_name, if None one will be attributed based on time. The default is None.
    LR_D : float, optional
        Learning Rate of the discriminator. The default is 0.004.
    LR_G : float, optional
        Learning Rate of the generator. The default is 0.004.
    decay : float, optional
        decay. The default is 1e-6.
    step_gamma : int, optional
        Number of epochs before applying gamma's multiplier with the scheduler. The default is 5.
    GAMMA : float, optional
        Value to multiply the learning rate wwith after step_gamma epoch. The default is 0.89.
    amsgrad : bool, optional
        Use or not amsgrad parameter of Adam. The default is False.
    logs : bool, optional
        log the training if True. The default is False.
    save : bool, optional
        save the model every 10 epochs if true. The default is False.

    Returns
    -------
    None.

    """

    if model_name is None:
        model_name = f"model_wgangp_LR={LR_D}-{LR_G}-{int(time.time())}"

    # Define optimizers
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G)
    # Define scheduler
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=step_gamma, gamma=GAMMA)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=step_gamma, gamma=GAMMA)


    
    # Training loop
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        fid_score_avg=0
        
        if save==True:
            if epoch!=0 and epoch%10==0:
                torch.save(generator.state_dict(), f'generator_wgapgp_epoch_{epoch}.model')
                torch.save(discriminator.state_dict(), f'discriminator_wgapgp_epoch_{epoch}.model')
                
###############################################################################
        for j in range(n_critic_training-1):
            for i in tqdm(range(0, len(train), BATCH_SIZE)):
                real_images = torch.Tensor(np.array(train[i:i+BATCH_SIZE])).view(-1, 1, 256, 256).to(device)
                batch_size = real_images.size(0)

                # Train discriminator
                optimizer_D.zero_grad()
                
                # Generate noise for the discriminator
                noise = torch.randn_like(real_images) * noise_std
                
                # Add noise to the real images
                noisy_real_images = real_images + noise
                #noise is important to avoid mode collapse
    
                # Generate fake images
                latent_vector = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(latent_vector)
    
                # Adversarial loss for real and fake images
    
                real_predictions = discriminator(noisy_real_images)
                fake_predictions = discriminator(fake_images.detach())
                real_loss = real_predictions.mean()
                fake_loss = fake_predictions.mean()
                d_loss = -real_loss + fake_loss
                
                # Gradient Penalty
                gp = gradient_penalty(discriminator, real_images, fake_images, LAMDA=LAMBDA, cuda=True)
                d_loss += gp
                
                d_loss.backward()
                optimizer_D.step()
            print(f"{j+1}th out of {n_critic_training} train of the discriminator")
###############################################################################

        for i in tqdm(range(0, len(train), BATCH_SIZE)):
            real_images = torch.Tensor(np.array(train[i:i+BATCH_SIZE])).view(-1, 1, 256, 256).to(device)
            batch_size = real_images.size(0)

###############################################################################
            # Train discriminator
            optimizer_D.zero_grad()
            
            # Generate noise for the discriminator
            noise = torch.randn_like(real_images) * noise_std
            
            # Add noise to the real images
            noisy_real_images = real_images + noise

            # Generate fake images
            latent_vector = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(latent_vector)

            # Adversarial loss for real and fake images

            real_predictions = discriminator(noisy_real_images)
            fake_predictions = discriminator(fake_images.detach())
            real_loss = real_predictions.mean()
            fake_loss = fake_predictions.mean()
            d_loss = -real_loss + fake_loss
            
            # Gradient Penalty
            gp = gradient_penalty(discriminator, real_images, fake_images, LAMDA=LAMBDA, cuda=True)
            d_loss += gp
            
            d_loss.backward()
            optimizer_D.step()
###############################################################################
            # Train generator
            optimizer_G.zero_grad()

            # Generate fake images again
            latent_vector = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(latent_vector)

            # Adversarial loss for fake images (generator)
            g_loss = -discriminator(fake_images).mean()

            g_loss.backward()
            optimizer_G.step()
###############################################################################

        
        
        # Update scheduler
        scheduler_D.step()
        scheduler_G.step()
        

        # Print training metrics
        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Discriminator Loss: {d_loss.item():.4f}, "
              f"Generator Loss: {g_loss.item():.4f}, "
              f"Gradient penalty Loss: {gp:4f}")


        if logs:
            with open("wgan-gp_logs.txt", "a") as f:
                f.write(f"model name = {model_name}, Epoch:{epoch}, "
                        f"Discriminator Loss: {d_loss.item()}, "
                        f"Generator Loss: {g_loss.item()}, "
                        f"Gradient penalty Loss: {gp}\n")

    return model_name



latent_space = 1024
# note : gpu memory might be too small for that model, in order to reduce the size
# with minimum effort, bring latent_space to ~512 and/or reduce the size of the layer
# fc1 from the generator and/or reduce the size of end_layer2 of the discriminator
G = Generator(latent_space)
G.apply(weights_init)
D = Discriminator()
D.apply(weights_init)
G = G.to(device)
D = D.to(device)

# Load the dataset
with open("cat_train_256_no_noise.pkl", "rb") as f:
    train = pickle.load(f)
train=[x[0] for x in train]
# torch.cuda.empty_cache()

# latent_space = 512
# G = Generator(latent_space)
# D = Discriminator()
# G.load_state_dict(torch.load("generator_wgapgp_epoch_8603.model"))
# D.load_state_dict(torch.load("discriminator_wgapgp_epoch_8603.model"))
# G = G.to(device)
# D = D.to(device)

model_name=train_wgangp(G, D, train, latent_space, device, BATCH_SIZE=96, EPOCHS=1000,
                    noise_std = 0.01,LAMBDA=10,n_critic_training=5,
                    model_name=None, LR_D=1e-6,LR_G=5e-6, decay=1e-6, 
                    step_gamma=10, GAMMA=0.93,
                    amsgrad=False,logs=True,save=True)






# =============================================================================
# =============================================================================
# # # Save the trained model
# # # torch.save(G.state_dict(), 'generator_wassersteingan.model')
# # # torch.save(D.state_dict(), 'discriminator_wassersteingan.model')
# =============================================================================
# 
# =============================================================================


