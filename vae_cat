# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(11)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VAE model architecture
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),  # Batch Normalization
            nn.ELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # Mean and log variance layers
        self.middle=8192
        self.fc_mu = nn.Linear(self.middle, 256)#256 for the first models
        self.fc_logvar = nn.Linear(self.middle, 256)
        self.fc_z = nn.Linear(256, 512 * 4 * 4)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)  # Move eps to GPU if available
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(z.size(0), 512, 4, 4)  # Reshape back to image dimensions
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# =============================================================================
# =============================================================================
# # # # test the dimension output
# # # net_vae=VAE()
# # # net_vae.to(device)
# # # img=np.zeros([1,256,256])
# # # img=torch.Tensor(img)
# # # img=img.view(-1,1,256,256).to(device)
# # # a,b,c=net_vae(img)
# # # print(np.shape(a))
# =============================================================================
# =============================================================================
def get_lr(optimizer):
    """
    Parameters
    ----------
    optimizer : 
        optimizer currently used for training.

    Returns
    -------
    param_group['lr']
        float value of the current learning_rate.

    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_vae_loss(reconstructed, mu, logvar, beta_vae, data, reconstruction_loss):
    """
    Parameters
    ----------
    reconstructed :
        reconstruction of the data fed to the model.
    mu : 
        1st middle layer of the vae parallel to logvar.
    logvar :
        2nd middle layer of the vae parallel to mu.
    beta_vae : float
        coefficient to weigh kl divergence compared to reconstruction loss.
    data :
        data fed to the model.
    reconstruction_loss : 
        loss function for the reconstruction of the image.

    Returns
    -------
    loss : float
        vae loss of the model.

    """
    recon_loss = reconstruction_loss(reconstructed, data)

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # The KL divergence measures the dissimilarity between the latent space distribution and a known target distribution.
    # It encourages the latent space to follow a normal distribution, promoting a compact and smooth latent representation.
    # The formula for KL divergence in VAEs comes from the assumption that the latent space should follow a normal distribution.

    loss = recon_loss + beta_vae*kl_divergence
    # quantifies the overall quality of the VAE's reconstruction and the regularity of the learned latent space.
    return loss

def find_loss(model,data,loss_function=nn.MSELoss(),beta_vae=1):
    """
    Parameters
    ----------
    model : 
        vae model.
    data : list
        data on the format the model has been trained on.
    loss_function : optional
        loss function. The default is nn.MSELoss().
    beta_vae : float, optional
        coefficient to weigh kl divergence compared to reconstruction loss. The default is 1.

    Returns
    -------
    rez : list
        list of all individual reconstruction losses

    """
    test_X=[x[0] for x in data]
    test_y=[x[1] for x in data]
    model.eval()
    test_loss=0
    lenn=len(test_X)
    rez=[]
    for i in range(0,lenn):
        batch_X=torch.Tensor(np.array(test_X[i]))
        batch_y=torch.Tensor(np.array(test_y[i]))
        batch_X=batch_X.view(-1,1,256,256)
        batch_y=batch_y.view(-1,1,256,256)
        batch_X,batch_y = batch_X.to(device),batch_y.to(device)
        with torch.no_grad():
            reconstructed, mu, logvar=model(batch_X)
            recon_loss = loss_function(reconstructed, batch_y)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta_vae*kl_divergence
            rez.append(loss.item())
            test_loss+=loss
    test_loss/=lenn
    print(f"cat loss {round(test_loss.item(),5)}")
    return rez


    
def train_valid_vae(vae,train,validation,device,BATCH_SIZE=32,EPOCHS=50,model_name=None,LR=0.004,
                    decay=1e-6,step_gamma=5,GAMMA=0.89,amsgrad=False,beta_vae=1,logs=True):
    """
    Parameters
    ----------
    vae : 
        vae to train.
    train :
        training data.
    validation :
        validation data.
    device :
        GPU or CPU (see "torch.cuda.is_available()").
    BATCH_SIZE : int, optional
        batch_size. The default is 32.
    EPOCHS : int, optional
        number of epochs to train. The default is 50.
    model_name : str, optional
        model_name, if None one will be attributed based on time. The default is None.
    LR : float, optional
        Learning Rate. The default is 0.004.
    decay : float, optional
        decay. The default is 1e-6.
    step_gamma : int, optional
        Number of epochs before applying gamma's multiplier with the scheduler. The default is 5.
    GAMMA : float, optional
        Value to multiply the learning rate wwith after step_gamma epoch. The default is 0.89.
    amsgrad : bool, optional
        Use or not amsgrad parameter of Adam. The default is False.
    beta_vae : float, optional
        weight coefficient of the kl divergence. The default is 1.
    logs : bool, optional
        log the training if True. The default is True.

    Returns
    -------
    model_name : str
        model_name to read log/to save model.

    """
    if model_name==None:
        model_name=f"model_vae_LR={LR}_beta_vae={beta_vae}-{int(time.time())}"
    # Define optimizer
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=step_gamma, gamma=GAMMA)
    reconstruction_loss = nn.MSELoss()
    
    # Training loop
  
    for epoch in range(EPOCHS):
        vae.train() # Set the model in training mode
        for i in tqdm(range(0,len(train),BATCH_SIZE)):
    
            data = torch.Tensor(np.array(train[i:i+BATCH_SIZE])).view(-1, 1, 256, 256).to(device)
    
            # Forward pass
            reconstructed, mu, logvar = vae(data)
 
            loss = compute_vae_loss(reconstructed, mu, logvar, beta_vae, data, reconstruction_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute validation loss
        vae.eval()  # Set the model in evaluation mode
        val_loss = 0.0
        nbr_batch=0
        val_loss=0
        recon_loss=0
        with torch.no_grad():
            for j in range(0,len(validation),BATCH_SIZE):
                nbr_batch+=1
                batch_X=torch.Tensor(np.array(validation[j:j+BATCH_SIZE])).view(-1,1,256,256).to(device)
                val_reconstructed, val_mu, val_logvar=vae(batch_X)
                val_batch_loss = compute_vae_loss(val_reconstructed, val_mu, val_logvar, beta_vae, batch_X, reconstruction_loss)
                val_loss += val_batch_loss.item()
                recon_loss += reconstruction_loss(val_reconstructed, batch_X)
        val_loss/=nbr_batch
        recon_loss/=nbr_batch
        # Print training and validation loss
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], lr={get_lr(optimizer)}, Training Loss: {loss.item():.4f},\,"
              f"Validation Loss: {val_loss:.4f}, reconstruction loss: {recon_loss:.4f}") 
        if logs==True:
            with open("log_vae.log","a") as f:
                f.write(f"model name = {model_name}, Epoch:{epoch},Loss: {loss}, Val_loss: {val_loss}\n")    
    return model_name

# =============================================================================
# # Load the dataset
# with open("cat_train_256_no_noise.pkl", "rb") as f:
#     train = pickle.load(f)
# train_vae=[x[0] for x in train]
# with open("cat_valid_256_no_noise.pkl", "rb") as f:
#     validation = pickle.load(f)
# validation_vae=[x[0] for x in validation]
# =============================================================================

# Initialize the VAE model and move it to GPU if available
# =============================================================================
# vae = VAE().to(device)
# model_name=train_valid_vae(vae,train_vae,validation_vae,device,LR=0.0005,EPOCHS=10,decay=1e-6,beta_vae=1,
#                            step_gamma=10,GAMMA=0.89,amsgrad=False,logs=False)
# model_name=train_valid_vae(vae,train_vae,validation_vae,device,LR=0.0001,EPOCHS=10,decay=1e-6,beta_vae=1,
#                            step_gamma=10,GAMMA=0.89,amsgrad=False,logs=False)
# model_name=train_valid_vae(vae,train_vae,validation_vae,device,LR=0.00001,EPOCHS=10,decay=1e-6,beta_vae=1,
#                            step_gamma=10,GAMMA=0.89,amsgrad=False,logs=False)
# model_name=train_valid_vae(vae,train_vae,validation_vae,device,LR=0.000001,EPOCHS=10,decay=1e-6,beta_vae=1,
#                            step_gamma=10,GAMMA=0.89,amsgrad=False,logs=False)
# =============================================================================
# this small training meant for exploration was accidentally very good at finding outliers !
# (and horrible at reconstruction)
# generally we need to train with beta_vae = 0 and progressively increase the value





# =============================================================================
# =============================================================================
# =============================================================================
# # # # Save the trained model
# # # torch.save(vae.state_dict(), 'vae_model_betavae_2.model')
# # # 
# # # 
# # # # Load the pretrained VAE model
# # # vae = VAE().to(device)
# # # vae.load_state_dict(torch.load('vae_model_betavae_1.model'))
# # # vae.eval()  # Set the model in evaluation mode
# # # 
# # # # Pass the test image through the VAE
# # # with torch.no_grad():
# # #     reconstructed, _, _ = vae(test_tensor)
# # # =============================================================================
# =============================================================================
# 
# =============================================================================


# # # # test on image zone
# =============================================================================
# =============================================================================
# # name=train
# # rez=find_loss(vae,name)
# # rez=pd.DataFrame(rez)
# # rez=rez.sort_values(0,ascending=False)
# # for x in rez.index[:38]: # 38 is the last value to get all anomalies found by the ae
# #     plt.figure()
# #     plt.imshow(name[x][0])
# #     plt.show()
# #     print(name[x][2])
# =============================================================================
# =============================================================================

