# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time
import pandas as pd
from collections import OrderedDict
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.covariance import EllipticEnvelope


from vae_cat import VAE

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

vae_model = VAE()
vae_model.load_state_dict(torch.load('vae_model_betavae_1.model'))
          
            
class Encoder_VAE(nn.Module):
    def __init__(self, vae_model):
        super(Encoder_VAE, self).__init__()

        # Encoder layers
        self.encoder = vae_model.encoder

        # Mean and log variance layers
        self.fc_mu = vae_model.fc_mu
        self.fc_logvar = vae_model.fc_logvar

        # Linear layer for z
        self.fc_z = vae_model.fc_z

    def encode(self, x):
        # Pass the input through the encoder layers
        x = self.encoder(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Compute the mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode the input to obtain mu and logvar
        mu, logvar = self.encode(x)

        # Reparameterize to obtain z
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
    
encodeur_vae = Encoder_VAE(vae_model)
encodeur_vae.to(device)
encodeur_vae.eval()
# =============================================================================
# =============================================================================
# # # # verify that the encoder is properly loaded
# # vae_model.to(device)
# # vae_model.eval()
# # with open("cat_valid_256_no_noise.pkl", "rb") as f:
# #     validation = pickle.load(f)
# # img = validation[0][0]
# # img = torch.Tensor(img).view(-1, 1, 256, 256).to(device)
# # vae_predict,mu1,sigma1 = vae_model(img)
# # encoder_vae_predict,mu,sigma = encodeur_vae(img)
# # if torch.equal(mu, mu1):
# #     print("mu and mu1 are exactly the same")
# # else:
# #     print("mu and mu1 are different")
# #     
# # if torch.equal(sigma ,sigma1):
# #     print("sigma and sigma1 are exactly the same")
# # else:
# #     print("sigma and sigma1 are different")
# =============================================================================
# =============================================================================

train = np.load("cat_train_256_no_noise.pkl", allow_pickle=True)
train=[[x[0],x[2]] for x in train]
new_data=[]
new_data_mu=[]
new_data_sigma=[]

for i in range(0,len(train)):
    img=torch.Tensor(np.array(train[i][0]))
    img=img.view(-1,1,256,256)
    img = img.to(device)
    with torch.no_grad():
        output,output_mu,output_sigma=encodeur_vae(img)
        new_data.append([train[i][0],output.cpu().detach().numpy()])
        new_data_mu.append([train[i][0],output_mu.cpu().detach().numpy()])
        new_data_sigma.append([train[i][0],output_sigma.cpu().detach().numpy()])
        
        
       
encoded_images_cat=[x[1] for x in new_data]
encoded_images_cat_mu=[x[1] for x in new_data_mu]
encoded_images_cat_sigma=[x[1] for x in new_data_sigma]

encoded_images=encoded_images_cat
encoded_images_mu=encoded_images_cat_mu
encoded_images_sigma=encoded_images_cat_sigma




def get_bandwidth(encoded_images):
    """
    Parameters
    ----------
    encoded_images : list
        list of all reconstructed image by the model.

    Returns
    -------
    best_bandwidth : float
        best_bandwidth value for kde according to grid_search.

    """
    # Set up the kernel density estimator
    kde = KernelDensity()

    # Define the parameter grid for bandwidth selection
    param_grid = {'bandwidth': np.logspace(-1, 1, 20)}

    # Perform grid search cross-validation
    grid_search = GridSearchCV(kde, param_grid, cv=5)  # 'cv' determines the number of folds

    # Fit the grid search on the data
    grid_search.fit(encoded_images)

    # Get the best bandwidth parameter
    best_bandwidth = grid_search.best_params_['bandwidth']

    # Print the best bandwidth
    print("Best bandwidth:", best_bandwidth)
    return best_bandwidth

# =============================================================================
# =============================================================================
# # KDE then Gaussian mixture
# =============================================================================
# =============================================================================
nsamples, nx, ny = np.shape(encoded_images)
encoded_images = np.reshape(encoded_images,[nsamples,nx*ny])
# log_dens = kde.score_samples(encoded_images)
kde = KernelDensity(kernel='gaussian',bandwidth=10).fit(encoded_images)
kde_z=kde
lenn=len(encoded_images)
list_encoded=[]
for i in range(lenn):
    img_encoded=new_data[i][1]
    list_encoded.append(kde.score_samples(img_encoded)[0])
print(min(list_encoded),max(list_encoded),sum(list_encoded)/lenn,max(list_encoded)-min(list_encoded))
# print(list_encoded[7046],list_encoded[4034]) # two image identified from reconstruction of error of ae
        
        
# list_bad_kernel=[]
# for i in range(lenn):
#     if list_encoded[i]<=max(list_encoded[7046],list_encoded[4034]):
#         list_bad_kernel.append([i,list_encoded[i]])
# print(len(list_bad_kernel))
# name_index=[ [ train[x[0]][1],x[1] ] for x in list_bad_kernel]


df=pd.DataFrame(list_encoded,columns=['list_encoded'])

gm = GaussianMixture(n_components=2, covariance_type='full',tol=1e-5,random_state=0,max_iter=200,n_init=3).fit(encoded_images)
gm_z=gm
# |      - 'full'
# |      - 'tied': all components share the same general covariance matrix.
# |      - 'diag': each component has its own diagonal covariance matrix.
# gm = BayesianGaussianMixture(n_components=2, random_state=0).fit(encoded_images)
# gm.predict(img_enccoded)
list_predict_mixture=[]
list_predict_mixture_class=[]
list_predict_mixture_class_confidence=[]
for i in range(lenn):
    list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
    list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
    list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])

print(min(list_predict_mixture),max(list_predict_mixture),sum(list_predict_mixture)/lenn)
# print(list_predict_mixture[7046],list_predict_mixture[4034]) # two image identified from reconstruction of error of ae
df[['mixture_class','confidence_predict_mixture_class','predict_mixture_class']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# list_predict_mixture_index=[]    
# for i in range(lenn):
#     if list_predict_mixture[i]>2100:
#         list_predict_mixture_index.append([i,list_encoded[i]])
# name_index=[name[x][2] for x in list_predict_mixture_index]
# =============================================================================
# =============================================================================
# KDE and gaussian mixture with MU encoding
# =============================================================================
# =============================================================================
nsamples, nx, ny = np.shape(encoded_images_mu)
encoded_images = np.reshape(encoded_images_mu,[nsamples,nx*ny])
# log_dens = kde.score_samples(encoded_images)
kde = KernelDensity(kernel='gaussian',bandwidth=0.001).fit(encoded_images)
kde_mu=kde
lenn=len(encoded_images)
list_encoded=[]
for i in range(lenn):
    img_encoded=new_data_mu[i][1]
    list_encoded.append(kde.score_samples(img_encoded)[0])
print(min(list_encoded),max(list_encoded),sum(list_encoded)/lenn,max(list_encoded)-min(list_encoded))
# print(list_encoded[7046],list_encoded[4034]) # two image identified from reconstruction of error of ae   
# list_bad_kernel=[]
# for i in range(lenn):
#     if list_encoded[i]<=max(list_encoded[7046],list_encoded[4034]):
#         list_bad_kernel.append([i,list_encoded[i]])
# print(len(list_bad_kernel))
# name_index=[ [ train[x[0]][1],x[1] ] for x in list_bad_kernel]

df['list_encoded_mu']=pd.DataFrame(list_encoded)

gm = GaussianMixture(n_components=2, covariance_type='full',tol=1e-5,random_state=0,max_iter=200,n_init=3).fit(encoded_images)
gm_mu=gm #save gaussian mixture for later
# |      - 'full'
# |      - 'tied': all components share the same general covariance matrix.
# |      - 'diag': each component has its own diagonal covariance matrix.
# gm = BayesianGaussianMixture(n_components=2, random_state=0).fit(encoded_images)
# gm.predict(img_enccoded)
list_predict_mixture=[]
list_predict_mixture_class=[]
list_predict_mixture_class_confidence=[]
for i in range(lenn):
    list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
    list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
    list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])

print(min(list_predict_mixture),max(list_predict_mixture),sum(list_predict_mixture)/lenn)
# print(list_predict_mixture[7046],list_predict_mixture[4034]) # two image identified from reconstruction of error of ae
df[['mixture_class_mu','confidence_predict_mixture_class_mu','predict_mixture_class_mu']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# list_predict_mixture_index=[]    
# for i in range(lenn):
#     if list_predict_mixture[i]>2100:
#         list_predict_mixture_index.append([i,list_encoded[i]])
# name_index=[name[x][2] for x in list_predict_mixture_index]
# =============================================================================
# =============================================================================
# KDE and gaussian mixture with SIGMA (logvar) encoding
# =============================================================================
# =============================================================================
nsamples, nx, ny = np.shape(encoded_images_sigma)
encoded_images = np.reshape(encoded_images_sigma,[nsamples,nx*ny])
# log_dens = kde.score_samples(encoded_images)
kde = KernelDensity(kernel='gaussian',bandwidth=0.0008).fit(encoded_images)
kde_sigma=kde
lenn=len(encoded_images)
list_encoded=[]
for i in range(lenn):
    img_encoded=new_data_sigma[i][1]
    list_encoded.append(kde.score_samples(img_encoded)[0])
print(min(list_encoded),max(list_encoded),sum(list_encoded)/lenn,max(list_encoded)-min(list_encoded))
# print(list_encoded[7046],list_encoded[4034]) # two image identified from reconstruction of error of ae       
        
# list_bad_kernel=[]
# for i in range(lenn):
#     if list_encoded[i]<=max(list_encoded[7046],list_encoded[4034]):
#         list_bad_kernel.append([i,list_encoded[i]])
# print(len(list_bad_kernel))
# name_index=[ [ train[x[0]][1],x[1] ] for x in list_bad_kernel]


df['list_encoded_sigma']=pd.DataFrame(list_encoded)

gm = GaussianMixture(n_components=2, covariance_type='full',tol=1e-5,random_state=0,max_iter=200,n_init=3).fit(encoded_images)
gm_sigma=gm #save gaussian mixture for later
# |      - 'full'
# |      - 'tied': all components share the same general covariance matrix.
# |      - 'diag': each component has its own diagonal covariance matrix.
# gm = BayesianGaussianMixture(n_components=2, random_state=0).fit(encoded_images)
# gm.predict(img_enccoded)
list_predict_mixture=[]
list_predict_mixture_class=[]
list_predict_mixture_class_confidence=[]
for i in range(lenn):
    list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
    list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
    list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])

print(min(list_predict_mixture),max(list_predict_mixture),sum(list_predict_mixture)/lenn)
# print(list_predict_mixture[7046],list_predict_mixture[4034]) # two image identified from reconstruction of error of ae
df[['mixture_class_sigma','confidence_predict_mixture_class_sigma','predict_mixture_class_sigma']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# list_predict_mixture_index=[]    
# for i in range(lenn):
#     if list_predict_mixture[i]>2100:
#         list_predict_mixture_index.append([i,list_encoded[i]])
# name_index=[name[x][2] for x in list_predict_mixture_index]




list_reconstruction_erreur=[]
vae_model.to(device)
vae_model.eval()
loss_function=nn.MSELoss()
for i in range(lenn):
    X=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)
    y=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)#train[i][1] if train has noise, but then need not to remove train[i][1] above
    X,y=X.to(device),y.to(device)
    with torch.no_grad():
        output,_,_=vae_model(X)
        loss=loss_function(output,y).item()
        list_reconstruction_erreur.append(loss)
    
df['reconstruction_error']=pd.DataFrame(list_reconstruction_erreur)


df['index']=df.index
df.sort_values(by=['reconstruction_error'],ascending=False,inplace=True)
#  1 -  2 -  3 - 11 - 12 -15 - 21 - 38 --> they are anomalies
#4034-2723-3837-2127-491-6239-5204-7046
lst=[4034,2723,3837,2127,491,6239,5204,7046]
df['anom']=df['index'].apply(lambda x: 1 if x in lst else 0)

muu=[x.reshape(256) for x in encoded_images_mu]
# Example using latent space mu
model_el = EllipticEnvelope(contamination=0.002) #didn't seem to work for reconstruction or sigma
model_el.fit(muu)
outlier_predictions = model_el.predict(muu)
# ok if error big enough (3837 & 7046)
df['mu_predict']=outlier_predictions



                                 
dff=df[(df['reconstruction_error']>=0.1760) & (df['mixture_class_sigma']==1)]
dff.drop(columns=['mixture_class','confidence_predict_mixture_class','mixture_class_mu','confidence_predict_mixture_class_mu'],inplace=True)
dff.drop(columns=['confidence_predict_mixture_class_sigma','mixture_class_sigma'],inplace=True)
   
df_test=dff[(dff['reconstruction_error']>=0.274887263774871)|(dff['list_encoded_sigma']<=1581.34417984892)]
df_test=df_test[df_test['predict_mixture_class_sigma']<=-114667525.21696]
df_test=df_test[ (df_test['list_encoded_mu']<=1524.2464783405) | (df_test['reconstruction_error'] >=0.204174175858497) ]

df_save=df.copy()
df_describe=df.describe()



# =============================================================================
# =============================================================================
# # # basically the same with validation dataset -no new anomlies will be found-
# # # but a lot of "poor" images will be found regardless
# # with open("cat_valid_256_no_noise.pkl", "rb") as f:
# #     validation = pickle.load(f)
# # 
# # train=[[x[0],x[2]] for x in validation]
# # new_data=[]
# # new_data_mu=[]
# # new_data_sigma=[]
# # lenn=len(train)
# # 
# # for i in range(0,len(train)):
# #     img=torch.Tensor(np.array(train[i][0]))
# #     img=img.view(-1,1,256,256)
# #     img = img.to(device)
# #     with torch.no_grad():
# #         output,output_mu,output_sigma=encodeur_vae(img)
# #         new_data.append([train[i][0],output.cpu().detach().numpy()])
# #         new_data_mu.append([train[i][0],output_mu.cpu().detach().numpy()])
# #         new_data_sigma.append([train[i][0],output_sigma.cpu().detach().numpy()])
# #         
# #         
# #        
# # encoded_images_cat=[x[1] for x in new_data]
# # encoded_images_cat_mu=[x[1] for x in new_data_mu]
# # encoded_images_cat_sigma=[x[1] for x in new_data_sigma]
# # 
# # encoded_images=encoded_images_cat
# # encoded_images_mu=encoded_images_cat_mu
# # encoded_images_sigma=encoded_images_cat_sigma
# # 
# # 
# # list_reconstruction_erreur=[]
# # vae_model.to(device)
# # vae_model.eval()
# # loss_function=nn.MSELoss()
# # for i in range(lenn):
# #     X=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)
# #     y=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)#train[i][1] if train has noise, but then need not to remove train[i][1] above
# #     X,y=X.to(device),y.to(device)
# #     with torch.no_grad():
# #         output,_,_=vae_model(X)
# #         loss=loss_function(output,y).item()
# #         list_reconstruction_erreur.append(loss)
# # 
# # df_valid=pd.DataFrame(list_reconstruction_erreur,columns=['reconstruction_error'])
# # df_valid['index']=df_valid.index
# # 
# # 
# # 
# # 
# # nsamples, nx, ny = np.shape(encoded_images)
# # encoded_images = np.reshape(encoded_images,[nsamples,nx*ny])
# # lenn=len(encoded_images)
# # list_encoded=[]
# # for i in range(lenn):
# #     img_encoded=new_data[i][1]
# #     list_encoded.append(kde.score_samples(img_encoded)[0])
# # df_valid['list_encoded']=pd.DataFrame(list_encoded)
# # 
# # 
# # nsamples, nx, ny = np.shape(encoded_images_mu)
# # encoded_images = np.reshape(encoded_images_mu,[nsamples,nx*ny])
# # lenn=len(encoded_images)
# # list_encoded=[]
# # for i in range(lenn):
# #     img_encoded=new_data_mu[i][1]
# #     list_encoded.append(kde_mu.score_samples(img_encoded)[0])
# # df_valid['list_encoded_mu']=pd.DataFrame(list_encoded)
# # 
# # nsamples, nx, ny = np.shape(encoded_images_sigma)
# # encoded_images = np.reshape(encoded_images_sigma,[nsamples,nx*ny])
# # lenn=len(encoded_images)
# # list_encoded=[]
# # for i in range(lenn):
# #     img_encoded=new_data_sigma[i][1]
# #     list_encoded.append(kde_sigma.score_samples(img_encoded)[0])
# # df_valid['list_encoded_sigma']=pd.DataFrame(list_encoded)
# # df_valid_describe=df_valid.describe()
# # 
# # 
# # list_predict_mixture=[]
# # list_predict_mixture_class=[]
# # list_predict_mixture_class_confidence=[]
# # gm=gm_z
# # for i in range(lenn):
# #     list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
# #     list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
# #     list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])
# # df_valid[['mixture_class','confidence_predict_mixture_class','predict_mixture_class']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# # 
# # list_predict_mixture=[]
# # list_predict_mixture_class=[]
# # list_predict_mixture_class_confidence=[]
# # gm=gm_mu
# # for i in range(lenn):
# #     list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
# #     list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
# #     list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])
# # df_valid[['mixture_class_mu','confidence_predict_mixture_class_mu','predict_mixture_class_mu']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# # 
# # list_predict_mixture=[]
# # list_predict_mixture_class=[]
# # list_predict_mixture_class_confidence=[]
# # gm=gm_sigma
# # for i in range(lenn):
# #     list_predict_mixture_class.append(gm.predict(new_data[i][1])[0])
# #     list_predict_mixture_class_confidence.append(gm.predict_proba(new_data[i][1])[0])
# #     list_predict_mixture.append(gm.score_samples(new_data[i][1])[0])
# # df_valid[['mixture_class_sigma','confidence_predict_mixture_class_sigma','predict_mixture_class_sigma']]=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture)))        
# # 
# # df_valid_describe=df_valid.describe()
# # 
# # 
# # dff_valid=df_valid[(df_valid['reconstruction_error']>=0.1760) & (df_valid['mixture_class_sigma']==1)]
# # df_test_valid=dff_valid[(dff_valid['reconstruction_error']>=0.274887263774871)|(dff_valid['list_encoded_sigma']<=1581.34417984892)]
# # df_test_valid=df_test_valid[df_test_valid['predict_mixture_class_sigma']<=-114667525.21696]
# # df_test_valid=df_test_valid[ (df_test_valid['list_encoded_mu']<=1524.2464783405) | (df_test_valid['reconstruction_error'] >=0.204174175858497) ]
# =============================================================================
# 
# =============================================================================
