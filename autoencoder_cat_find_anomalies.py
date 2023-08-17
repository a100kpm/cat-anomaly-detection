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
from autoencodeur_cat import neural_AE

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the gpu")
else:
    device = torch.device("cpu")
    print("running on the cpu") 
   


model_state_dict=torch.load("net_ae_state_dict.model")
# model_state_dict=torch.load("model_retrain_9.model")
print('model_state_dict_loaded')
with open("net_ae_architecture.pkl" , 'rb') as f:
    model_architecture = pickle.load(f)
print('model_architecture_loaded')

def keep_lin_for_encodeur(linear_dict):
    """
    Parameters
    ----------
    linear_dict : dictionary
        dictionary of the layers of the model.

    Returns
    -------
    linear_dict_rez : dictionary
        dictionary of the linear layers of the model.

    """
    size=-1
    for x in linear_dict:
        size_temp=len(linear_dict[x])
        if size==-1:
            size=size_temp
            last_word=x.split('.')[0]
        elif size_temp<=size:
            size=size_temp
            last_word=x.split('.')[0]
            
    linear_dict_rez=OrderedDict()
    keep=True
    for x in linear_dict:
        if keep==False and x.split('.')[0]!=last_word:
            break
        linear_dict_rez[x]=linear_dict[x]
        if x.split('.')[0]==last_word:
            keep=False
    return linear_dict_rez
            
    
    
def get_encodeur_state(model_state_dict):
    """
    Parameters
    ----------
    model_state_dict : dictionary
        dictionary of the layers of the model.

    Returns
    -------
    encodeur_dict : dictionary
        dictionary of the layers of the encoder of the ae.

    """
    encodeur_dict=OrderedDict()
    linear_dict=OrderedDict()
    for x in model_state_dict:
        if x[:7]=='conv2dT' or x[:11]=='batch_normT':
            pass
        elif x[:3]=='lin':
            linear_dict[x]=model_state_dict[x]
        else:
            encodeur_dict[x]=model_state_dict[x]
    linear_dict=keep_lin_for_encodeur(linear_dict)
    for x in linear_dict:
        encodeur_dict[x]=linear_dict[x]
    return encodeur_dict

def count_lin_conv(encodeur_set):
    """
    Parameters
    ----------
    encodeur_set : set
        name of the layers of the encoder.

    Returns
    -------
    int
        number of convolutional layers of the encoder.
    compteur : int
        number of linear layers of the encoder.

    """
    lenn=len(encodeur_set)
    compteur=0
    for x in encodeur_set:
        if x[:3]=='lin':
            compteur+=1
    return int((lenn-compteur)/2),compteur

encodeur_dict=get_encodeur_state(model_state_dict)
encodeur_set=set([x.split('.')[0] for x in encodeur_dict])


class encodeur_AE(nn.Module):
    def __init__(self,model_state_dict,model_architecture):
        super(encodeur_AE, self).__init__()
        encodeur_dict=get_encodeur_state(model_state_dict)
        encodeur_set=set([x.split('.')[0] for x in encodeur_dict])
        count_conv,count_lin=count_lin_conv(encodeur_set)
        self.count_conv=count_conv
        self.count_lin=count_lin
        self.encodeur_dict=encodeur_dict
        
        for x in encodeur_set:
            setattr(self, f"{x}",model_architecture[x])

            
    def count_lin_conv(encodeur_set):
        lenn=len(encodeur_set)
        compteur=0
        for x in encodeur_set:
            if x[:3]=='lin':
                compteur+=1

        return int((lenn-compteur)/2),compteur



    def forward(self,x):
        for i in range(1,self.count_conv+1):
            x=F.elu(eval(f"self.batch_norm{i}(self.conv{i}(x))"))
        if self.count_lin>=1:
            x=x.view(-1,np.shape(self.encodeur_dict["lin1.weight"])[1])
            for i in range(1,self.count_lin+1):
                x=F.elu(eval(f"self.lin{i}(x)"))
        return x
            
net_ae_encodeur=encodeur_AE(model_state_dict,model_architecture)
net_ae_encodeur.to(device)
net_ae_encodeur.eval()

train = np.load("cat_train_256_no_noise.pkl", allow_pickle=True)
train=[[x[0],x[2]] for x in train]
new_data=[]

for i in range(0,len(train)):
    img=torch.Tensor(np.array(train[i][0]))
    img=img.view(-1,1,256,256)
    img = img.to(device)
    with torch.no_grad():
        output=net_ae_encodeur(img)
        new_data.append([train[i][0],output.cpu().detach().numpy()])
        
        
    
    
    

encoded_images_cat=[x[1] for x in new_data]
# encoded_images_dog=[x[1] for x in new_data_dog]
encoded_images=encoded_images_cat
# encoded_images=encoded_images_cat+encoded_images_dog
# new_data.extend(new_data_dog)
# train.extend(train_dog)
nsamples, nx, ny = np.shape(encoded_images)
encoded_images = np.reshape(encoded_images,[nsamples,nx*ny])


# =============================================================================
# =============================================================================
# # KDE working more or less
# =============================================================================
# =============================================================================
# log_dens = kde.score_samples(encoded_images)
kde = KernelDensity(kernel='gaussian',bandwidth=0.04).fit(encoded_images)
lenn=len(encoded_images)
list_encoded=[]
for i in range(lenn):
    img_encoded=new_data[i][1]
    list_encoded.append(kde.score_samples(img_encoded)[0])
print(min(list_encoded),max(list_encoded),sum(list_encoded)/lenn)
print(list_encoded[7046],list_encoded[4034]) # two image identified from reconstruction of error

        
        
# list_bad_kernel=[]
# for i in range(lenn):
#     if list_encoded[i]<=max(list_encoded[7046],list_encoded[4034]):
#         list_bad_kernel.append([i,list_encoded[i]])
# print(len(list_bad_kernel))
# name_index=[ [ train[x[0]][1],x[1] ] for x in list_bad_kernel]


# for x in list_bad_kernel:
#     plt.figure()
#     plt.imshow(train[x[0]][0])
#     plt.show()
#     print(train[x[0]][1])

        
        

# =============================================================================
# =============================================================================
# #         optics doesn't work here
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # from sklearn.cluster import OPTICS
# # clustering = OPTICS(min_samples=5,xi=0.0001).fit(encoded_images)
# # a=clustering.labels_
# =============================================================================
# =============================================================================
        

from sklearn.mixture import GaussianMixture
# from sklearn.mixture import BayesianGaussianMixture

gm = GaussianMixture(n_components=2, covariance_type='diag',tol=1e-5,random_state=0,max_iter=200).fit(encoded_images)
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
# print(list_predict_mixture[7046],list_predict_mixture[4034]) # two image identified from reconstruction of error
        
# list_predict_mixture_index=[]    
# for i in range(lenn):
#     if list_predict_mixture[i]>2100:
#         list_predict_mixture_index.append([i,list_encoded[i]])
# name_index=[name[x][2] for x in list_predict_mixture_index]

list_reconstruction_erreur=[]
model = neural_AE()
model.load_state_dict(torch.load("net_ae_state_dict.model"))
model.to(device)
model.eval()
loss_function=nn.MSELoss()
for i in range(lenn):
    X=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)
    y=torch.Tensor(np.array(train[i][0])).view(-1,1,256,256)#train[i][1] if train has noise, but then need not to remove train[i][1] above
    X,y=X.to(device),y.to(device)
    with torch.no_grad():
        output=model(X)
        loss=loss_function(output,y).item()
        list_reconstruction_erreur.append(loss)
    



df=pd.DataFrame(list(zip(list_predict_mixture_class,list_predict_mixture_class_confidence,list_predict_mixture,list_encoded,list_reconstruction_erreur)))
df['index']=range(len(df))
df_0=df[df[0]==0]

# df_0.sort_values(by=[2],ascending=True,inplace=True)#32
# for i in range(14):
#     val=df_0.iloc[i]['index']
#     plt.figure()
#     plt.imshow(train[val][0])
#     plt.show()
#     print(train[val][1])
   
    
# df_0.sort_values(by=[3],ascending=True,inplace=True)#14
# for i in range(14):
#     val=df_0.iloc[i]['index']
#     plt.figure()
#     plt.imshow(train[val][0])
#     plt.show()
#     print(train[val][1])

# df_0.sort_values(by=[4],ascending=False,inplace=True)#32
# for i in range(14):
#     val=df_0.iloc[i]['index']
#     plt.figure()
#     plt.imshow(train[val][0])
#     plt.show()
#     print(train[val][1])

# =============================================================================
# =============================================================================
# # # # experiments to see patern in problematics images
# # # min_df3=df_0[3].min()
# # # max_df3=df_0[3].max()
# # # df_0[3]=df_0[3]-min_df3
# # # df_0[3]=df_0[3]/max_df3
# # # df_0[2]=df_0[2]-df_0[2].min()
# # # df_0[2]=df_0[2]/df_0[2].max()
# # # df_0[4]=df_0[4]-df_0[4].min()
# # # df_0[4]=df_0[4]/df_0[4].max()
# # 
# # # for i in range(1,5):
# # #     plt.figure()
# # #     plt.plot(range(len(df_0)),df_0[i])
# # #     plt.show()
# # 
# #
# # 
# # # df_0.sort_values(by=[3],ascending=True,inplace=True)
# # # mini_df_0=df_0.iloc[:30] #take from sort [3] #4-5-6-9-14-30
# # # for i in range(30):
# # #     val=mini_df_0.iloc[i]['index']
# # #     plt.figure()
# # #     plt.imshow(train[val][0])
# # #     plt.show()
# # #     print(train[val][1])
# # # mini_df_0.sort_values(by=[2],ascending=True,inplace=True)#2-3-6-11-21-29
# # # plt.figure()
# # # plt.plot(mini_df_0[2],mini_df_0[4],marker='o')
# # # # plt.plot(mini_df_0[2],mini_df_0[3])
# # # mini_df_0[5]=mini_df_0[2]-mini_df_0[3]
# # # plt.plot(mini_df_0[2],mini_df_0[5],marker='o')
# # # mini_df_0[6]=(mini_df_0[4]+mini_df_0[5])
# # # plt.plot(mini_df_0[2],mini_df_0[6],marker='o')
# # # plt.show()
# # # list_anomalie=[]
# # 
# # # for i in [2,3,6,11,21,29]:
# # #     list_anomalie.append(mini_df_0.iloc[i-1]['index'])
# #     
# # # df_0[5]=df_0[2]-df_0[3]
# # # df_0[6]=df_0[4]+df_0[5]
# # # df_0['anomalie']=0
# # # df_0['anomalie']=df_0['index'].apply(lambda x:1 if x in list_anomalie else 0)
# # 
# # 
# # 
# # 
# # # df_0.sort_values(by=['anomalie'],ascending=False,inplace=True)
# # # for i in range(6):
# # #     val=df_0.iloc[i]['index']
# # #     plt.figure()
# # #     plt.imshow(train[val][0])
# # #     plt.show()
# # #     print(train[val][1])
# #     
# # # nbr_choix=32
# # # dff.sort_values(by=[2],ascending=True,inplace=True)
# # # list_index1=dff['index'].iloc[:nbr_choix]
# # # dff.sort_values(by=[3],ascending=True,inplace=True)
# # # list_index2=dff['index'].iloc[:nbr_choix]
# # # dff.sort_values(by=[4],ascending=False,inplace=True)
# # # list_index3=dff['index'].iloc[:nbr_choix]
# # 
# # 
# # # list_find_common=list(set(list_index1) & set(list_index2) & set(list_index3))
# # 
# # 
# # # dff=df_0[(df_0[3]<=((max(list_encoded[7046],list_encoded[4034]))-min_df3)/max_df3) & (df_0[4]>=0.174)]
# =============================================================================
# =============================================================================

dff=df_0[ ( (df_0[4]>=0.006893) | (df_0[4]<=0.00180781) ) & (df_0[3]<=915.7296025) ] #number from experiments
# no noise


# list_anomalie=[4034, 3837, 2723, 7046, 491, 5204]

for i in dff['index']:
    plt.figure()
    plt.imshow(train[i][0])
    plt.show()
    
    # 491, 5204
