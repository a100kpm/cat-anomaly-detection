# -*- coding: utf-8 -*-
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd



class neural_AE(nn.Module):
    def __init__(self):
        super(neural_AE, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.maxpool3= nn.MaxPool2d(2,stride=2,return_indices=True)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.maxpool6= nn.MaxPool2d(2,stride=2,return_indices=True)
        
        self.batch_norm1=nn.BatchNorm2d(16)
        self.batch_norm2=nn.BatchNorm2d(32)
        self.batch_norm3=nn.BatchNorm2d(64)
        self.batch_norm4=nn.BatchNorm2d(128)
        self.batch_norm5=nn.BatchNorm2d(256)
        self.batch_norm6=nn.BatchNorm2d(512)

        ## middle layers ##
        self.middle_ext=512#8192
        self.middle=512#400
        self.lin1=nn.Linear(self.middle_ext,self.middle)
        self.lin2=nn.Linear(self.middle,self.middle_ext)
            
        ## decoder layers ##
        self.maxunpool6=nn.MaxUnpool2d(2, stride=2, padding=0)
        self.conv2dT6=nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.conv2dT5=nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2)
        self.conv2dT4=nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.maxunpool3=nn.MaxUnpool2d(2, stride=2, padding=0)
        self.conv2dT3=nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv2dT2=nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.conv2dT1=nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)
        
        self.batch_normT6=nn.BatchNorm2d(256)
        self.batch_normT5=nn.BatchNorm2d(128)
        self.batch_normT4=nn.BatchNorm2d(64)
        self.batch_normT3=nn.BatchNorm2d(32)
        self.batch_normT2=nn.BatchNorm2d(16)
        self.batch_normT1=nn.BatchNorm2d(1)

        
    def encodeur(self,x):
        x=F.elu(self.batch_norm1(self.conv1(x)))
        x=F.elu(self.batch_norm2(self.conv2(x)))
        x=F.elu(self.batch_norm3(self.conv3(x)))
        x,indice3=self.maxpool3(x)
        x=F.elu(self.batch_norm4(self.conv4(x)))
        x=F.elu(self.batch_norm5(self.conv5(x)))
        x=F.elu(self.batch_norm6(self.conv6(x)))
        x,indice6=self.maxpool6(x)
        return x,indice3,indice6
    
    def decodeur(self,x,indice3,indice6):
        x=self.maxunpool6(x,indice6)
        x=F.elu(self.batch_normT6(self.conv2dT6(x)))
        x=F.elu(self.batch_normT5(self.conv2dT5(x)))
        x=F.elu(self.batch_normT4(self.conv2dT4(x)))
        x=self.maxunpool3(x,indice3)
        x=F.elu(self.batch_normT3(self.conv2dT3(x)))
        x=F.elu(self.batch_normT2(self.conv2dT2(x)))
        x=F.tanh((self.conv2dT1(x)))
        return x

    def forward(self, x):
        x,indice3,indice6=self.encodeur(x)
        x=x.view(-1,self.middle_ext)
        x=F.elu(self.lin1(x))
        # x=self.dropout(x)
        x=F.elu(self.lin2(x))
        # # x=self.dropout(x)
        x=x.view(-1,self.middle_ext,1,1) #was 512,4,4 without pooling
        x=self.decodeur(x,indice3,indice6)
        return x

# =============================================================================
# # test the dimension output    
# net_ae=neural_AE()
# net_ae.to(device)
# img=torch.Tensor(np.array(train[0][0]))
# print(np.shape(net_ae(img.view(-1,1,256,256).to(device))))
# =============================================================================
    
def get_lr(optimizer):
    """
    Parameters
    ----------
    optimizer
        optimizer currently used for training.
    
    Returns
    -------
    param_group['lr']
        float value of the current learning_rate.

    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_valid(net_ae,train,valid,device,ratio_validation=0.5,BATCH_SIZE=32,EPOCHS=50,
                model_name=None,LR=0.004,decay=1e-6,step_gamma=5,GAMMA=0.89,amsgrad=False,
                log_path="log_ae.log"):
    """
    Parameters
    ----------
    net_ae : 
        auto_encoder to train.
    train :
        training data.
    valid :
        validation data.
    device :
        GPU or CPU (see "torch.cuda.is_available()").
    ratio_validation : float, optional
        How much of the validation set we use on each epoch. The default is 0.5.
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

    Returns
    -------
    model_name : str
        model_name to read log/to save model.

    """
    if model_name==None:
        model_name=f"model-{int(time.time())}"
    optimizer = optim.Adam(net_ae.parameters(),lr=LR,weight_decay=decay,amsgrad=amsgrad,betas=(0.95, 0.9999))
    scheduler = StepLR(optimizer, step_size=step_gamma, gamma=GAMMA)
    loss_function=nn.MSELoss()
    
    tqdm._instances.clear()
    X=[x[0] for x in train]
    y=[x[1] for x in train]
    
    X=torch.Tensor( np.array( [i for i in X] ) ).view(-1,1,256,256)
    y=torch.Tensor( np.array( [i for i in y] ) ).view(-1,1,256,256)
    
    for epoch in range(EPOCHS):
        #train
        print(f"lr={get_lr(optimizer)}")
        net_ae.train()
        for i in tqdm(range(0,len(train),BATCH_SIZE)):

            batch_X=X[i:i+BATCH_SIZE].view(-1,1,256,256)
            batch_y=y[i:i+BATCH_SIZE].view(-1,1,256,256)
            
            batch_X,batch_y = batch_X.to(device),batch_y.to(device)
            #zero gradient
            net_ae.zero_grad()
            outputs=net_ae(batch_X)
            loss=loss_function(outputs,batch_y)
            loss.backward()
            optimizer.step()
            
        #eval on validation
        net_ae.eval() 
        np.random.shuffle(valid)
        validation=valid[:int(ratio_validation*len(valid))]
        X_val=[x[0] for x in validation]
        y_val=[x[1] for x in validation]
        nbr_batch=0
        val_loss=0
        for i in range(0,len(validation),BATCH_SIZE):
            nbr_batch+=1
            batch_X=torch.Tensor(np.array(X_val[i:i+BATCH_SIZE]))
            batch_y=torch.Tensor(np.array(y_val[i:i+BATCH_SIZE]))
            batch_X=batch_X.view(-1,1,256,256)
            batch_y=batch_y.view(-1,1,256,256)
            batch_X,batch_y = batch_X.to(device),batch_y.to(device)
            with torch.no_grad():
                outputs=net_ae(batch_X)
                val_loss+=loss_function(outputs,batch_y)
        val_loss/=nbr_batch
            
            
        scheduler.step()
        print(f"Epoch:{epoch}.Loss: {loss}, Val_loss: {val_loss}")
        with open(log_path,"a") as f:
            f.write(f"model name = {model_name}, Epoch:{epoch},Loss: {loss}, Val_loss: {val_loss}\n")    
    return model_name

def create_loss_graph(model_name,log_path='log_ae.log'):
    """
    Parameters
    ----------
    model_name : str
        model_name is used to read the log associated with that model.

    log_path : str, optional
        path of the log file. The default is 'log_ae.log'.

    Returns
    -------
    None. Provide a plot.

    """
    contents = open(log_path, "r").read().split("\n")

    times = []
    losses = []
    val_losses = []

    for c in contents:
        if model_name in c:
            model_name,timestamp, loss, val_loss = c.split(",")

            times.append(float(timestamp[7:]))

            losses.append(float(loss[6:]))


            val_losses.append(float(val_loss[11:]))


    fig = plt.figure()

    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.plot(times,losses, label="loss")
    ax1.plot(times,val_losses, label="val_loss")
    ax1.legend(loc=2)
    plt.title(label=model_name)
    plt.show()
    plt.pause(0.0001)
    
#use gpu if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the gpu")
else:
    device = torch.device("cpu")
    print("running on the cpu")
    
    
    
    
# =============================================================================
# # load data (256 ref to image being 256x256)
# train = np.load("cat_train_256_no_noise.pkl", allow_pickle=True)
# valid = np.load("cat_valid_256_no_noise.pkl", allow_pickle=True)
# test = np.load("cat_test_256_no_noise.pkl", allow_pickle=True)
# =============================================================================


# =============================================================================
# # usefull to liberate gpu memory when trying out a lot of model
# torch.cuda.empty_cache()
# dump_tensors()
# =============================================================================


# =============================================================================
# =============================================================================
# # training model
# =============================================================================
# net_ae=neural_AE()
# net_ae.to(device)
# model_name=train_valid(net_ae,train,valid,device,EPOCHS=600,LR=0.004,BATCH_SIZE=32,decay=1e-6,step_gamma=10,GAMMA=0.95)
# create_loss_graph(model_name=model_name)

# =============================================================================
# # save model
# torch.save(net_ae.state_dict(), "net_ae_state_dict.model")
# =============================================================================

# =============================================================================
# # test neural network performance
# =============================================================================
# =============================================================================
# # test on cat data set
# loss_function=nn.MSELoss()
# # loss_function=nn.L1Loss()
# # test_cat
# BATCH_SIZE=32
# test_catX=[x[0] for x in test]
# test_caty=[x[1] for x in test]
# net_ae.eval()
# test_loss=0
# nbr_batch=0
# for i in range(0,len(test_catX),BATCH_SIZE):
#     nbr_batch+=1
#     batch_X=torch.Tensor(np.array(test_catX[i:i+BATCH_SIZE]))
#     batch_y=torch.Tensor(np.array(test_caty[i:i+BATCH_SIZE]))
#     batch_X=batch_X.view(-1,1,256,256)
#     batch_y=batch_y.view(-1,1,256,256)
#     batch_X,batch_y = batch_X.to(device),batch_y.to(device)
#     with torch.no_grad():
#         outputs=net_ae(batch_X)
#         test_loss+=loss_function(outputs,batch_y)
# test_loss/=nbr_batch
# print(f"cat loss {round(test_loss.item(),5)}")


# =============================================================================
# # test on dog dataset (shouldn't be pertinent)
# # test_dog
# test_dog = np.load("dog_test_256_no_noise.pkl", allow_pickle=True)
# test_dogX=[x[0] for x in test_dog]
# test_dogy=[x[1] for x in test_dog]
# net_ae.eval()
# test_loss_dog=0
# nbr_batch=0
# for i in range(0,len(test_dogX),BATCH_SIZE):
#     nbr_batch+=1
#     batch_X=torch.Tensor(np.array(test_dogX[i:i+BATCH_SIZE]))
#     batch_y=torch.Tensor(np.array(test_dogy[i:i+BATCH_SIZE]))
#     batch_X=batch_X.view(-1,1,256,256)
#     batch_y=batch_y.view(-1,1,256,256)
#     batch_X,batch_y = batch_X.to(device),batch_y.to(device)
#     with torch.no_grad():
#         outputs=net_ae(batch_X)
#         test_loss_dog+=loss_function(outputs,batch_y)
# test_loss_dog/=nbr_batch
# print(f"dog loss {round(test_loss_dog.item(),5)}")
# =============================================================================

def find_loss(model,data,loss_function):
    """
    Parameters
    ----------
    model :
        autoencoder.
    data : 
        data on the format the model has been trained on.
    loss_function :
        loss_function used to calculate the reconstruction error

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
            outputs=model(batch_X)
            loss=loss_function(outputs,batch_y)
            rez.append(loss.item())
            test_loss+=loss
    test_loss/=lenn
    print(f"cat loss {round(test_loss.item(),5)}")
    return rez


# =============================================================================
# # read losses, give first insight on which images might be anomalies
# loss_function=nn.MSELoss()
# name=train #not efficient
# rez=find_loss(net_ae,name,loss_function=loss_function)
# rez=pd.DataFrame(rez)
# rez=rez.sort_values(0,ascending=False)
# for x in rez.index[:6]:
#     plt.figure()
#     plt.imshow(name[x][0])
#     plt.show()
#     print(name[x][2])
# =============================================================================
    

# =============================================================================
# =============================================================================
# # check if model is properly loaded
# =============================================================================
# torch.save(net_ae.state_dict(), "net_ae_state_dict.model")
# model = neural_AE()
# model.load_state_dict(torch.load("net_ae_state_dict.model"))
# model.to(device)
# model.eval()
# BATCH_SIZE=32
# test_catX=[x[0] for x in test]
# test_caty=[x[1] for x in test]
# test_loss=0
# nbr_batch=0
# for i in range(0,len(test_catX),BATCH_SIZE):
#     nbr_batch+=1
#     batch_X=torch.Tensor(np.array(test_catX[i:i+BATCH_SIZE]))
#     batch_y=torch.Tensor(np.array(test_caty[i:i+BATCH_SIZE]))
#     batch_X=batch_X.view(-1,1,256,256)
#     batch_y=batch_y.view(-1,1,256,256)
#     batch_X,batch_y = batch_X.to(device),batch_y.to(device)
#     with torch.no_grad():
#         outputs=model(batch_X)
#         test_loss+=loss_function(outputs,batch_y)
# test_loss/=nbr_batch
# =============================================================================
# # save architecture
# =============================================================================
# =============================================================================
# architecture=model._modules
# with open("net_ae_architecture.pkl", 'wb') as f:
#     pickle.dump(architecture, f, pickle.HIGHEST_PROTOCOL)
# model_architecture=pickle.loads("net_ae_architecture.pkl")
# with open("net_ae_architecture.pkl" , 'rb') as f:
#     model_architecture = pickle.load(f)
# =============================================================================

if __name__ == '__main__':
    import skimage
    def add_noise_for_training(train,valid,val=0.01):
        train=[ [skimage.util.random_noise(x[0],var=val),x[1] ] for x in train]
        valid=[ [skimage.util.random_noise(x[0],var=val),x[1] ] for x in valid]
        return train,valid
    
    model = neural_AE()
    # model.load_state_dict(torch.load("net_ae_state_dict.model"))
    
    
    train = np.load("cat_train_256_no_noise.pkl", allow_pickle=True)
    valid = np.load("cat_valid_256_no_noise.pkl", allow_pickle=True)
    # train=np.load("train_cat_noise_14.pkl",allow_pickle=True) #restart with the i-th time noised image
    # valid=np.load("valid_cat_noise_14.pkl",allow_pickle=True) #restart with the i-th time noised image
    model.to(device)
    model_name=train_valid(model,train,valid,device,EPOCHS=400,LR=0.004,BATCH_SIZE=32,decay=1e-6,step_gamma=10,GAMMA=0.89,model_name="model_ae",amsgrad=True)
    torch.save(model.state_dict(), f"{model_name}.model")
    torch.cuda.empty_cache()
    
    for i in range(1,10):
        print(f"training numero {i}")
        model.to(device)
        train,valid=add_noise_for_training(train,valid,val=0.01)
        model_name=train_valid(model,train,valid,device,EPOCHS=200,LR=0.001,BATCH_SIZE=32,decay=1e-6,step_gamma=7,GAMMA=0.89,model_name=f"retrain_{i}",amsgrad=True)
        # create_loss_graph(model_name=model_name)
        
        torch.save(model.state_dict(), f"model_{model_name}.model")
        torch.cuda.empty_cache()
    
    # train = np.load("cat_train_256_no_noise.pkl", allow_pickle=True)
    # valid = np.load("cat_valid_256_no_noise.pkl", allow_pickle=True)

    # num_img=0
    # plt.figure()
    # plt.imshow(train[num_img][0])
    # plt.show()
    # plt.figure()
    # plt.imshow(train[num_img][1])
    # plt.show()
    # img=train[0][1]
    # img=torch.Tensor(np.array(img))
    # img=img.view(-1,1,256,256)
    # np.shape(img)
    # img.to(device)
    # img=img.to(device)
    # reconstruct=model(img)
    # reconstruct=np.squeeze(reconstruct)
    # plt.figure()
    # plt.imshow(reconstruct.detach().cpu())
    # plt.show()
    
    # rez=find_loss(model,train)
    # rez=pd.DataFrame(rez)
    # rez=rez.sort_values(0,ascending=False) #show the images with biggest error reconstruction
    # for x in rez.index[:6]:
    #     plt.figure()
    #     plt.imshow(train[x][0])
    #     plt.show()
    #     plt.figure()
    #     plt.imshow(train[x][1])
    #     plt.show()
    #     # print(train[x][2])
        
    # with open("valid_cat_noise_14.pkl","wb") as f: #change number associated 
    #                                                 #with number of time image were noised
    #     pickle.dump(valid,f)
