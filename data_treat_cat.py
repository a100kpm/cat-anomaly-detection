# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from sklearn import model_selection
import skimage
from matplotlib import image

# =============================================================================
# =============================================================================
# =============================================================================
# # # data found here: https://www.kaggle.com/competitions/dogs-vs-cats/data
# # # then dogs and cats have been splitted based on name in PetImages\Cat or PetImages\Dog
# # # modify the path below to fit your needs
# # # modify resize of image (img_size) to fit your needs
# # # uncomment lines 48,88 and either 142-148 or 111-117 in order to create
# # # the training, validation and test set
# =============================================================================
# =============================================================================
# =============================================================================

project_base_image_path=r"C:\Users\*******************************\PetImages" 
cat_path=project_base_image_path+"\Cat"
dog_path=project_base_image_path+"\Dog"
# labels={"cat":cat_path,"dog":dog_path}
labels={"cat":cat_path}
img_size=256

            
def resize_dataset(labels,img_size):
    """
    Parameters
    ----------
    labels : str
        cat or dog.
    img_size : TYPE
        size of the resize of the images

    Returns
    -------
    Save as pickle in folder

    """
    for key,value in labels.items():
        data=[]
        for file in tqdm(os.listdir(value)):
            try:
                path=os.path.join(value,file)
                img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(img_size,img_size))
                data.append([img,file])
            except Exception as e:
                pass
                #print(str(e))
        data=[ [ x[0]/255.0,x[1] ] for x in data ]
        with open(key+"_data.pkl", "wb") as f:
                pickle.dump(data, f)
        print(f"saved {len(data)} resized {key} data")

# resize_dataset(labels,img_size)


# file_list=["cat_data.pkl","dog_data.pkl"]
file_list=["cat_data.pkl"]
def get_training_data(file_list):
    """
    Parameters
    ----------
    file_list : list
        list of path.

    Returns
    -------
    return_data : list
        return all data from path(s)

    """
    return_data=[]
    for file in file_list:
        data = np.load(file, allow_pickle=True)
        data=[x for x in data]
        return_data.extend(data)
    np.random.shuffle(return_data)
    return return_data

# data=get_training_data(file_list)


def split_and_noise(data,ratio=0.6,test_set=0.5,number_noise_image=3,test_noise=False):
    """
    Parameters
    ----------
    data : list
        list of all images
    ratio : float, optional
        ratio of training data over all data. The default is 0.6.
    test_set : float, optional
        ratio of test data over test + validation data. The default is 0.5.
    number_noise_image : int, optional
        number of noisy image to create. The default is 3.
    test_noise : bool, optional
        If True create noise images for the test set. The default is False.

    Returns
    -------
    new_train : list
        training data set.
    new_valid : list
        validation data set.
    new_test : list
        test data set.

    """
    train,valid=model_selection.train_test_split(data,train_size=ratio)
    test,valid=model_selection.train_test_split(valid,train_size=test_set)
    new_train=[]
    for x in train:
        new_train.append([x[0],x[0],x[1]])
        for i in range(number_noise_image):
            noise_im=skimage.util.random_noise(x[0],var=0.001)
            new_train.append([noise_im,x[0],x[1]])
    new_valid=[]
    for x in valid:
        new_valid.append([x[0],x[0],x[1]])
        for i in range(number_noise_image):
            noise_im=skimage.util.random_noise(x[0],var=0.001)
            new_valid.append([noise_im,x[0],x[1]])
    new_test=[]
    if test_noise:
        for x in test:
            new_test.append([x[0],x[0],x[1]])
            for i in range(number_noise_image):
                noise_im=skimage.util.random_noise(x[0],var=0.001)
                new_test.append([noise_im,x[0],x[1]])
    else:
        new_test=[[x[0],x[0],x[1]] for x in test]
        
    return new_train,new_valid,new_test
            
# train,valid,test=split_and_noise(data)
# with open(f"cat_{img_size}_train.pkl","wb") as f:
#     pickle.dump(train,f)
# with open(f"cat_{img_size}_valid.pkl","wb") as f:
#     pickle.dump(valid,f)
# with open(f"cat_{img_size}_test.pkl","wb") as f:
#     pickle.dump(test,f)
    

def split_no_noise(data,ratio=0.6,test_set=0.5):
    """
    Parameters
    ----------
    data : list
        list of all images
    ratio : float, optional
        ratio of training data over all data. The default is 0.6.
    test_set : float, optional
        ratio of test data over test + validation data. The default is 0.5.

    Returns
    -------
    train : list
        training data set.
    valid : list
        validation data set.
    test : list
        test data set.

    """
    train,valid=model_selection.train_test_split(data,train_size=ratio)
    test,valid=model_selection.train_test_split(valid,train_size=test_set)
    train=[[x[0],x[0],x[1]] for x in train]
    valid=[[x[0],x[0],x[1]] for x in valid]
    test=[[x[0],x[0],x[1]] for x in test]
        
    return train,valid,test
            
    
# train,valid,test=split_no_noise(data)
# with open(f"cat_train_{img_size}_no_noise.pkl","wb") as f:
#     pickle.dump(train,f)
# with open(f"cat_valid_{img_size}_no_noise.pkl","wb") as f:
#     pickle.dump(valid,f)
# with open(f"cat_test_{img_size}_no_noise.pkl","wb") as f:
#     pickle.dump(test,f)
