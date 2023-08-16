import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as transf
from skimage.util import random_noise
import torch
from numpy.random import choice
import joblib
from skimage.transform import resize
from torchvision import models, transforms
import tqdm
from skimage.filters import unsharp_mask



class ADNI_Dataset(Dataset):
    
    def __init__(self, dataframe, ATTR, transform=True, phase="train"):
        self.phase=phase
        self.dataframe=dataframe
        self.transform = transform
        self.channel=96
        self.ATTR=ATTR
        
        
    def __len__(self):
        return len(self.dataframe)

        
    def __getitem__(self, idx):
        
        filepath = os.path.join("D:/Project/resized_data/", self.dataframe['Image Data ID'].iloc[idx]+".nii")
        image = self.load_img(filepath)

        image=self.normalize(image)
        image=self.augmentation(image, self.transform)
        image=np.expand_dims(image, axis=0)
        
                
        if(self.dataframe['Group'].iloc[idx]=="CN"):
            label = 0
        if(self.dataframe['Group'].iloc[idx]=="AD"):
            label = 1
            
        if self.ATTR == "gender":
            if(self.dataframe['Sex'].iloc[idx]=="M"):
                attr = 0
            if(self.dataframe['Sex'].iloc[idx]=="F"):
                attr = 1
                
        elif self.ATTR == "age":
            if(self.dataframe['Age'].iloc[idx]<=75):
                attr = 0
            elif(self.dataframe['Age'].iloc[idx]>75):
                attr = 1
            
                
        
        image=torch.from_numpy(image)
        label=torch.tensor(label)
        attr=torch.tensor(attr)
            
            
        return image, label, attr
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.array(data.dataobj)
        return data
    
    def normalize(self, arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    def augmentation(self, img, transform):
      
        shear_r=4
        rotate_r=6
        crop_r=20
        
        # =====================
        shear_x=random.randint(-1*shear_r,shear_r)
        shear_y=random.randint(-1*shear_r,shear_r)
        rotate_x=random.randint(-1*rotate_r,rotate_r)
        image_first_shear=[]
        for i in range(img.shape[2]):
            temp=img[:,:,i] 
            if(transform):
                temp = Image.fromarray(temp)
                temp = transf.rotate(temp, rotate_x, expand=False, center=None)
                temp = transf.affine(temp, shear=(shear_x,shear_y), translate=(0, 0), scale=1.0, angle=0)
                temp = np.array(temp)
            image_first_shear.append(temp)
        image_first_shear=np.array(image_first_shear)
        # =====================
        shear_x=random.randint(-1*shear_r,shear_r)
        shear_y=random.randint(-1*shear_r,shear_r)
        rotate_x=random.randint(-1*rotate_r,rotate_r)
        crop_x=140+random.randint(-1*crop_r,0)
        transform_crop = transforms.Compose([
            transforms.CenterCrop(crop_x)
        ])
        image_second_shear=[]
        
        for i in range(image_first_shear.shape[1]): 
            temp=image_first_shear[:,i,:] 
            
            if(transform):
                temp = Image.fromarray(temp)
                temp = transf.rotate(temp, rotate_x, expand=False, center=None)
                temp = transf.affine(temp, shear=(shear_x,shear_y), translate=(0, 0), scale=1.0, angle=0)
                temp = np.array(temp)
            image_second_shear.append(temp)
        image_second_shear=np.array(image_second_shear)

        # =====================
        shear_x=random.randint(-1*shear_r,shear_r)
        shear_y=random.randint(-1*shear_r,shear_r)
        rotate_x=random.randint(-1*rotate_r,rotate_r)
        image_third_shear=[]
        for i in range(image_second_shear.shape[2]): 
            temp=image_second_shear[:,:,i]
            if(transform):
                temp = Image.fromarray(temp)
                temp = transf.rotate(temp, rotate_x, expand=False, center=None) #rotate
                temp = transf.affine(temp, shear=(shear_x,shear_y), translate=(0, 0), scale=1.0, angle=0) #shear
                temp = np.array(temp)
                temp = torch.tensor(random_noise(temp, mode='gaussian', mean=0, var=0.0001, clip=True)) #gaussian
                temp=np.array(temp)
            image_third_shear.append(temp)
        image_third_shear=np.array(image_third_shear)
        # =====================
        
        ret=[]
        for i in range(self.channel):
            temp=image_third_shear[:,i,:]
            ret.append(temp)
        
        ret=np.array(ret) 
        
        return ret


