import nibabel as nib
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

def transfer_to_df(dataframe):
    image_id=[]
    Group=[]
    Sex=[]
    Age=[]
    missing_data = []
    for i in range(len(dataframe)):
        s = dataframe['Links'].iloc[i]
        img_id = s.split('\\')[-3]
        
        filepath = os.path.join("D:/Project/resized_data/", img_id+".nii")
        if(os.path.exists(filepath)):
            image_id.append(img_id)
            Group.append(dataframe['Group'].iloc[i])
            Sex.append(dataframe['Sex'].iloc[i])
            Age.append(dataframe['Age'].iloc[i])
        else:
            missing_data.append(img_id)
    dic = {"Image Data ID": image_id, "Group": Group, "Sex": Sex, "Age": Age}
    new_dataframe = pd.DataFrame(dic)
    return new_dataframe