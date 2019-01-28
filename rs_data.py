# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 19:23:44 2018

@author: huijian
"""
# torch
import torch
from torch.utils.data import Dataset
# sklearn
from sklearn import preprocessing
# numpy
import numpy as np


def read_txt(file):
    data = np.loadtxt("rs_data.txt")
    return data

def pre_process(data,ratio=0.7):
    """
    data: (68639,161)
    161 = 32(output) + 32(output) + 32(output) + 1(output) + 32(output) + 32(input)
    col: 129-160 for features
    indexbands = [0,2,7,10,13,24,30]
    """
    idx = [0,2,7,10,13,24,30]
    idx = np.array(idx)
    
    idx_list = []
    
    idx_list.append(idx)
    idx_list.append(idx+32)
    idx_list.append(idx+64)
    idx_list.append(np.array([96]))
    idx_list.append(idx+97)
     
    input_idx = []
    for i in idx_list:
        input_idx = input_idx + i.tolist()
        
    output_idx = [i for i in range(129,161)]  
    
    total_idx = input_idx + output_idx
    total_idx = np.array(total_idx)
    required_data = data[:,total_idx]
    
    idx = required_data.shape[0]
    idx = np.array([i for i in range(idx)])
    
    # shuffle the indexes    
    np.random.shuffle(idx)
    required_data = required_data[idx,:]
    
    # divide the data
    truncated = int(required_data.shape[0] * ratio)
    train_data = required_data[0:truncated,:]
    test_data = required_data[truncated:,:]
    
    return train_data,test_data,required_data,input_idx, output_idx

# torch dataset
class RS_Data(Dataset):
    def __init__(self,data,transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        
        sample = {}
        tmp_sample = self.data[index,:] # shape is (61,)
        
        sample["input"] = tmp_sample[29:]
        sample["output"] = tmp_sample[0:29]
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

if __name__ == "__main__":
    
    # get the data
    file = "rs_data.dat"
    original_data = read_txt(file)
    
    # normalization ( with scikit-learn) if exist
    # the range is normalized to [0,1] to fit sigmoid layer()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    # and the max_value is stored in min_max_scaler.data_max_
    processed_data = min_max_scaler.fit_transform(original_data)
    # division 
    train_data,test_data,required_data,input_idx,output_idx = pre_process(processed_data)
    
    t_data = RS_Data(data=train_data)
    print(t_data['input'])