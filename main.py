# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:40:37 2018

@author: huijian
"""

import torch
# mylibs
from trainer import Trainer
from rs_data import RS_Data,read_txt,pre_process
import torch.optim as optim
from torch.utils.data import DataLoader
from archs.ann_net import ANN
from math import *
# sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

def draw_fig(sample_pred, sample_output, sample_delta,i=21):
    fig, ax = plt.subplots()
    ax.scatter(sample_output[:,i],sample_delta[:,i])
    plt.show()
    # ax.scatter(sample_output[:,i], sample_pred[:,i])
    # plt.show()

if __name__=="__main__":
    
    # prepare data
    # get the data
    file = "rs_data.dat"
    original_data = read_txt(file)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    processed_data = min_max_scaler.fit_transform(original_data)
    train_data,test_data,required_data,input_idx,output_idx = pre_process(processed_data)
    output_scale = min_max_scaler.scale_[input_idx]
    output_min = min_max_scaler.data_min_[input_idx]
    
    train_data = RS_Data(data=train_data)
    test_data = RS_Data(data=test_data)
    
    # define the model
    model_path = "./model/"
    cuda = torch.cuda.is_available()
    
    input_dim = 32
    output_dim = 29
    


    _train = False

    # build the network
    if _train:
        net = ANN(input_dim,output_dim)
        trainer = Trainer(net=net,model_path=model_path,cuda=cuda)
    else:
        model_name = "model2"
        trainer = Trainer(net=None,model_path=model_path,cuda=cuda)
        trainer.restore_model(model_name)
    
    # train the model
    train_data= train_data
    val_data =test_data
    train_bs = 100
    val_bs = 1
    epochs = 2000
    if _train:
        trainer.train_model(train_data,val_data,train_bs,val_bs,epochs)
    else:
        sample_pred,sample_output,sample_delta = trainer.predict(test_data, output_scale, output_min)
        draw_fig(sample_pred, sample_output, sample_delta, 21)
