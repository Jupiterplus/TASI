# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:04:05 2018

@author: huijian
"""
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self,input_dim =32, output_dim=29):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,output_dim),
                nn.Sigmoid(),
                )
    def forward(self,x):
        x = self.net(x)
        return x
    
if __name__ == "__main__":
    net = ANN(input_dim=32,output_dim=29)
    random_input = torch.rand((10,32)) # bs = 10
    print(random_input)
    random_output = torch.rand((1,29)) 
    criterion = nn.MSELoss()
    output = net(random_input)