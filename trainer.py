# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:37:09 2018

@author: huijian
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from math import *
import numpy as np

class Trainer(object):
    def __init__(self,net,model_path="../model/",cuda=False):
        self.net = net
        self.model_path = model_path
        
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = None
    
    def __sample(self,sample):
        sample_input = sample["input"].to(torch.float32)
        sample_output = sample["output"].to(torch.float32)
        if self.cuda:
            sample_input = sample_input.to(self.device)
            sample_output = sample_output.to(self.device)
        return sample_input,sample_output
    
    def save_model(self,model_name = "net"):
        if self.cuda:
            self.net = self.net.cpu()
        torch.save(self.net,self.model_path+model_name+".pkl")
        if self.cuda:
            self.net = self.net.to(self.device)
        print("Model saved!")
    
    def restore_model(self,model_name = "net"):
        # if self.cuda:
        #     self.net = self.net.cpu()
        self.net = torch.load(self.model_path+model_name+".pkl")
        if self.cuda:
            self.net = self.net.to(self.device)
        print("Model restored!")
    
    def train_step(self,sample,iters,epoch):
        
        # clear the optimizer
        self.optimizer.zero_grad()
        
        sample_input, sample_output = self.__sample(sample)
        print(sample_input.size())
        sample_pred = self.net(sample_input)
        
        loss = self.criterion(input=sample_pred,target=sample_output)
        loss.backward()
        
        self.optimizer.step()
        if (iters+1)%10==0:
            if self.cuda:
                loss = loss.cpu()
            loss = float(loss.detach().numpy())
            print("(Train)Epoch:{}/Iters{} - Loss(mse):{:.5}".format(
                    epoch+1,iters+1,sqrt(loss)))
        
    def validate(self,data_loader):
        total_loss = 0
        total_bs = 0
        sample_delta = np.array([])
        for iters,sample in enumerate(data_loader):
            sample_input,sample_output = self.__sample(sample)
            
            with torch.no_grad():
                sample_pred = self.net(sample_input)
            
            tmp_loss = self.criterion(input=sample_pred,target=sample_output)
            if self.cuda:
                tmp_loss = tmp_loss.cpu()

            sample_delta = np.append(sample_delta,sample_pred - sample_output,axis=0)
            total_loss = total_loss + float(tmp_loss.detach().numpy())
            total_bs = total_bs + sample_input.size(0)
            del tmp_loss
        loss = total_loss/total_bs
        delta = np.sqrt(np.sum(sample_delta*sample_delta, axis=0))
        print("(Validation) - Loss(mse):{:.5}".format(sqrt(loss)))
        print("(Validation) - RMSE(per): {}".format(delta))
        return 
    
    def train_model(self,train_data,val_data,train_bs=16,val_bs=1,epochs=10000):
        
        if self.cuda:
            self.net = self.net.to(self.device)
        
        # define the criterion
        self.criterion = torch.nn.MSELoss()
        # define the dataset
        self.train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)
        self.val_loader = DataLoader(dataset=val_data,batch_size=val_bs,shuffle=False)
        # define the optimizer
        self.optimizer = optim.Adam(params=self.net.parameters(),lr=5e-5,betas=(0.9,0.99))
        
        # training process 
        self.net.train()
        for e in range(epochs):
            for iters,sample in enumerate(self.train_loader):
                self.train_step(sample=sample,iters=iters,epoch=e)
            if (e+1)%1==0:
                self.net.eval()
                # self.validate(data_loader=self.val_loader)
                model_name = "net"+"_epoch_" + str(e)
                self.save_model(model_name)
                self.net.train()
                
    def predict(self,sample_input,output_scale,output_min):
        data_loader = DataLoader(dataset=sample_input,batch_size=len(sample_input),shuffle=False)
        total_loss = 0
        total_bs = 0
        for iters,sample in enumerate(data_loader):
            total_loss+=1
            sample_input,sample_output = self.__sample(sample)
            
            with torch.no_grad():
                sample_pred = self.net(sample_input)
            sample_delta = sample_pred - sample_output
            total_bs = total_bs + sample_input.size(0)
        if self.cuda:
            sample_delta = sample_delta.cpu()
            sample_pred = sample_pred.cpu()
            sample_output = sample_output.cpu()
        sample_delta = np.divide(sample_delta.numpy(),output_scale)
        sample_pred = np.divide(sample_pred.numpy(),output_scale)+output_min
        sample_output = np.divide(sample_output.numpy(),output_scale)+output_min
        delta = np.sqrt(np.sum(sample_delta*sample_delta, axis=0)/total_bs)
        print("(predict) - RMSE: {}".format(delta))
        return sample_pred,sample_output,sample_delta
        
        
        
        