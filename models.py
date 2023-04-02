# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:39:38 2023

@author: Jayus
"""
import torch
from torch import nn
import modules as m
import numpy as np

class base_model(nn.Module):
    def __init__(self,user_num,item_num,cate_num,hidden_size):
        super(base_model,self).__init__()
        self.u_emb = nn.Embedding(user_num, hidden_dim)
        self.i_emb = nn.Embedding(item_num, hidden_dim)
        self.c_emb = nn.Embedding(cate_num, hidden_dim)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_dim * 3, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
    def forward(self,user,hist,item,cate):
        user = self.u_emb(user).squeeze()
        item = self.i_emb(item).squeeze()
        cate = self.c_emb(cate).squeeze()
        h = []
        for i in range(len(hist)):
            h.append(self.i_emb(hist[i]).squeeze().detach().numpy())
        h = torch.tensor(np.array(h),dtype = torch.float32)
        cur = torch.zeros_like(h[0])
        for i in range(len(h)):
            cur += h[i]
        res = torch.cat([user,item,cate,cur],-1)
        res = self.linear(res)
        return res
    
    
class DIN(nn.Module):
    def __init__(self,user_num,item_num,cate_num,hidden_size):
        super(DIN,self).__init__()
        self.u_emb = nn.Embedding(user_num, hidden_size)
        self.i_emb = nn.Embedding(item_num, hidden_size)
        self.c_emb = nn.Embedding(cate_num, hidden_size)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_dim * 3, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
        self.au = m.ActivationUnit(hidden_size)
    def forward(self,user,hist,item,cate):
        user = self.u_emb(user).squeeze()
        item = self.i_emb(item).squeeze()
        cate = self.c_emb(cate).squeeze()
        h = []
        weights = []
        for i in range(len(hist)):
            hist_i = self.i_emb(hist[i])
            h.append(hist_i.squeeze().detach().numpy())
            weight = self.au(hist_i,item)
            weights.append(weight)
        cur = torch.zeros_like(h[0])
        for i in range(len(h)):
            cur += weights[i] * h[i]
        res = torch.cat([user,item,cate,cur],-1)
        res = self.linear(res)
        return res


class DIEN(nn.Module):
     def __init__(self,user_num,item_num,cate_num,hidden_size):
        super(DIEN,self).__init__()


class SIM(nn.Module):
    def __init__(self,user_num,item_num,cate_num,hidden_size):
        super(SIM,self).__init__()
    




        
        
        
