# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:40:11 2023

@author: Jayus
"""
import torch 
from torch import nn
import tools as t

class Dice(nn.Module):
    def __init__(self,size,eps,dim=2):
        super(Dice,self).__init__()
        self.name = 'dice'
        t.dimJudge(dim,2)
        self.bn = nn.BatchNorm1d(size,eps=eps)
        self.sig = nn.Sigmoid()
        self.alpha = torch.zeros((size,))
        self.beta = torch.zeros((size,))
    
    def forward(self,x):
        x_n = self.sig(self.beta * self.bn(x))
        return self.alpha * (1-x_n) * x + x_n * x

class Attention(nn.Module):
    def __init__(self,inSize,outSize):
        super(Attention,self).__init__()
        self.name = 'attention'
        self.wq = nn.Linear(inSize,outSize)
        self.wk = nn.Linear(inSize,outSize)
        self.wv = nn.Linear(inSize,outSize)
        self.sig = nn.Sigmoid()
        
    def forward(self,q,x):
        Q = q * self.wq
        K = x * self.wk
        V = x * self.wv
        X = Q * K.T
        attentionScore = self.sig(X/torch.sqrt(X.dim))
        return attentionScore * V

if __name__ == '__main__':
    dice = Dice(2,1e-8)
    x = torch.tensor([[1,2],[3,4],[5,6],[7,8]],dtype = torch.float32)
    print(dice(x))


    
        
        
        