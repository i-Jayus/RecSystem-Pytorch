# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:40:11 2023

@author: Jayus
"""
import torch 
from torch import nn
import tools as t

class PRelu(nn.Module):
    def __init__(self,size):
        super(PRelu,self).__init__()
        self.name = 'Prelu'
        self.alpha = torch.zeros((size,))
        self.relu = nn.Relu()
        
    def forward(self,x):
        pos = self.relu(x) #only for positive part
        neg = self.alpha * (x - abs(x)) * 0.5 #only for negetive part
        return pos + neg


class Dice(nn.Module):
    def __init__(self,emb_size,eps=1e-8,dim=3):
        super(Dice,self).__init__()
        self.name = 'dice'
        self.dim = dim
        t.dimJudge(dim,2,3) 
        self.bn = nn.BatchNorm1d(emb_size,eps=eps)
        self.sig = nn.Sigmoid()
        if dim == 2:   #[B,C]
            self.alpha = torch.zeros((emb_size,))
            self.beta = torch.zeros((emb_size,))
        elif dim == 3: #[B,C,E]
            self.alpha = torch.zeros((emb_size,1))
            self.beta = torch.zeros((emb_size,1))
    
    def forward(self,x):
        if self.dim == 2:
            x_n = self.sig(self.beta * self.bn(x))
            return self.alpha * (1-x_n) * x + x_n * x
        elif self.dim == 3:
            x = torch.transpose(x,1,2)
            x_n = self.sig(self.beta * self.bn(x))
            output = self.alpha * (1-x_n) * x + x_n * x
            output = torch.transpose(output,1,2)
            return output
            

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

class ActivationUnit(nn.Module):
    def __init__(self,inSize,af='dice',hidden_size=36):
        super(ActivationUnit,self).__init__()
        self.name = 'activation_unit'
        self.linear1 = nn.Linear(inSize,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
        if af == 'dice':
            self.af = Dice(hidden_size,dim=2)
        elif af == 'prelu':
            self.af = PRelu()
        else:
            print('only dice and prelu can be chosen for activation function')
        
    def forward(self,user,item): #[B,C]
        cross = torch.mm(user,item.T)
        x = torch.cat([user,cross,item],-1) #[B,B+2*C]
        x = self.linear1(x)
        x = self.af(x)
        x = self.linear2(x)
        return x

if __name__ == '__main__':
    dice = Dice(4,dim=2)
    au = ActivationUnit(10)
    x = torch.tensor([[1,2,3,4],[5,6,1,2]],dtype = torch.float32)
    y = dice(x)
    print(y)
    print(au(x,y))


    
        
        
        