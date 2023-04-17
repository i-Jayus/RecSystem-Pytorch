# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:39:38 2023

@author: Jayus
"""
import torch
from torch import nn
import modules as m
import numpy as np

class STAMP(nn.Module):
    def __init__(self,user_num,item_num,cate_num,hidden_size=64):
        super(STAMP, self).__init__()
        

class base_model(nn.Module):
    def __init__(self,user_num,item_num,cate_num,hidden_size=64):
        """
        base model input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cate_num: int numbers of categories
        :param hidden_size: embedding_size
        """
        super(base_model, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.u_emb = nn.Embedding(user_num, hidden_size)
        self.i_emb = nn.Embedding(item_num, hidden_size)
        self.c_emb = nn.Embedding(cate_num, hidden_size)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_size * 4, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
        
    def forward(self,user,hist,item,cate):
        """
        :param user: user id
        :param hist: list of history behaviors of user
        :param item: item id
        :param cate: category id of item
        """
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
    def __init__(self,user_num,item_num,cate_num,hidden_size=64):
        """
        DIN input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cate_num: int numbers of categories
        :param hidden_size: embedding_size
        """
        super(DIN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.u_emb = nn.Embedding(user_num, hidden_size)
        self.i_emb = nn.Embedding(item_num, hidden_size)
        self.c_emb = nn.Embedding(cate_num, hidden_size)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_size * 4, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
        self.au = m.ActivationUnit(hidden_size)
        
    def forward(self,user,hist,item,cate):
        """
        :param user: user id
        :param hist: list of history behaviors of user
        :param item: item id
        :param cate: category id of item
        """
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
            cur += torch.tensor(weights[i] * h[i], dtype=torch.float32)
            
        res = torch.cat([user,item,cate,cur],-1)
        res = self.linear(res)
        return res

'''
coming soon------------------


class DIEN(nn.Module):
    def __init__(self,user_num,item_num,cate_num,embedding_dim=32,hidden_dim=64):
        """
        DIEN input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cate_num: int numbers of categories
        :param embedding_dim: embedding size
        :param hidden_dim: input dim for interest extractor
        """
        super(DIEN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.user_embedding = nn.Embedding(user_num, embedding_dim)
        self.item_embedding = nn.Embedding(item_num, embedding_dim)
        self.cate_embedding = nn.Embedding(cate_num, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            m.Dice(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, uid, iid_his, cid_his):
        uid_emb = self.user_embedding(uid)
        iid_his_emb = self.item_embedding(iid_his)
        cid_his_emb = self.cate_embedding(cid_his)

        his_emb = torch.cat([iid_his_emb, cid_his_emb], dim=-1)
        his_emb = his_emb.permute(1, 0, 2)

        _, hidden = self.gru(his_emb)

        x = torch.cat([uid_emb.squeeze(0), his_emb[-1]], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x
'''

class SIM(nn.Module):
    def __init__(self,user_num,item_num,cate_num,time_span,hidden_size=64,mode='hard',thre=0.8):
        """
        SIM input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cate_num: int numbers of categories
        :param time_span: time stamps
        :param hidden_size: embedding_size
        :param mode: sequence cutting strategy
        :param thre: threshold for soft strategy for sequence cutting
        """
        super(SIM,self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.time_span = time_span
        self.user_embedding = nn.Embedding(user_num,hidden_size)
        self.item_embedding = nn.Embedding(item_num,hidden_size)
        self.cate_embedding = nn.Embedding(cate_num,hidden_size)
        self.time_embedding = nn.Embedding(time_span,hidden_size)
        self.mode = mode
        self.thre = thre
        self.linear =  nn.Sequential(
            nn.Linear(hidden_size * 6, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
        self.au = m.ActivationUnit(hidden_size * 2)
        
    def forward(self,user,hist,item,cate,time):  #hist: [item,cate,time]
        """
        :param user: user id
        :param hist: list of history behaviors of user
        :param item: item id
        :param cate: category id of item
        :param time: current time stamp
        """
        user = self.user_embedding(user).squeeze()
        item = self.item_embedding(item).squeeze()
        cate = self.cate_embedding(cate).squeeze()
        time = self.time_embedding(time).squeeze()
        item = torch.cat([item,time],-1)
        h = []
        for i in range(len(hist)):
            cate_i = self.cate_embedding(hist[i][1]).squeeze()
            if mode == 'hard' and cate_i == cate:    
                hist_i = self.item_embedding(hist[i][0])
                time_i = self.time_embedding(hist[i][2])
                h.append(torch.cat([hist_i.squeeze().detach().numpy(),\
                                    time_i.squeeze().detach().numpy()],-1))
            elif mode == 'soft':
                hist_i = self.item_embedding(hist[i][0])
                time_i = self.time_embedding(hist[i][2])
                h_i = torch.cat([hist_i.squeeze().detach().numpy(),\
                                    time_i.squeeze().detach().numpy()],-1)
                sim = torch.cosine_similarity(item, h_i, dim=0)
                if sim >= self.thre:
                    h.append(h_i)            
            else: 
                print('you can just choose "soft" or "hard" mode for SIM')
                return
            
        h = torch.tensor(np.array(h),dtype = torch.float32)
        weights = []
        for i in range(len(h)):
            weight = self.au(h[i],item)
            weights.append(weight)
            
        cur = torch.zeros_like(h[0])
        for i in range(len(h)):
            cur += torch.tensor(weights[i] * h[i], dtype=torch.float32)
            
        res = torch.cat([user,item,cate,cur],-1)
        res = self.linear(res)
        return res
        



        
            
        
        




        
        
        
