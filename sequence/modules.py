# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:40:11 2023

@author: Jayus
"""
import torch 
from torch import nn
import tools as t
import random 
import numpy as np
import copy

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
        
    def forward(self,item1,item2): #[B,C]
        cross = torch.mm(item1,item2.T)
        x = torch.cat([item1,cross,item2],-1) #[B,B+2*C]
        x = self.linear1(x)
        x = self.af(x)
        x = self.linear2(x)
        return x
    

def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]

def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []

    if length == 1:
        return 0

    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    avg_diff = total / len(diffs)

    total = 0
    for diff in diffs:
        total = total + (diff - avg_diff) ** 2
    result = total / len(diffs)

    return result

class Insert(object):
    """
    Insert similar items every time call.
    Priority is given to places with large time intervals.
    maximum: Insert at larger time intervals
    minimum: Insert at smaller time intervals
    """

    def __init__(self, item_similarity_model, mode, insert_rate=0.4, max_insert_num_per_pos=1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.mode = mode
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        insert_idx = []
        for i in range(insert_nums):
            temp = diff_sorted[i]
            insert_idx.append(temp)

        """
        The index of time_diff is 1 smaller than the item. 
        The item should be inserted to the right of item_index. 
        Put the original item first in each cycle, so that the inserted item is inserted to the right of the original item
        """
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):

            inserted_sequence += [item]

            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item, top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item, top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item, top_k=top_k)

        return inserted_sequence


class Substitute(object):
    """
    Substitute with similar items
    maximum: Substitute items with larger time interval
    minimum: Substitute items with smaller time interval
    """

    def __init__(self, item_similarity_model, mode, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        if len(copied_sequence) <= 1:
            return copied_sequence
        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        substitute_idx = []
        for i in range(substitute_nums):
            temp = diff_sorted[i]
            substitute_idx.append(temp)

        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
                copied_sequence[index] = copied_sequence[index] = \
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence


class Crop(object):
    """
    maximum: Crop subsequences with the maximum time interval variance
    minimum: Crop subsequences with the minimum time interval variance
    """

    def __init__(self, mode, tao=0.2):
        self.tao = tao
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length <= 2:
            return [copied_sequence[start_index]]

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        cropped_sequence = copied_sequence[start_index:start_index + sub_seq_length]
        return cropped_sequence


class Mask(object):
    """
    Randomly mask k items given a sequence
    maximum: Mask items with larger time interval
    minimum: Mask items with smaller time interval
    """

    def __init__(self, mode, gamma=0.7):
        self.gamma = gamma
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]

        if len(copied_sequence) <= 1:
            return copied_sequence

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum', 'random']
        if self.mode == 'random':
            copied_sequence = copy.deepcopy(item_sequence)
            mask_nums = int(self.gamma * len(copied_sequence))
            mask = [0 for i in range(mask_nums)]
            mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
            for idx, mask_value in zip(mask_idx, mask):
                copied_sequence[idx] = mask_value
            return copied_sequence
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        mask_idx = []
        for i in range(mask_nums):
            temp = diff_sorted[i]
            mask_idx.append(temp)

        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """
    Randomly shuffle a continuous sub-sequence
    maximum: Reorder subsequences with the maximum time interval variance
    minimum: Reorder subsequences with the minimum variance of time interval
    """

    def __init__(self, mode, beta=0.2):
        self.beta = beta
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        if sub_seq_length < 2:
            return copied_sequence

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


if __name__ == '__main__':
    dice = Dice(4,dim=2)
    au = ActivationUnit(10)
    x = torch.tensor([[1,2,3,4],[5,6,1,2]],dtype = torch.float32)
    y = dice(x)
    print(y)
    print(au(x,y))


    
        
        
        