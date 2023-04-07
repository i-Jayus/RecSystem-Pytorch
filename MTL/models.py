# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:05:40 2023

@author: 12709
"""
import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self,expert_num,hidden_size,input_size):
        """
        MOE input parameters
        :param expert_num: int numbers of experts
        :param hidden_size: moe layer input dimension
        :param input_size: data embedding size
        """
        super(MoE,self).__init__()
        self.expert_num = expert_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for i in range(expert_num)])
        self.gate = nn.Linear(input_size,expert_num)
        self.fc = nn.Linear(hidden_size,2)
        self.softmax = nn.SoftMax()
        self.relu = nn.ReLU()
    
    def forward(self,x): # [user_embeddng, item_embedding]
        expert_outputs = []
        for i in range(self.expert_num):
            expert_outputs.append(self.relu(self.experts[i](x)))
        expert_output = torch.stack(expert_outputs)
        gate_output = self.softmax(self.gate(x), dim = 1)
        res = torch.zeros(expert_num)
        for i in range(self.expert_num):
            res += gate_output[i] * expert_output[i]
        res = self.fc(res)
        return res

class MMoE(nn.Module):
    def __init__(self,expert_num,task_num,hidden_size,input_size):
        """
        MMOE input parameters
        :param expert_num: int numbers of experts
        :param task_num: int numbers of tasks
        :param hidden_size: moe layer input dimension
        :param input_size: data embedding size
        """
        super(MMoE,self).__init__()
        self.expert_num = expert_num
        self.task_num = task_num
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for i in range(expert_num)])
        self.gates = nn.ModuleList([nn.Linear(input_size, expert_num) \
                                          for i in range(task_num)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 2) \
                                          for i in range(task_num)])
        self.relu = nn.ReLU()
        self.softmax = nn.SoftMax()
    
    def forward(self, x): # [user_embeddng, item_embedding]
        expert_outputs = []
        gate_outputs = []

        for i in range(self.expert_num):
            expert_outputs.append(self.relu(self.experts[i](x)))
        for i in range(self.task_num):
            gate_outputs.append(self.softmax(self.gates[i](x), dim=1))

        expert_output = torch.stack(expert_outputs)
        gate_output = torch.stack(gate_outputs)
        res = []
        for i in range(self.task_num):
            tmp = torch.zeros(expert_num)
            for j in range(self.expert_num):
                tmp += gate_output[i][j] * expert_output[j]
            res.append(tmp)
        res = torch.stack(res)
        out = []
        for i in range(self.task_num):
            out.append(self.fcs[i](res[i]))
        out = torch.stack(out)
        return out


            
    
    
    
    