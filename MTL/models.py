# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:05:40 2023

@author: 12709
"""
import torch
import torch.nn as nn

class ESMM(nn.Module):
    def __init__(self,user_num,item_num,hidden_size_main,hidden_size_auxiliary,embedding_size):
        """
        ESMM input parameters
        :param user_num: number of users
        :param item_num: number of items
        :param hidden_size_main: hidden size in main network for cvr
        :param hidden_size_auxiliary: hidden size in auxiliary network for ctcvr
        :param embedding_size: embedding size
        """
        super(ESMM,self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.main = hidden_size_main
        self.aux = hidden_size_auxiliary
        self.embedding_size = embedding_size
        self.user_embedding = nn.Embedding(user_num, embedding_size)
        self.item_embedding = nn.Embedding(item_num, embedding_size)
        self.mlp_main = nn.Sequential(
            nn.Linear(embedding_size * 2, self.main),
            nn.ReLU(),
            nn.Linear(self.main, self.main),
            nn.ReLU(),
            nn.Linear(self.main, 1),
            nn.Sigmiod()
        )
        self.mlp_aux = nn.Sequential(
            nn.Linear(embedding_size * 2, self.aux),
            nn.ReLU(),
            nn.Linear(self.aux, self.aux),
            nn.ReLU(),
            nn.Linear(self.aux, 1),
            nn.Sigmiod()
        )
        
        def forward(self,user,item):
            user = self.user_embedding(user)
            item = self.item_embedding(item)
            vector = torch.cat([user,item],-1)
            cvr = self.mlp_main(vector)
            ctr = self.mlp_aux(vector)
            ctcvr = ctr * cvr
            return ctr, ctcvr #cvr = ctcvr/ctr
            
    
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
        self.fc = nn.Linear(hidden_size,1)
        self.softmax = nn.SoftMax()
        self.relu = nn.ReLU()
    
    def forward(self,x): # [user_embeddng, item_embedding]
        expert_outputs = []
        for i in range(self.expert_num):
            expert_outputs.append(self.relu(self.experts[i](x)))
            
        expert_output = torch.stack(expert_outputs)
        gate_output = self.softmax(self.gate(x), dim = 1)
        res = torch.zeros(self.expert_num)
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
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) \
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
            tmp = torch.zeros(self.expert_num)
            for j in range(self.expert_num):
                tmp += gate_output[i][j] * expert_output[j]
            res.append(tmp)
            
        res = torch.stack(res)
        out = []
        for i in range(self.task_num):
            out.append(self.fcs[i](res[i]))
            
        out = torch.stack(out)
        return out

class CGC(nn.Module):
    def __init__(self,expert_list,expert_num,task_num,hidden_size,input_size):
        """
        CGC input parameters
        :param expert_list: list of numbers of specific experts for different tasks 
        :param expert_num: int numbers of common experts
        :param task_num: int numbers of taks
        :param hidden_size: mlp layer input dimension
        :param input_size: data embedding size
        """
        super(CGC,self).__init__()
        self.expert_list = expert_list
        self.expert_num = expert_num
        self.task_num = task_num
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.specific_experts = []
        for i in range(task_num):
            num = expert_list[i]
            lis = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for j in range(num)])
            self.specific_experts.append(lis)
            
        self.common_experts = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for i in range(expert_num)])
        self.towers = nn.ModuleList([nn.Linear(hidden_size, 1) \
                                            for i in range(task_num)])
        gates = []
        for i in range(task_num):
            specific_num = expert_list[i]
            gate = nn.Sequential(
                nn.Linear((expert_num + specific_num) * hidden_size, expert_num + specific_num),
                nn.Softmax()
            )
            gates.append(gate)
        
        self.gates = nn.ModuleList(gates)
    
    def forward(self,x): # [user_embeddng, item_embedding]
        common_output = []
        for i in range(self.expert_num):
            common_output.append(self.common_experts[i](x))
        
        common_output = torch.stack(common_output)
        specific_output = []
        for i in range(self.task_num):
            cur_experts = self.specific_experts[i]
            tmp = []
            for j in range(len(cur_experts)):
                tmp.append(cur_experts[j](x))
            tmp = torch.stack(tmp)
            specific_output.append(tmp)

        res = []
        for i in range(self.task_num):
            tmp = torch.cat(specific_output[i], -1)
            tmp = torch.cat([tmp, common_output], -1)
            weights = self.gates[i](torch.flatten(tmp))
            feature = torch.zeros_like(common_output[0])
            for j in range(self.expert_num + self.expert_list[i]):
                feature += weights[j] * tmp[j]

            feature = self.towers[i](feature)
            res.append(feature)
        
        res = torch.stack(res)
        return res
        

class PLE(nn.Module):
    def __init__(self,expert_list,expert_num,task_num,hidden_size,input_size):
        """
        PLE input parameters
        :param expert_list: list of numbers of specific experts for different tasks 
        :param expert_num: int numbers of common experts
        :param task_num: int numbers of taks
        :param hidden_size: mlp layer input dimension
        :param input_size: data embedding size
        """
        super(PLE,self).__init__()
        self.expert_list = expert_list
        self.expert_num = expert_num
        self.task_num = task_num
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        #===================================== feature extractor =========================================
        self.specific_extractor = []
        for i in range(task_num):
            num = expert_list[i]
            lis = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for j in range(num)])
            self.specific_extractor.append(lis)
            
        self.common_extractors = nn.ModuleList([nn.Linear(input_size, hidden_size) \
                                            for i in range(expert_num)])
        ex_gates = []
        for i in range(task_num + 1):#task_gates + common_gate(there is a gate for common experts in extrator but not in final layer)
            if i < task_num:
                specific_num = expert_list[i]
            else:
                specific_num = sum(expert_list)
            ex_gate = nn.Sequential(
                nn.Linear((expert_num + specific_num) * hidden_size, expert_num + specific_num),
                nn.Softmax()
            )
            ex_gates.append(ex_gate)
        
        self.ex_gates = nn.ModuleList(ex_gates)
        #===================================== final predictor ============================================
        self.towers = nn.ModuleList([nn.Linear(hidden_size, 1) \
                                            for i in range(task_num)])
        self.specific_experts = []
        for i in range(task_num):
            num = expert_list[i]
            lis = nn.ModuleList([nn.Linear(hidden_size, hidden_size) \
                                            for j in range(num)])
            self.specific_experts.append(lis)
            
        self.common_experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) \
                                            for i in range(expert_num)])
        gates = []
        for i in range(task_num):
            specific_num = expert_list[i]
            gate = nn.Sequential(
                nn.Linear((expert_num + specific_num) * hidden_size, expert_num + specific_num),
                nn.Softmax()
            )
            gates.append(gate)
        
        self.final_gates = nn.ModuleList(gates)
    
    def forward(self,x):  # [user_embeddng, item_embedding]
        #============================= featrue extraction ============================
        common_feature = []
        for i in range(self.expert_num):
            common_feature.append(self.common_extractors[i](x))
        
        common_feature = torch.stack(common_feature)
        specific_feature = []
        for i in range(self.task_num):
            cur_extractors = self.specific_extractor[i]
            tmp = []
            for j in range(len(cur_extractors)):
                tmp.append(cur_extractors[j](x))
            tmp = torch.stack(tmp)
            specific_feature.append(tmp)

        features = []
        for i in range(self.task_num + 1):
            if i < self.task_num:
                tmp = torch.cat(specific_feature[i], -1)
                tmp = torch.cat([tmp, common_feature], -1)
            else:
                tmp = torch.cat([specific_feature, common_feature], -1)
            weights = self.ex_gates[i](torch.flatten(tmp))
            feature = torch.zeros_like(common_feature[0])
            for j in range(len(weights)):
                feature += weights[j] * tmp[j]
            features.append(feature)
        #========================== final prediction =========================
        common_output = []
        for i in range(self.expert_num):
            common_output.append(self.common_experts[i](features[-1]))
        
        common_output = torch.stack(common_output)
        specific_output = []
        for i in range(self.task_num):
            cur_experts = self.specific_experts[i]
            tmp = []
            for j in range(len(cur_experts)):
                tmp.append(cur_experts[j](features[i]))
            tmp = torch.stack(tmp)
            specific_output.append(tmp)

        res = []
        for i in range(self.task_num):
            tmp = torch.cat(specific_output[i], -1)
            tmp = torch.cat([tmp, common_output], -1)
            weights = self.final_gates[i](torch.flatten(tmp))
            feature = torch.zeros_like(common_output[0])
            for j in range(self.expert_num + self.expert_list[i]):
                feature += weights[j] * tmp[j]
            feature = self.towers[i](feature)
            res.append(feature)
        
        res = torch.stack(res)
        return res
        
        
        
        

            
    
    
    
    