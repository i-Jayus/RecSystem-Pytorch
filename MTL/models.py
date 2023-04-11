# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:05:40 2023

@author: 12709
"""
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import heapq

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
    
class kuaishouEBR(nn.Module):
    def __init__(self,k,hidden_size,input_size,task_num,seq_len,recall_num):
        """
        kuaishouEBR input parameters
        :param k: the number of clusters for k-means
        :param hidden_size: mlp hidden_size
        :param input_size: data embedding size
        :param task_num: the number of tasks
        :param seq_len: the length of history sequence
        :param recall_num: the number of recall items
        """
        super(kuaishouEBR,self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.task_num = task_num  # the task_num seems to be equal to k? 
        self.seq_len = seq_len
        self.recall_num = recall_num
        self.user_tower = nn.Sequential(
            nn.Linear((input_size * seq_len) * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.TAL = nn.Sequential(  # the sample weight from k parts
            nn.Linear(hidden_size * (k+1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k)
        )
        self.k_means = KMeans(n_clusters = k, random_state=0) # This step is better to be preprocessed in dataset preprocessing.
        self.prompt_embedding = nn.Embedding(k, input_size)   # Here I just give a instance because of complexity.

    def forward(self,user,item):
        """
        :param user: user embedding, the concat of item embedding of history behavior of user
        :param item: list of embeddings of all items
        """
        clusters = self.k_means.fit(item.detach().numpy())
        prompts = clusters.labels_
        buckets = [[] for _ in range(self.k)]
        prompt_table = {} #record the cluster indicatior of item
        for i in range(len(prompts)): #split the item pool into k parts
            buckets[prompts[i]].append(item[i])
            prompt_table[item[i]] = prompts[i]

        prompted_user = []
        for his_item in user: 
            prompt = prompt_table[his_item]
            tmp = torch.flatten(torch.cat(user[i], self.prompt_embedding(prompt), -1))
            prompted_user.append(tmp)

        prompted_user = torch.flatten(prompted_user)    
        user = self.user_tower(prompted_user)
        k_ans = [] # kth top_k recall item
        for i in range(self.k): # this step can be run in parallel.
            heap = []
            heapq.heapify(heap)
            for j in range(len(buckets[i])):
                item = buckets[i][j]
                item = self.item_tower(item)
                sim = torch.cosine_similarity(user, item)
                if len(heap) < self.recall_num:
                    heapq.heappush((sim, item)) # here is better to return item_id 
                else:
                    heapq.heappush((sim, item))
                    heapq.heappop()
            k_ans.append(heap[:,1])
        
        res = []
        for i in range(self.recall_num):
            kth_item = torch.cat([k_ans[j][i] for j in range(self.k)], -1)
            kth_inter = torch.flatten(torch.cat([user, kth_item], -1))
            res.append(self.TAL(kth_inter))
        
        res = torch.stack(res)
        return res
        


        
        
        
        
        
        

            
    
    
    
    