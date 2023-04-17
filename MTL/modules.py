# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:39:38 2023

@author: Jayus
"""
import torch
from torch import nn
import numpy as np

class AIT(nn.Module): #adaptive information transfer module for AITM
    def __init__(self, input_size, hidden_size):
        """
        AIT input parameters
        :param hidden_size: hidden_size
        :param input_size: input_size of [p_{t-1},qt]
        """
        super(AIT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.h1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.h2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.h3 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.transfer_unit = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, p, q): 
        p1 = self.h1(p) 
        p2 = self.h2(p)
        p3 = self.h3(p)
        q1 = self.h1(q)
        q2 = self.h2(q)
        q3 = self.h3(q)
        wp = torch.dot(p2,p3) / torch.sqrt(p2.shape[-1])
        wq = torch.dot(q2,q3) / torch.sqrt(q2.shape[-1])
        w = wp + wq
        wp, wq = wp / w, wq / w
        z = wp * p1 + wq * q1 #current info 
        p = self.transfer_unit(z) #transfer info to next task
        return z, p 