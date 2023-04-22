# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:55:31 2023

@author: Jayus
"""
def dimJudge(dim1,dim2,dim3):
    assert dim1 == dim2 or dim1 == dim3, 'dimension is not correct'

def Hamming_distance(x1, x2):
    z = x1 ^ x2
    res = 0
    while z:
        res += z & 1
        z = z >> 1
    return res

def Hamming_distance_list(x1, x2):
    res = 0
    assert len(x1) == len(x2), 'the length dose not match'
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            res += 1
    return res




