# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:59:48 2023

@author: 12709
"""

def getMaxValue(self , s: str, k: int) -> int:
    n = len(s) #前i位分成j段最小值
    dp = [[float('inf')]*(k+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1,n+1):
        for j in range(1,k+1):
            for m in range(i):
                score = (i-m)*len(set(s[m:i]))
                dp[i][j] = min(dp[i][j],dp[m][j-1]+score)
    return dp[n][k]

s = 'abacb'
k = 2
print(getMaxValue(s,k))
    