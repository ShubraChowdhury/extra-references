# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:31:00 2017

@author: shubra
"""

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

if __name__ == "__main__":
    Y =[1,1,0]
    P =[.8,.7,.1]
    print(cross_entropy(Y, P))

    Y =[0,0,1]
    P =[.8,.7,.1]
    print(cross_entropy(Y, P))    