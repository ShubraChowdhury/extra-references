# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:46:33 2017

@author: shubra
"""

#import math

def batches(batch_size, features, labels):
    
    assert len(features) == len(labels)
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i+batch_size
        
        batch = [features[start_i:end_i],labels[start_i:end_i] ]
        output_batches.append(batch)
    return output_batches
    
    
from pprint import pprint

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))
   