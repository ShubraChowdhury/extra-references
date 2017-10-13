# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 08:17:33 2017

@author: shubra
"""

import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="C:/Training/udacity/AI_NanoDegree/Term2/2.Deep Neural Networks Videos/datasets/imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)


b = np.load("C:/Training/udacity/AI_NanoDegree/Term2/2.Deep Neural Networks Videos/datasets/imdb.npz")
print(b.files)
print(b['x_test'][0])
print(b['x_train'][0])
print(b['y_train'][0])
print(b['y_test'][0])