# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 23:24:58 2017

@author: shubra
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=3, input_shape=(150, 150, 15)))
model.summary()