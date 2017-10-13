# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:00:01 2017

@author: shubra
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=3, input_shape=(150, 150, 15)))
model.summary()