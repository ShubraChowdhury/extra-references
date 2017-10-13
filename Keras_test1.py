# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:35:47 2017

@author: shubra
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()


model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
#model.add(Dense(units=10))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
##
### y must have an output vector for each input vector
#y = np.array([[0], [0], [0], [1]], dtype=np.float32)
#
#model.fit(X, y, epochs=5, batch_size=32)