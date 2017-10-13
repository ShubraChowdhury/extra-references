# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:30:32 2017

@author: shubra
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# X has shape (num_rows, num_cols), where the training data are stored
# as row vectors
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# y must have an output vector for each input vector
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
y = np_utils.to_categorical(y)

# Create the Sequential model
model = Sequential()

# 1st Layer - Add an input layer of 32 nodes with the same input shape as
# the training samples in X
print("Shape =",X.shape[1])
model.add(Dense(32, input_dim=X.shape[1]))

# 2rd Layer - Add a softmax activation layer
model.add(Activation('softmax'))

# 4th Layer - Add a fully connected output layer
model.add(Dense(2))

# 5th Layer - Add a sigmoid activation layer
model.add(Activation('sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer="adam", metrics = ["accuracy"])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

model.summary()
model.fit(X, y, epochs=1000, verbose=0)

score =model.evaluate(X,y)
print("Score ",score)

# Checking the predictions
print("\nPredictions:")
print(model.predict_proba(X))


print("\n MODEL WEIGHTS ",model.get_weights())

print("\n LOSS FOR X " ,model.get_losses_for(X))