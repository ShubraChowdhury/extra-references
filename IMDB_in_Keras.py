print(" After Removing Final Layer \n",model1.predict(img_input_test).shape)# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:45:24 2017

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
                                                      num_words=1000,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)
print("Shape of x_train",x_train.shape)
print("Shape of x_test",x_test.shape)

"""
1. Restricting the words to a limit of 1000 means the words will be matched 
   for words with index 1000 or less , refer kera_tokenizer.py to see
   how the index works 
   tokenizer = Tokenizer(num_words=1000)
   
2. If a match is found in the words then it will be 1 if not matched then it will
   be =0
   x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')

"""
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

"""
This will change the shape of x_train and x_test
"""
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)
print("Shape of x_train",x_train.shape)
print("Shape of x_test",x_test.shape)


"""
Some Validation
"""
i = 0
for j,x_train_match in enumerate(x_train[24999]):
#    print(x_train_match)
    if x_train_match == 1.0:
        i +=1
print("For x_train row 24999 the number of match with index <= 1000 is ",i)  


i = 0
for j,x_test_match in enumerate(x_test[24999]):
#    print(j,x_train_match)
    if x_test_match == 1.0:
        i +=1
print("For x_test row 24999 the number of match with index <= 1000 is ",i) 

print(len(x_train))

i = 0
row_num_array =[]
x_test_array_value_match = []
for row_num,x_test_array in enumerate(x_test):
    row_num_array.append(row_num)
    for x_test_array_num,x_test_array_value in enumerate(x_test_array):
#        print(row_num,x_test_array_num,x_test_array_value)
    
        if x_test_array_value == 1:
            i +=1
            
            x_test_array_value_match.append(i)
    i = 0
    
#for rn,cnt in zip(row_num_array,x_test_array_value_match):    
#    print("For row :", rn," number of match with index <= 1000 is ",cnt)  


print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)
print("Shape of x_train",x_train.shape)
print("Shape of x_test",x_test.shape)
print(y_train[24999])
print(y_test[24999])

"""

This will convert y_train and y_test to a (25000,2) 
"""
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)
print("Shape of x_train",x_train.shape)
print("Shape of x_test",x_test.shape)
print(y_train[24999])
print(y_test[24999])

"""
Building and Compiling a Model
"""


model = Sequential()
model.add(Dense(512,activation='relu', input_dim=1000)) 
"""
Above will give Parameter = 512*1000 + 512 =512512
"""
model.add(Dropout(0.5))
model.add(Dense(2 ,activation='softmax'))
"""
Above will give Parameter = 2*512 (as output is set only once) + 2 =1026
"""

model.summary()
"""
Total Parameter = 512512+1026 = 513538
"""
# TODO: Compile the model using a loss function and an optimizer.
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics = ['accuracy'])

"""
FIT Training data and pass testing data as validation 
"""

hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test), 
          verbose=2)

score_test = model.evaluate(x_test, y_test, verbose=0)
print("Test Data Accuracy: ", score_test[1])


score_train = model.evaluate(x_train, y_train, verbose=0)
print("Train Data Accuracy: ", score_train[1])


print(model.predict_proba(x_test))


print("\nTesting %s: %.2f%%" % (model.metrics_names[0],score_test[0]*100))
print("\nTesting %s: %.2f%%" % (model.metrics_names[1],score_test[1]*100))
print("\nTraining %s: %.2f%%" % (model.metrics_names[0],score_train[0]*100))
print("\nTraining %s: %.2f%%" % (model.metrics_names[1],score_train[1]*100))