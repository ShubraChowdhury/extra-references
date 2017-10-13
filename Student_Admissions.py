# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:58:23 2017

@author: shubra
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_loc='C:/Training/udacity/AI_NanoDegree/Term2/2.Deep Neural Networks Videos/code/aind2-dl-master/student_data.csv'
data = pd.read_csv(file_loc)
print(data.head())

def plot_points(data1):
    X = np.array(data1[["gre","gpa"]])
    y = np.array(data1["admit"])
    
    admitted = X[(np.argwhere (y ==1))]
    rejected = X[ (np.argwhere (y ==0))]
#    print(admitted,rejected)
    plt.scatter([s[0][0] for s in rejected],[s[0][1] for s in rejected],s=25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted],[s[0][1] for s in admitted],s=25, color = 'green', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
plot_points(data)
plt.show()

#rank1 = data[data["rank"]==1]
#rank2 = data[data["rank"]==2]
#rank3 = data[data["rank"]==3]
#rank4 = data[data["rank"]==4]
#print(rank1['admit'].count(),rank2['admit'].count(),rank3['admit'].count(),rank4['admit'].count())
#
#plot_points(rank1)
#plt.title("Rank 1")
#plt.show()
#
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


import keras
from keras.utils import np_utils

# remove NaNs
data = data.fillna(0)

# One-hot encoding the rank
processed_data = pd.get_dummies(data, columns=['rank'])

print(processed_data.head())

# Normalizing the gre and the gpa scores to be in the interval (0,1)
processed_data["gre"] = processed_data["gre"]/800
processed_data["gpa"] = processed_data["gpa"]/4


print(processed_data.head())

"""
Now, we split our data input into X, and the labels y , and one-hot encode 
the output, so it appears as two classes (accepted and not accepted).
Get all rows except the first column of Admit 
"""
X = np.array(processed_data)[:,1:] 
#y = np_utils.to_categorical(np.array(data["admit"],2))

y = keras.utils.to_categorical(data["admit"],2)
 
print(y[0])

# Checking that the input and output look correct
print("Shape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nFirst 10 rows of X")
print(X[:10])
print("\nFirst 10 rows of y")
print(y[:10])

# break training set into training and validation sets
(X_train, X_test) = X[50:], X[:50]
(y_train, y_test) = y[50:], y[:50]

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#model = Sequential()
#model.add(Dense(128,  activation='relu', input_shape=(7,)))

model = Sequential()
model.add(Dense(128, input_dim=7))
model.add(Activation('sigmoid'))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=0)


# Scoring the model
score = model.evaluate(X_train, y_train)
print("\nAccuracy: ", score[-1])
print(score)

score = model.evaluate(X_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(X_test, y_test)
print("\n Testing Accuracy:", score[1])

# Checking the predictions
print("\nPredictions:")
print(model.predict_proba(X_test))
#print("\n x_test",X_test)

