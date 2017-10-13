# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:09:36 2017

@author: shubra
"""

import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
#    print(X[0][0], W[0])
    for i in range(len(X)):
        y_predict = prediction(X[i],W,b)
        if y[i]-y_predict == 1:
            W[0] = W[0]+X[i][0]*learn_rate
            W[1] = W[1]+X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_predict == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
#    print (X.T[0].min())
    
#    x_min, x_max = min(X.T[0]), max(X.T[0])
#    y_min, y_max = min(X.T[1]), max(X.T[1])
    
    
    x_min, x_max = X.T[0].min(), X.T[0].max()
    y_min, y_max = X.T[1].min(), X.T[1].max()
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data.csv")
    df.columns = ["X", "y", "b"]
    print (df.head())
#    print(df.y,df.X,df.b)
    
    X = df.X
    y =df.y
    b=df.b
#    perceptronStep(X, y, W, b, learn_rate = 0.01)
    trainPerceptronAlgorithm(X, y)
#    with open('data.csv') as csvfile:
#        x = csv.reader(csvfile)
#        for row in x:
##            print(row[:-1])
#            X.append(row[:-1])
#            y.append(row[2])
#            trainPerceptronAlgorithm(row[:-1],row[2])