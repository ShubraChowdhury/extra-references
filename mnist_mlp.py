# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:12:24 2017

@author: shubra
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

"""
1. Load MNIST Database
"""
# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data(path="C:/Training/udacity/AI_NanoDegree/Term2/3.Convolutional Neural Networks Videos/datasets/mnist.npz")

print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))

print("X_train",X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)

"""
Get the name of the Class this I will use to compare the 
Predicted value/label to the value/label of test
"""
num_classes = (np.unique(y_train))
print(num_classes)

"""
2. Visualize the First Six Training Images
"""
# plot first six training images
fig = plt.figure(figsize=(10,10))
for i in range(6):
    ax = fig.add_subplot(1, 6, i+1, xticks=[i], yticks=[i])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
    print("Integer at position %d is "%i,y_train[i])
#    print(y_test[i])
#    print(X_train[i])
""" ANNOTATE x=2 and y=1 and the text "local max" will be written
at x =3 and y =1.5  AND (2,1) and (3,1.5) will be connected by an ARROW
ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
"""    

"""
3. View an Image in More Detail
"""
def visualize_input(img,ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            """
            ANNOTATE x and y coordinate of the Image supplied in the 
            method call
            """
#            ax.annotate(img[x][y],xy=(x,y))
            """
            ANNOTATE with letters printed in the center
            """
#            ax.annotate(img[x][y],xy=(x,y),horizontalalignment='center',
#                        verticalalignment='center')
            """
            NOW ANNOTATE with COLOR condition, if the value of 
            x, y is less than threshold then make it white 
            else blue
            """
            ax.annotate(str(round(img[x][y],2)),xy=(y,x),horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'blue')
    
fig = plt.figure(figsize=(12,12))   
ax = fig.add_subplot(111) 
visualize_input(X_train[4], ax)

"""
4. Rescale the Images by Dividing Every Pixel in Every Image by 255
# rescale [0,255] --> [0,1]
"""
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

"""
5. Encode Categorical Integer Labels Using a One-Hot Scheme
"""    
from keras.utils import np_utils
print("Before One Hot At Position 4th",y_train[4])
print("Integer upto 10 \n",y_train[:10])

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10) 
print("After One Hot At Position 4th",y_train[4])
print("Integer upto 10 \n",y_train[:10])

print(X_train.shape[1:])


"""
6. Define the Model Architecture
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()

"""
7. Compile the Model
"""
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])

"""
8. Calculate the Classification Accuracy on the Test Set (Before Training)
"""

score = model.evaluate(X_test,y_test,verbose=1)
print("Scores =",score)
accuracy = 100*score[1]
# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)

"""
9. Train the Model
"""
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1,save_best_only=True)

#checkpointer = ModelCheckpoint(filepath='mnist_model_best_hdf5.txt', verbose=1,save_best_only=True)

hist=model.fit(X_train,y_train,batch_size=128,epochs=2,
               validation_split=0.2, callbacks=[checkpointer],
               verbose=1,shuffle=True)
"""
10. Load the Model with the Best Classification Accuracy on the Validation Set
"""
# load the weights that yielded the best validation accuracy
model.load_weights('mnist.model.best.hdf5')

#wts = model.get_weights()
#print(wts[0:1])

"""
11. Calculate the Classification Accuracy on the Test Set
"""

# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)

y_hat = model.predict(X_test)




fig = plt.figure(figsize=(10,10))
for i in range(2):
    ax = fig.add_subplot(1,2, i+1, xticks=[i], yticks=[i])
#    ax.imshow(X_test[i], cmap='gray')
    ax.imshow(np.squeeze(X_test[i]), cmap='gray')
    pred_idx = np.argmax(y_hat[i])
    true_idx = np.argmax(y_test[i])
    ax.set_title("{} ({})".format(num_classes[pred_idx], num_classes[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
#    ax.set_title(str(y_test[i]))
#    print("Integer at position %d is "%i,y_test[i])
    
    