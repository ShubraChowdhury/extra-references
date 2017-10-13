# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:35:34 2017

@author: shubra
"""

import keras
from keras.datasets import cifar10

#path="cifar-10-python.tar.gz"
"""
1. Load CIFAR-10 Database
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

"""
Returns:
2 tuples:
x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
"""
"""
2. Visualize the First 10 Training Images
"""
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1, xticks=[],yticks=[])
#    ax.imshow(np.squeeze(x_train[i]))
    ax.imshow(x_train[i])
    ax.set_title(str(y_train[i]))
"""
3. Rescale the Images by Dividing Every Pixel in Every Image by 255
"""
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

"""
4. Break Dataset into Training, Testing, and Validation Sets
"""
from keras import utils
import numpy as np

num_classes = len(np.unique(y_train))
#print(num_classes)

# Change labels into categorical data

y_train = utils.to_categorical(y_train,num_classes)
y_test = utils.to_categorical(y_test,num_classes)

(x_train,x_valid) = x_train[5000:],x_train[:5000]
(y_train,y_valid) = y_train[5000:],y_train[:5000]


print('x train shape ', x_train.shape)

print('train sample ', x_train.shape[0])
print('test sample ', x_test.shape[0])
print('valid sample ', x_valid.shape[0])

"""
5. Define the Model Architecture
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()

"""
6. Compile the Model
"""
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

"""
7. Train the Model
"""
from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer], 
          verbose=2, shuffle=True)

"""
8. Load the Model with the Best Validation Accuracy
"""
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')


"""
9. Calculate Classification Accuracy on Test Set
"""
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

"""
10. Visualize Some Predictions
"""
# get predictions on the test set
y_hat = model.predict(x_test)

"""
10. Visualize Some Predictions
"""

# define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))