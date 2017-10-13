# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 21:19:46 2017

@author: shubra
"""

"""
1. Load CIFAR-10 Database
"""
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)= cifar10.load_data()

"""
2. Visualize the First 24 Training Images
"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(20,2))

for i in range(24):
    ax= fig.add_subplot(2,12,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    ax.set_title(y_train[i])

"""
3. Rescale the Images by Dividing Every Pixel in Every Image by 255
"""

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

"""
4. Break Dataset into Training, Testing, and Validation Sets
"""
# break training set into training and validation sets
(x_train,x_valid)=x_train[5000:],x_train[:5000]
(y_train,y_valid)=y_train[5000:],y_train[:5000]

num_clases= len(np.unique(y_train))
from keras import utils 
y_train = utils.to_categorical(y_train,num_clases)
y_valid = utils.to_categorical(y_valid,num_clases)
y_test = utils.to_categorical(y_test,num_clases)


# print shape of training set
print('x_train shape:', x_train.shape)

# print number of training, validation, and test images
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')

"""
5. Create and Configure Augmented Image Generator
"""

from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
        width_shift_range=0.1,height_shift_range=0.1,
        horizontal_flip=True)

datagen_valid = ImageDataGenerator(
        width_shift_range=0.1,height_shift_range=0.1,
        horizontal_flip=True)

datagen_train.fit(x_train)
datagen_valid.fit(x_valid)

"""
6. Visualize Original and Augmented Images
"""

# Subset of Original data

x_train_subset = x_train[:12]
fig = plt.figure(figsize=(20,2))
for i in range(0,len(x_train_subset)):
    ax = fig.add_subplot(1,12,i+1)
    ax.imshow(x_train_subset[i])
fig.suptitle("Subset of Original Training Image",fontsize=10)
    

# visualize augmented images
fig = plt.figure(figsize=(20,2))
for x_batch in datagen_train.flow(x_train_subset, batch_size=12):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(x_batch[i])
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break;
    
"""
7. Define the Model Architecture
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
8. Compile the Model
"""
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

"""
9. Train the Model
"""
from keras.callbacks import ModelCheckpoint   

batch_size = 32
epochs = 100

# train the model
checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=datagen_valid.flow(x_valid, y_valid, batch_size=batch_size),
                    validation_steps=x_valid.shape[0] // batch_size)


"""
10. Load the Model with the Best Validation Accuracy
"""

model.load_weights('aug_model.weights.best.hdf5')


"""
11. Calculate Classification Accuracy on Test Set
"""
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])


"""
12. Visualize Some Predictions
"""
# get predictions on the test set
y_hat = model.predict(x_test)



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