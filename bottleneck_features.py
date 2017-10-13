# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 14:45:49 2017

@author: shubra
"""

"""
1. Load and Preprocess Sample Images
Before supplying an image to a pre-trained network in Keras, there are some required preprocessing steps. You will learn more about this in the project; for now, we have implemented this functionality for you in the first code cell of the notebook. We have imported a very small dataset of 8 images and stored the preprocessed image input as img_input. Note that the dimensionality of this array is (8, 224, 224, 3). In this case, each of the 8 images is a 3D tensor, with shape (224, 224, 3).
"""

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import glob

img_paths = glob.glob("C:/Training/udacity/AI_NanoDegree/Term2/3.Convolutional Neural Networks Videos/code/aind2-cnn-master/transfer-learning/images/*.jpg")

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

# calculate the image input. you will learn more about how this works the project!
img_input = preprocess_input(paths_to_tensor(img_paths))

print(img_input.shape)

"""
2. Recap How to Import VGG-16

Recall how we import the VGG-16 network (including the final classification layer) that has been pre-trained on ImageNet.
"""
from keras.applications.vgg16 import VGG16
model = VGG16()
model.summary()

print(" Before Removing Final Layer \n",model.predict(img_input).shape)

"""
3. Import the VGG-16 Model, with the Final Fully-Connected Layers Removed
"""

from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False)
model.summary()

print(" After Removing Final Layer \n",model.predict(img_input).shape)

#score = model.evaluate(img_input)
#print("Score = ", score)