# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:23:04 2017

@author: shubra
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load in the face detector XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read in an image
image = cv2.imread('../image/face1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()
#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

plt.imshow(faces)
plt.show()