# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:29:24 2017

@author: shubra
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('water_baloons.jpg')
image_copy = np.copy(image)
print(type(image), image.shape)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)

#RGB Channels

r = image_copy[:, :, 0]
g = image_copy[:, :, 1]
b = image_copy[:, :, 2]



f, (ax1, ax2, ax3)= plt.subplots(1,3, figsize=(20,10))
ax1.set_title("red")
ax1.imshow(r, cmap='gray' )
ax2.set_title("green")
ax2.imshow(g , cmap='gray')
ax3.set_title("blue")
ax3.imshow(b , cmap='gray')


#convert RGB to HSV
hsv = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HSV)

h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]


f, (ax1, ax2, ax3)= plt.subplots(1,3, figsize=(20,10))
ax1.set_title("Hue")
ax1.imshow(h , cmap='gray')
ax2.set_title("Saturation")
ax2.imshow(s , cmap='gray')
ax3.set_title("Value")
ax3.imshow(v , cmap='gray')


#Define Color Selection criteria for RGB
lower_pink=np.array([180,0,100])
upper_pink=np.array([255,255,230])

#Define Color Selection criteria for HSV values
lower_hsv=np.array([160,0,0])
upper_hsv=np.array([180,255,255])


msk_rgb = cv2.inRange(image_copy,lower_pink,upper_pink)

mask_image = np.copy(image_copy)
mask_image[msk_rgb ==0]=[0,0,0]


plt.imshow(mask_image)
plt.title("RGB Selection")
plt.show()


#HSV MAsk
msk_hsv = cv2.inRange(image_copy,lower_hsv,upper_hsv)
mask_image = np.copy(image_copy)
mask_image[msk_hsv ==0]=[0,0,0]
plt.imshow(mask_image)
plt.title("HSV Selection")
plt.show()



