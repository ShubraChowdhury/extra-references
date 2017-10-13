# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:51:40 2017

@author: shubra
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in the image
image = cv2.imread('stop_sign.jpg')
# Convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ---------------------------------------------------------- #

## TODO: Define the color selection criteria
lower_red = np.array([100,0,0]) 
upper_red = np.array([255,90,90])

# Mask the image 
masked_image = np.copy(image)
mask = cv2.inRange(masked_image, lower_red, upper_red)

## TODO: Apply the mask to masked_image
## by setting the pixels in the red range to black
## Click `Test Run` to display the output before submitting

masked_image[mask !=0] = [0, 0, 0]

# ---------------------------------------------------------- #
# Display it
plt.imshow(masked_image)