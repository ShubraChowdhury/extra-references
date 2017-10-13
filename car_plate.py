# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:04:16 2017

@author: shubra
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read in the image
image = cv2.imread('car_plate.jpg')
# Convert to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ---------------------------------------------------------- #

## TODO: Define the geometric tranform function
## This function take in an image and returns a 
## geometrically transformed image
def geo_tx(image):
    image_size = (image.shape[1], image.shape[0])
    
    ## TODO: Define the four source coordinates
    source_pts = np.float32(
        [[0, 0],
         [0, 1300],
         [2000, 1300],
         [2000, 0]])
    
    
    
    ## TODO: Define the four destination coordinates    
    ## Tip: These points should define a 400x200px rectangle
    warped_pts = np.float32(
        [[-50, -60],
         [-50, 2000],
         [2000, 1000],
         [1950, -160]])
    
    ## TODO: Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(source_pts,warped_pts)
    
    # Compute the inverse
    M_inv = cv2.getPerspectiveTransform(warped_pts,source_pts)
    
    ## TODO: Using M, create a warped image named `warped`
    warped =  cv2.warpPerspective(image,M,image_size, flags= cv2.INTER_LINEAR)

    return warped
    
    
# ---------------------------------------------------------- #
# Make a copy of the original image and warp it
warped_image = np.copy(image)
warped_image = geo_tx(warped_image)

if(warped_image is not None):
    # Visualize
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Source image')
    ax1.imshow(image)
    ax2.set_title('Warped image')
    ax2.imshow(warped_image)
else:
    print('No warped image was returned by your function.')