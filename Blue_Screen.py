# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:42:00 2017

@author: shubra
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('pizza_bluescreen.jpg')

print('This image is :', type(image),'\n',
      'with dimensions :', image.shape)

"""
This image is : <class 'numpy.ndarray'> 
 with dimensions : (152, 217, 3) --> (height, width,color component)
"""
plt.imshow(image)
plt.show()
#COPY IMAGE
image_copy = np.copy(image)

# Matplotlib uses BGR and cv2 uses RGB  so now change it to cv2 format
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

"""
Define Lower and Upper bounds of color that I want to isolate-blue
lower_blue= contains lowest value of red, green and blue that are considered 
part of blue screen background
set 0 for red and green means its OK to have no red and green
But lowest value of blue will be still high let say 230 but not 255

Now define upper threshold 
this will have little bit red and green say 50 each
and blue will have highest value of 255
 
"""

lower_blue = np.array([0,0,230])
upper_blue = np.array([50,50,255])

"""
Create a image mask --> mask are very common way to isolate 
a selected area of interest and do something with that area

Create a mask over the blue area using a open csv inRange function

inRange --> takes image and lower and upper bound and defines a mask 
by asking if the color value of each image pixel falls
in the range of lower and upper threshold.

If it does fall in this range mask will allow it to display
and if not then the mask will block it out and turn pixel black

"""

mask = cv2.inRange(image_copy,lower_blue,upper_blue)

plt.imshow(mask,cmap='gray')
plt.show()

mask_image = np.copy(image_copy)

"""
one way to select blue screen area is asking for the part of image
that overlaps with the part of the mask that is white or not black

That is we will select part of the image where area of the 
mask is not equal to zero.

And to block this background area out we then set these pixels to black.
In RGB black is just zeros for all three color values.



"""
mask_image[mask !=0] = [0, 0, 0]

plt.imshow(mask_image)
plt.show()

# LOAD BACKGROUND
"""
First, I'll read in an image of outer space and convert it to RGB color.

"""
background_image= cv2.imread('space_background.jpg')
background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)

#Crop the image to origonal image 152, 217, 3
"""
I'll also crop it so that it's the same size as our pizza image;

"""
crop_background = background_image[0:152,0:217]

#MAsk the cropped background so that the pizza area is blocked
"""
Then I'll apply the mask, meaning I want the background 
to show through and not the Pizza area.
If we look back at the mask in this case I'm blocking

the part of the background image where the mask is equal to zero.
And for this we say mask == 0.


"""

crop_background[mask == 0] = [0,0,0]

plt.imshow(crop_background)
plt.show()

"""
Then finally I just need to add these two images together.
Since the black area is equivalent to zeros in pixel color value,
a simple addition will work.

"""

complete_image = crop_background + mask_image


plt.imshow(complete_image)
plt.show()


