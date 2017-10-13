
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
#get_ipython().magic('matplotlib inline')


# In[4]:


image = cv2.imread('../image/raised_hand.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


# In[6]:


plt.imshow(image_copy)


# In[13]:


# Convert to Gray Image
gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)

#Convert to black and white image using BINARY THRESHOLD INVERTED  (white pixel vlues = 255,255) . 
#SO feeds in White and Returns Black

retval, binary = cv2.threshold(gray, 240,240, cv2.THRESH_BINARY_INV)


plt.imshow(binary, cmap = 'gray')


# I'll be using the CV function, findContours.This takes in our binary image, then a contour retrieval mode which I'll have as a tree,and third, is our contour approximation method which I'll put as a simple chain and the outputs are list of contours in the hierarchy. The hierarchy is useful if you have many contours nested within one another.
# The hierarchy defines their relationship to one another and you can learn more about this in text.
# 
# 
# 
# 
# 
# 

# In[15]:


#Find Countor from threshold image
retval,contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image_copy2 = np.copy(image_copy)


# To draw the contours,I'll use an OpenCV function called, drawContours.This takes in the copy over image followed by
# our list of contours then which contours to display.Negative one means all of the contours.And finally the color and size I want the contours to be or have them be a thin green line and I'll display the output. Green means (0,255,0) and size of the line =2
# 
# 
# 
# 
# 

# In[16]:


all_contours = cv2.drawContours(image_copy2,contours,-1,(0,255,0),2)


# In[17]:


plt.imshow(all_contours)


# In[ ]:




