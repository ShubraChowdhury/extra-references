
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
#get_ipython().magic('matplotlib inline')


# In[4]:


image = cv2.imread('../image/phone.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)


# In[6]:


gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')


# In[8]:


#Parameter for Canny
low_tresh = 50
high_thresh = 100
edges = cv2.Canny(gray, low_tresh,high_thresh)
plt.imshow(edges, cmap='gray')


# In[12]:


#Define Hough Transform Parameters
rho =1
theta = np.pi/180
threshold = 60
min_line_length = 450
max_line_gap = 5
# Find lines using Hough transform
lines = cv2.HoughLinesP(edges,rho,theta,threshold,min_line_length,max_line_gap)

line_image = np.copy(image_copy)
# iterate over lines and draw line over image

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(line_image)
        


# In[ ]:




