
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
get_ipython().magic('matplotlib inline')


# In[2]:


image = cv2.imread('../image/butterfly.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)


# In[4]:


#Prepare data for K -Mean, reshape image of 2D array of pixels and 3 Color values RGB

pixel_vals = image_copy.reshape((-1,3))
#convert to float
pixel_vals = np.float32(pixel_vals)


# In[7]:


#Define Stopping Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#Define K-Mean Cluster
k = 6
retval,labels,centers=cv2.kmeans(pixel_vals,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


# In[9]:


# Convert data into 8 bit value
centers = np.uint8(centers)
segment_data = centers[labels.flatten()]


# In[11]:


# Reshape data into original image
segment_data = segment_data.reshape(image_copy.shape)
label_reshape = labels.reshape(image_copy.shape[0],image_copy.shape[1])


# In[12]:


plt.imshow(segment_data)


# In[14]:


#VISUALIZE ONE Segment

plt.imshow(label_reshape ==1, cmap='gray' )


# In[15]:


plt.imshow(label_reshape ==0, cmap='gray' )


# In[16]:


plt.imshow(label_reshape ==4, cmap='gray' )


# In[17]:


plt.imshow(label_reshape !=1, cmap='gray' )


# In[ ]:




