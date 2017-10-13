
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

get_ipython().magic('matplotlib inline')


# In[2]:


image = cv2.imread('brain_mr.JPG')


# In[3]:


plt.imshow(image)


# In[5]:


image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)


# In[6]:


gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)


# In[7]:


gray_blur = cv2.GaussianBlur(gray, (5,5),0)


# In[15]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.set_title("Origonal")
ax1.imshow(gray, cmap="gray")

ax2.set_title("Blur")
ax2.imshow(gray_blur, cmap="gray")


# In[28]:


sobel_x = np.array([[-2,0,2],
                    [-1,0,1],
                    [-2,0,2]])

filtered = cv2.filter2D(gray, -1, sobel_x)
filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_x)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.set_title("filtered")
ax1.imshow(filtered, cmap="gray")

ax2.set_title("filtered Blur")
ax2.imshow(filtered_blurred, cmap="gray")


# In[26]:





# In[ ]:




