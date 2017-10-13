
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
#get_ipython().magic('matplotlib inline')


# In[3]:


image = cv2.imread('Sunflower.jpg')


# In[4]:


image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


# In[5]:


plt.imshow(image_copy)
plt.show()

# In[7]:


gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

# In[9]:


#Define Lower and upper threshold for Hysteresis
lower = 120
upper = 240

edges = cv2.Canny(gray, lower, upper)


# In[11]:


plt.imshow(edges, cmap='gray')
plt.show()

# In[12]:


wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 180, 240)


# In[14]:


f, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')


# In[ ]:




