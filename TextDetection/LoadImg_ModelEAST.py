#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[4]:


import os


# In[5]:


os.path.abspath(os.getcwd())



# In[7]:


os.chdir('text-detection')



# In[2]:


args = {"image":"../text-detection/textdetection4.jpg", "east":"../text-detection/east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}


# In[9]:


args['image']="../text-detection/textdetection4.jpg"
image = cv2.imread(args['image'])


# In[10]:


net = cv2.dnn.readNet(args["east"])

