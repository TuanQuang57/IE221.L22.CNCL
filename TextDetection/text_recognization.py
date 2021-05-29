#!/usr/bin/env python
# coding: utf-8

# In[3]:


import text_detection


# In[18]:


import cv2


# In[4]:


import pytesseract


# In[20]:


from matplotlib import pyplot as plt


# In[5]:


pytesseract.pytesseract.tesseract_cmd = r'D:\IT\Python3\Tesseract-OCR\tesseract.exe'


# In[14]:


boxes = text_detection.boxes
rW = text_detection.rW
rH = text_detection.rH
orig = text_detection.orig


# In[15]:


results = []


for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]

#Convert image to string.  
	configuration = ("-l eng --oem 1 --psm 8")

	text = pytesseract.image_to_string(r, config=configuration)

	results.append(((startX, startY, endX, endY), text))


# In[23]:


# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	print("{}\n".format(text))


# In[36]:


orig_image = orig.copy()

for ((start_X, start_Y, end_X, end_Y), text) in results:

	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 255, 0), 1)
	cv2.putText(orig_image, text, (start_X, start_Y+2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0, 255), 1)

plt.imshow(orig_image)
plt.title('Output')
plt.show()


# In[35]:


plt.figure(figsize=(1440000,1440000))


# In[ ]:





# In[ ]:




