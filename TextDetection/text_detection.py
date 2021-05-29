#!/usr/bin/env python
# coding: utf-8

# In[1]:


import LoadImg_ModelEAST


# In[2]:


import cv2


# In[3]:


import numpy as np


# In[4]:


from imutils.object_detection import non_max_suppression


# In[5]:


from matplotlib import pyplot as plt


# In[6]:


orig = LoadImg_ModelEAST.image.copy()
(origH, origW) = LoadImg_ModelEAST.image.shape[:2]


# In[7]:


(newW, newH) = (LoadImg_ModelEAST.args["width"], LoadImg_ModelEAST.args["height"])


# In[8]:


args = LoadImg_ModelEAST.args


# In[9]:


rW = origW / float(newW)
rH = origH / float(newH)


# In[10]:


image = cv2.resize(LoadImg_ModelEAST.image, (newW, newH))
(H, W) = image.shape[:2]


# In[11]:


blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)


# In[12]:


layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]


# In[13]:


LoadImg_ModelEAST.net.setInput(blob)
(scores, geometry) = LoadImg_ModelEAST.net.forward(layerNames)


# In[14]:


def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < LoadImg_ModelEAST.args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)


# In[15]:


(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)


# In[16]:


orig_image = orig.copy()


# In[18]:


results = []


# In[45]:


# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    cv2.rectangle(orig_image, (startX, startY), (endX, endY),(0, 0, 255), 2)
plt.imshow(orig_image)
plt.title('Text Detection')
plt.show()

