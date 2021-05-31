#!/usr/bin/env python
# coding: utf-8


import sys
import LoadImg_ModelEAST
from LoadImg_ModelEAST import t

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt


class Img_proc():

    def __init__(self):
        self.LoadImg = t()
        self.orig = self.LoadImg.loadImage().copy()
        self.results = []
        self.args = self.LoadImg.args
        self.net = self.LoadImg.loadModel()

    def resizeIMG(self, orig, args):
        (origH, origW) = orig.shape[:2]

        (newW, newH) = (self.args["width"], self.args["height"])

        rW = origW / float(newW)
        rH = origH / float(newH)

        image = cv2.resize(orig, (newW, newH))
        (H, W) = image.shape[:2]
        return [H, W, rW, rH]

    def blob(self, orig):
        H, W, rW, rH = self.resizeIMG(self.orig, self.args)
        blob1 = cv2.dnn.blobFromImage(orig, 1.0, (H, W),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        return blob1

    def geomatric(self, net):
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        net.setInput(self.blob(self.orig))
        (scores, geometry) = net.forward(layerNames)
        return scores, geometry

    def predictions(self, prob_score, geo):

        #print(prob_score, geo)
        (numR, numC) = prob_score.shape[2:4]
        #print(numR, numC)
        boxes = []
        confidence_val = []

        for y in range(0, numR):
            scoresData = prob_score[0, 0, y]
            x0 = geo[0, 0, y]
            x1 = geo[0, 1, y]
            x2 = geo[0, 2, y]
            x3 = geo[0, 3, y]
            anglesData = geo[0, 4, y]
            #print(scoresData[y], '{} th'.format(y))
            for i in range(0, numC):
                if scoresData[i] <= self.args["min_confidence"]:
                #print(scoresData[i])
                    continue

                (offX, offY) = (i * 4.0, y * 4.0)
                angle = anglesData[i]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x0[i] + x2[i]
                w = x1[i] + x3[i]
                endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                startX = int(endX - w)
                startY = int(endY - h)
                boxes.append((startX, startY, endX, endY))
                confidence_val.append(scoresData[i])

        return boxes, confidence_val

    def finalBox(self, net):
        (score, geometry) = self.geomatric(self.net)
        (boxes, confidence_val) = self.predictions(score, geometry)
        boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
        return boxes

    def output(self, orig, boxes):
        orig_image = orig.copy()
        (H, W), (rW, rH) = self.resizeIMG(self.orig, self.args)
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        plt.imshow(orig_image)
        plt.title('Text Detection')
        plt.show()

