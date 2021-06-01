#!/usr/bin/env python
# coding: utf-8

import os
import cv2
from dask.rewrite import args

class t():
    def __init__(self):
        self.args = {
                "image": "../exampleIMG/textdetection4.jpg",
                "east": "../modeltrained/east_text_detection.pb",
                "min_confidence": 0.5,
                "width": 320,
                "height": 320
        }

    def get_args(self):
        return self.args['image']

    def loadModel(self):
        net = cv2.dnn.readNet(self.args["east"])
        return net

    def loadImage(self):
        image = cv2.imread(self.args['image'])
        return image
