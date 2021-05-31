#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import text_detection
from text_detection import Img_proc
import cv2
import pytesseract
from matplotlib import pyplot as plt
import LoadImg_ModelEAST
from LoadImg_ModelEAST import t
from imutils.object_detection import non_max_suppression
pytesseract.pytesseract.tesseract_cmd = r'D:\IT\Python3\Tesseract-OCR\tesseract.exe'


class text_recog(Img_proc):
    def __init__(self):
        Img_proc.__init__(self)
        super().__init__()
        self.LoadImg = self.LoadImg
        self.args = self.args
        self.orig = self.orig
        self.net = self.net

    def resizeIMG(self, orig, args):
        super().resizeIMG(self.orig, self.args)

    def geomatric(self, net):
        super().geomatric(self.net)

    def predictions(self, prob_score, geo):
        score, geo = super().geomatric(self.net)
        super().predictions(score, geo)

    def finalBox(self, net):
        super().finalBox(self.net)

    def converted(self, boxes):
        H, W, rW, rH = super().resizeIMG(self.orig, self.args)
        print(H, W, rW, rH)
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            r = self.orig[startY:endY, startX:endX]
            configuration = ("-l eng --oem 1 --psm 8")
            text = pytesseract.image_to_string(r, config=configuration)
            self.results.append(((startX, startY, endX, endY), text))
        return self.results

    def output_text(self, results):
        bag_of_word = set()
        for ((start_X, start_Y, end_X, end_Y), text) in results:
            bag_of_word.add("{}\n".format(text))

        for word in bag_of_word:
            print(word)

    def output_textonpic(self, results):
        orig_image = self.orig.copy()

        for ((start_X, start_Y, end_X, end_Y), text) in results:
            text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
            cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y), (0, 255, 0), 1)
            cv2.putText(orig_image, text, (start_X, start_Y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        plt.imshow(orig_image)
        plt.title('Output')
        plt.show()