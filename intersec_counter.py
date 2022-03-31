import sys
import numpy as np
import cv2 as cv
import math
import pandas as pd
import imutils

def SUSAN_mask():
    mask = np.ones((7,7))
    mask[0,0] = 0
    mask[0,1] = 0
    mask[0,5] = 0
    mask[0,6] = 0
    mask[1,0] = 0
    mask[1,6] = 0
    mask[5,0] = 0
    mask[5,6] = 0
    mask[6,0] = 0
    mask[6,1] = 0
    mask[6,5] = 0
    mask[6,6] = 0
    return mask

def SUSAN(img):
    threshold_val = 37/2
    img = img.astype(np.float64)
    circle = SUSAN_mask()
    sus = np.zeros(img.shape)

    for i in range(3, img.shape[0] - 3):
        for j in range(3, img.shape[1] - 3):
            im = np.array(img[i - 3:i + 4, j - 3:j + 4])
            im =  im[circle == 1]
            im0 = img[i, j]
            val = np.sum(np.exp(-((im - im0) / 10)**6))
            if val <= threshold_val:
                val = threshold_val - val
            else:
                val = 0
            sus[i,j] = val
    return sus


img = cv.imread("graph_1.png", 0)
vertexAmount = int(input())
sus = SUSAN(img)
dist = np.zeros((sus.shape[0], sus.shape[1]))

for i in range(0, sus.shape[0]):
    for j in range(0, sus.shape[1]):
        if sus[i, j] > 0:
            dist[i, j] = 0
        else:
            r = 0
            c = 0
            while r < 10 and c == 0:
                r = r + 1
                for m in range(max(i-r, 0), min(i+r+1, sus.shape[0])):
                    for n in range(max(j-r, 0), min(j+r+1, sus.shape[1])):
                        if sus[m, n] > 0:
                            c = c + 1
            dist[i, j] = r

output = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
output[dist <= 8] = [0, 0, 0]
output[dist > 8] = [255, 255, 255]

grayOutput = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
edges = cv.Canny(blurredOutput, 50, 130)

contours = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
grContours = imutils.grab_contours(contours)
total = len(grContours)
        
print ("Number of edge intersections:", total - vertexAmount)

