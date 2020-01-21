#! /usr/bin/env python3

# Authors: Sajarin Dider and Daniel Mallia

# Use of EAST detector inspired by and some code used from: 
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# WE TAKE NO CREDIT FOR THE CODE ON LINES 34 - 117

# NOTE: This script uses the TensorFlow Implementation of the EAST text 
# detector, which is offered by OpenCV. 
# The actual Github for this implementation is here: 
# https://github.com/argman/EAST
# The download link for the .pb file (trained model) is here: 
# https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

import cv2 as cv
import numpy as np 
import copy
from random import seed
from random import random

seed()

def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

def returnWordBBoxes(image):

    # Need to scale image to a multiple of 32 - just steps down to next multiple of 32 
    newRows, newCols = image.shape[:2]
    newRows -= newRows % 32
    newCols -= newCols % 32
    output = cv.resize(image, (newRows, newCols), interpolation = cv.INTER_LINEAR)

    # Import pretrained EAST detector
    east = cv.dnn.readNet('frozen_east_text_detection.pb')

    # Required layer names
    layerNames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    blob = cv.dnn.blobFromImage(output)

    east.setInput(blob)
    (scores, geometry) = east.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append([int(startX), int(startY), int(w), int(h)]) # MAY NEED TO SUBSTITUTE WIDTH AND HEIGHT
            confidences.append(float(scoresData[x]))

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    
    #### USE OPENCV NMS
    indices = cv.dnn.NMSBoxes(rects, confidences, 0.5, 0.3)

    lines = []
    # ensure at least one detection exists
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            # extract the bounding box coordinates
            (x, y) = (rects[i][0], rects[i][1])
            (w, h) = (rects[i][2], rects[i][3])

            bottomLeft = y + h
            newLine = True

            for line in lines:
                if(bottomLeft > .97 * line[0] and bottomLeft < 1.03 * line[0]):
                    line.append((x, y, w, h))
                    newLine = False
                    break

            if(newLine):
                lines.append([bottomLeft, (x, y, w, h)])
                    

            # # draw a bounding box rectangle and label on the image
            # cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # text = str(confidences[i])
            # cv.putText(output, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
            #     0.5, (255, 0, 0), 2)

    
    for line in lines:
        color = (int(random() * 255), int(random() * 255), int(random() * 255))
        for box in line[1:]:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv.rectangle(output, (x, y), (x + w, y + h), color, 2)

    #show the output image
    cv.namedWindow('Text Detection', cv.WINDOW_NORMAL)
    cv.resizeWindow('Text Detection', 800, 600)
    cv.imshow('Text Detection', output)
    cv.waitKey(0)
    #cv.imwrite('text.jpg', output)

if __name__ == "__main__":

    image = cv.imread('testTransformed.jpg', cv.IMREAD_COLOR)
    returnWordBBoxes(image)
    # bndingBx = []#holds bounding box of each countour
    # corners = []
    # image = cv.imread('e.png',0)
    # blur = cv.GaussianBlur(image,(5,5),0)
    # threshold, threshImage = cv.threshold(blur,0,255,
    #     cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
     
    # contours, heirar = cv.findContours(threshImage, cv.RETR_CCOMP, 
    #     cv.CHAIN_APPROX_SIMPLE)
    # for num in range(0,len(contours)):
    #     #make sure contour is for letter and not cavity
    #     if(heirar[0][num][3] == -1):
    #         left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
    #         right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
    #         top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
    #         bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
    #         bndingBx.append([top,right,bottom,left])
    
    #     for bx in bndingBx:
    #         corners.append(findCorners(bx))
    
    #     #draw the countours on threshImage image
    #     x,y,w,h = cv.boundingRect(threshImage)

    # cv.imshow('thresh', threshImage)
    # cv.waitKey()


