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

import cv as cv
import numpy as np 
import copy

def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

def returnWordBBoxes(image):
    output = copy.deepcopy(image)

    # Need to scale image to a multiple of 32 - just steps down to next multiple of 32 
    newRows, newCols = output.shape[:2]
    newRows -= newRows % 32
    newCols -= newCols % 32
    output = cv.resize(image, (newRows , newCols), interpolation = cv.INTER_LINEAR)

    # Import pretrained EAST detector
    east = cv.dnn.readNet('frozen_east_text_detection.pb')

    # Required layer names
    layerNames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    blob = cv.dnn.blobFromImage(output)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

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
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    
    #### USE OPENCV NMS
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)

if __name__ == "__main__":

    bndingBx = []#holds bounding box of each countour
    corners = []
    image = cv.imread('e.png',0)
    blur = cv.GaussianBlur(image,(5,5),0)
    threshold, threshImage = cv.threshold(blur,0,255,
        cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
     
    contours, heirar = cv.findContours(threshImage, cv.RETR_CCOMP, 
        cv.CHAIN_APPROX_SIMPLE)
    for num in range(0,len(contours)):
        #make sure contour is for letter and not cavity
        if(heirar[0][num][3] == -1):
            left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
            right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
            top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
            bndingBx.append([top,right,bottom,left])
    
        for bx in bndingBx:
            corners.append(findCorners(bx))
    
        #draw the countours on threshImage image
        x,y,w,h = cv.boundingRect(threshImage)

    cv.imshow('thresh', threshImage)
    cv.waitKey()


