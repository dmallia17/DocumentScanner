# Authors: Sajarin Dider and Daniel Mallia

# Use of EAST detector inspired by: 
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

import cv2
import numpy as np 

def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

def returnWordBBoxes(image):


if __name__ == "__main__":

    bndingBx = []#holds bounding box of each countour
    corners = []
    image = cv2.imread('e.png',0)
    blur = cv2.GaussianBlur(image,(5,5),0)
    threshold, threshImage = cv2.threshold(blur,0,255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
     
    contours, heirar = cv2.findContours(threshImage, cv2.RETR_CCOMP, 
        cv2.CHAIN_APPROX_SIMPLE)
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
        x,y,w,h = cv2.boundingRect(threshImage)

    cv2.imshow('thresh', threshImage)
    cv2.waitKey()


