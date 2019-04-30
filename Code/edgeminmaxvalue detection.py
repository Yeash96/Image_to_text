import cv2
import os
import numpy as np
import imutils


imgpath = '..\\Input_image'
os.chdir(imgpath)
img = cv2.imread('Image001.jpg')
ratio = img.shape[0]/500.0
orig = img.copy()
img= imutils.resize(img, height = 500)
imgray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #noise adjustment 
dst = cv2.fastNlMeansDenoising(imgray, 10, 10, 7, 21)
    #Canny Edge Detection
#for i in range(0,3)
edg1 = cv2.Canny(dst, 10, 100)
edg2 = cv2.Canny(dst, 75, 200)
edg3 = cv2.Canny(dst, 200, 400)

    
cv2.namedWindow('org image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edg1 image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edg2 image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edg3 image', cv2.WINDOW_NORMAL)

cv2.imshow('org image', img )
cv2.imshow('edg1 image', edg1 )
cv2.imshow('edg2 image', edg2 )
cv2.imshow('edg3 image', edg3 )
