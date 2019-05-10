import cv2
import os
import numpy as np 
import imutils
from skimage.filters import threshold_local
import pytesseract
from PIL import Image


def order_points(pts):
    rect = np.zeros((4,2), dtype = "float32")

    s=pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
# import the necessary packages
import numpy as np
import cv2

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped






def main():
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
    edg = cv2.Canny(dst, 75, 200)
    #Contours
    cnts= cv2.findContours(edg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    cnts=sorted(cnts,key = cv2.contourArea, reverse = True)[:5]
    #print(cnts)
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c,0.02 * peri, True)
        if len(approx)== 4:
            screenCnt = approx
            print(screenCnt)
            break
    
    cv2.drawContours(img, [screenCnt],-1,(0,255,0),3)

    #birds eye perspective
    warped = four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

    warped=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8")*225
    
     #Harris Corner Detector
#    gray= np.float32(edg)
#    gre= cv2.cornerHarris(gray,2,3,0.04)
#    gre = cv2.dilate(gre,None)
#    edg[gre>0.01*gre.max()]=[0,0,255]


    postimage = "{}.png".format(os.getpid())
    cv2.imwrite(postimage,warped)
    text = pytesseract.image_to_string(Image.open(postimage))
    print(text)
    os.remove(postimage)




#display image
    cv2.namedWindow('dst image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('org image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('edg image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Scan image', cv2.WINDOW_NORMAL)
#cv2.namedWindow('gre image', cv2.WINDOW_NORMAL)
    cv2.imshow('dst image', dst )
    cv2.imshow('org image', img )
    cv2.imshow('edg image', edg )
    cv2.imshow('Scan image', warped )
    #cv2.imshow('postit',postit)
#cv2.imshow('gre image', gre )
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    
if __name__ == "__main__":
    main()
