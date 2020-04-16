# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
#clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread(r"C:\Users\thehitmanranjan\Desktop\photo_1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
rects.sort()
print(rects)
x=rects[0][0]
y=rects[0][1]
width=rects[0][2]
height=rects[0][3]
print(x,y,width,height)
roi = im[y:y+height, x:x+width]
cv2.imwrite("roi.png", roi)
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.


cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()