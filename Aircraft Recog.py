import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

aircraft_cascade = cv.CascadeClassifier('Aircraft2.xml')
heli_cascade = cv.CascadeClassifier('helikopter_1.xml')
img = cv.imread('z.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

planes = aircraft_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in planes:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    heli = heli_cascade.detectMultiScale(roi_gray)
    #draw bounding boxes around detected features
    for (ex,ey,ew,eh) in heli:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#plot the image
plt.imshow(img)