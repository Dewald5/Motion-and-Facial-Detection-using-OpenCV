# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:35:03 2020

@author: Dewald
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:46:57 2020

@author: Dewald
"""


import os
import numpy as np
import cv2

 
sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX
 
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
 
_, frame1 = cap.read()
_, frame2 = cap.read()
 
facecount = 0
count = 0


while(True):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)
    dist = distMap(frame1, frame3)
    frame1 = frame2
    frame2 = frame3

    mod = cv2.GaussianBlur(dist, (9,9), 0)
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    _, stDev = cv2.meanStdDev(mod)
    cv2.imshow('dist', mod)
    
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),(0,255,0), 2)
    if stDev > sdThresh:
            count += 1
            print(count)
            if count >= 100:
                file ="IphoneAlarm.mp3"
                os.startfile(file)
                count = 0;
                

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
      
cap.release()
cv2.destroyAllWindows()
