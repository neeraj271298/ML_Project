#importing required library
import cv2
import numpy as np
from PIL import Image

#create a cascade classifier to detect face
faceDet = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#create a recognizer
rec = cv2.face.LBPHFaceRecognizer_create()
#load recognizer with trained data
rec.read('recognizer/training.yml')

#open camera
vid_cap = cv2.VideoCapture(0)
while True:
    _ , face = vid_cap.read()
    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    points = faceDet.detectMultiScale(gray,1.3,5)
    for x,y,w,h in points:
        #draw a rectangle on face in image
        cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0),2)
        ## predicting the id of image
        id ,config = rec.predict(gray[y:y+h , x:x+w])
        if id == 1:
            id="neeraj"
        if id == 2:
            id = "uday"
        # put label on photo
        cv2.putText(face,id,(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('complete',face)
        

    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type 'q' on the keyboard:
        break
    
# close all stuffs
vid_cap.release()
cv2.destroyAllWindows()