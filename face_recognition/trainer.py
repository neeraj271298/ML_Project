#importing required library
import os
import cv2
from PIL import Image
import numpy as np

# create a recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
#path of dataset
path = 'dataSet'

#define a function that return a array of images with id
def getImage(path):
    #store all image path 
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    #iterate loop for each image
    for imagePath in imagePaths:
        #open image with PILLOW in gray Scale
        faceImg = Image.open(imagePath).convert('L')
        #convert it to numpy array because opencv works on numpy array
        faceNp = np.array(faceImg)
        # get id of person
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        print(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return faces,np.array(IDs)

# get data 
faces,IDs = getImage(path)
# train our recognizer 
recognizer.train(faces,np.array(IDs))
# save the training data
recognizer.save('recognizer/training.yml')
#close all windows
cv2.destroyAllWindows()