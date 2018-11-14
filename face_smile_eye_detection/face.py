
# import the library
import cv2

#load cascade file
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')

# def function to detect face
def detect(gray,frame):
    #detect face
    faces = face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # consider only detected face
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        # detect eye in face
        eyes = eye.detectMultiScale(roi_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        # detect wheather a person is smiles or not 
        smiles = smile.detectMultiScale(roi_gray,1.7,22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            
    return frame

# video 
video_cap = cv2.VideoCapture(0)
while True:
    _ , frame = video_cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break
    
# close all stuffs
video_cap.release()
cv2.destroyAllWindows()


        
