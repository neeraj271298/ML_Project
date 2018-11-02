
# import the library
import cv2

#load cascade file
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# def function to detect face
def detect(gray,frame):
    #detect face
    faces = face.detectMultiScale(gray,1.3,5)
    roi_gray = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # consider only detected face
        roi_gray = gray[y:y+h,x:x+w]

    return frame,roi_gray

# start camera to click picture
video_cap = cv2.VideoCapture(0)
ID = int(input('enter the id  '))
tempNum = 0
while True:
    tempNum = tempNum + 1
    _ , frame = video_cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas , storeImg = detect(gray,frame)
    cv2.imshow('Video',frame)
    cv2.imwrite('dataSet/person.'+str(ID)+'.'+str(tempNum)+'.jpg',storeImg)
    if tempNum > 50: # if we clicked 50 images 
        break;
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type 'q' on the keyboard:
        break
    
# close all stuffs
video_cap.release()
cv2.destroyAllWindows()

