import cv2
import numpy as np

gray_img = cv2.imread('D://pythonProject5/12345.jpeg',0)

facsCascade = cv2.CascadeClassifier(r'C:\Users\97434\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

faces = facsCascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5, minSize=(32, 32))

eyeCascade = cv2.CascadeClassifier(r'C:\Users\97434\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\data\haarcascade_eye.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
i = 1
for (x,y,w,h) in faces:
    fac_gray = gray_img[y:(y+h),x:(x+w)]
    gray_img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('face',fac_gray)
    cv2.waitKey(1000)


for (x,y,w,h) in faces:
    fac_gray = gray_img[y:(y+h),x:(x+w)]
    eyes = eyeCascade.detectMultiScale(fac_gray,1.1,5)
    for (ex, ey, ew, eh) in eyes:
        cv2.putText(gray_img,str(i),(x+ex+ew,y+ey+eh),font,1,(0,255,0),3)
        gray_img = cv2.circle(gray_img, (x+ex+ew-20,y+ey+eh-20), 10, (0, 255, 0), 3)
        i = i + 1

cv2.imshow('gray_img',gray_img)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()