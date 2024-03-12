import cv2
import numpy as np
import face_recognition

imgSP = face_recognition.load_image('ImagesBasic/SunadarPichai.jpg')
imgSP = cv2.cvtColor(imgSP,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/SPTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_location(imgSP)[0]
encodeSP = face_recognition.face_encodings(imgSP)[0]
cv2.rectangle(imgSP,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_location(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeSP],encodeTest)
print(results)

cv2.imshow('SundarPichai',imgSP)
cv2.imshow('SPTest',imgTest)
cv2.waitKey(0)