import cv2
import numpy as np
import face_recognition

imgSiddhi = face_recognition.load_image_file('ImagesBasic/Siddhi Manjrekar.jpg')
imgSiddhi = cv2.cvtColor(imgSiddhi,cv2.COLOR_BGR2RGB)
imgGayatri = face_recognition.load_image_file('ImagesBasic/Gayatri Patkar.jpg')
imgGayatri = cv2.cvtColor(imgGayatri,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSiddhi)[0]
encodeSiddhi = face_recognition.face_encodings(imgSiddhi)[0]
cv2.rectangle(imgSiddhi,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgGayatri)[0]
encodeGayatri = face_recognition.face_encodings(imgGayatri)[0]
cv2.rectangle(imgGayatri,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeSiddhi],encodeGayatri)
faceDis = face_recognition.face_distance([encodeSiddhi],encodeGayatri)
print(results,faceDis)
cv2.putText(imgGayatri,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Siddhi Manjrekar',imgSiddhi)
cv2.imshow('Gayatri Patkar',imgGayatri)
cv2.waitKey(0)
