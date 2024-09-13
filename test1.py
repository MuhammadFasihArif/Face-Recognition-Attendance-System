import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'data/images/train'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) == 0:
            continue  # Skip this image if no faces are detected
        encode = face_recognition.face_encodings(img, face_locations)[0]
        encodingList.append(encode)
    return encodingList


def markAttendance(name):
    with open('attendance.csv','r+')as f:
        #f.write('Name, Time\n')
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodingListKnown = findEncodings(images)
print('Encoding complete')

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if success!=True:
        break
    else:
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurface = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceloc in zip(encodeCurface, faceCurFrame):
            print(encodeFace.shape)
            matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
            faceDist = face_recognition.face_distance(encodingListKnown, encodeFace)
            #print(faceDist)

            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                #print(name)
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.rectangle(img, (x1,y2-35),(x2,y2), (0, 255, 0),cv.FILLED)
                cv.putText(img, name, (x1+6, y2-6), cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
                markAttendance(name)
                


    cv.imshow('webcam', img)
    c = cv.waitKey(1)
    if c == ord('q'):
        break

