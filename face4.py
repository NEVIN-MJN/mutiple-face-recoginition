import face_recognition
import cv2
import numpy as np
import os

# CReating a list
path = 'Training_images'                    #folder name
images = []                                 #array
classNames = []                             #array
myList = os.listdir(path)                   #listdir take the filenmaesand returns alist
#print(myList)

#for taking the names without.png
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    #print(curImg)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print("haiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
#for i in images:
#    print(i)
def findEncodings(images):
    count = 1
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  #converting to bgr to rgb
        encode = face_recognition.face_encodings(img)[0]            #creating encode
        encodeList.append(encode)                                   #appening to encodelist
    return encodeList
encodeListKnown = findEncodings(images)
#print(encodeListKnown)
count = 1
cap = cv2.VideoCapture("videos/test 21.mp4")    #
while True:
    ret, img = cap.read() 
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                    #converting imagefrm bgr to rgb

    facesCurFrame = face_recognition.face_locations(imgS)                  #finding the face locations in the frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace ,faceLoc in zip(encodesCurFrame, facesCurFrame):#
        matches = face_recognition.compare_faces(encodeFace, encodeListKnown)  #matching webcam image with encode mage
        print (matches)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4              #resizing the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)       #rectangle around the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)    # label baground
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #addinglabel           
            #markAttendance(name)
        else:
            name = "unknown"
            print(name)
            y1, x2, y2, x1 = faceLoc
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1 ), (x2, y2 ), (0, 255, 0), 2)           #rectangle around the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)        # label baground
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)   # label baground
           

    cv2.imshow('Webcam', img)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
