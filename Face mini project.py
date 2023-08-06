import cv2
import numpy as np
import face_recognition
import os  
from datetime import datetime
import dlib


path = 'C:/Users/vikas/OneDrive/Desktop/project/images'
images = []
personNames = []
myList = os.listdir(path)  
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#print(faceEncodings(images))
def attendance(name):
    with open('C:/Users/vikas/OneDrive/Desktop/project/data.csv', 'r+') as f:
    # r+ ka matlab read and append dono kr sakte hain.
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) 
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (79, 255, 160), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (160, 255, 40), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (29, 255, 255), 2)
            attendance(name)
        else:
            print("Name is not matched")

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:  #ascii code of enter key
        break
video=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()

while True:
    ret,frame=video.read()
    frame=cv2.flip(frame,1)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)

    num=0
    for face in faces:
        x,y=face.left(),face.top()
        hi,wi=face.right(),face.bottom()
        cv2.rectangle(frame,(x,y),(hi,wi),(0,0,255),2)
        num=num+1

        cv2.putText(frame,'face'+str(num),(x-12,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow('faces',frame)
    if cv2.waitKey(13)==ord('q'):
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

