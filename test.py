import cv2
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "/Users/parthsharma/Aerial Drawing/Header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    print(len(lmList))



    # Setting Header image
    img[0:125, 0:1280] = header
    cv2.imshow("image", img)
    cv2.waitKey(1)

