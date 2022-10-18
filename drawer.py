import os
import cv2
import HandTrackingModule as htm
import numpy as np


brushThickness = 10
eraserThickness = 60
xp,yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

overlayList = []
folderPath = "paint_dashboard"
myList = os.listdir(folderPath)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')

    overlayList.append(image)
header = overlayList[0]
drawColor = (180, 105, 255)

cv2.namedWindow("Canvas")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.75)
while True:
    #1:import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2: find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    #3: check which finger is up
    if len(lmList) != 0:
        #print(lmList)
        x1, y1 = lmList[8][1:]  # tip of index fingers
        x2, y2 = lmList[12][1:]  # tip of middle fingers
        fingers = detector.fingersUp()
        # print(fingers)

    #4: if selection mode: two finger are up!
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

            # checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (180, 105, 255)

                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (0, 255, 255)

                if 850 < x1 < 1050:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                if 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)


    # phase 5: if drawing mode: index finger is up!
        if fingers[1] and not fingers[2]:

            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.circle(img, (x1, y1), 15, (255,255,255), cv2.FILLED)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # phase 6:setting the header image
    img[0:126, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    # If window is closed by user then finish program execution
    if cv2.getWindowProperty("Canvas", cv2.WND_PROP_VISIBLE) == 0:
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow("Canvas", img)
    cv2.waitKey(1)
