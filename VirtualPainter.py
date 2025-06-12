import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#################################
brushThickness = 15
eraserThickness = 50
#################################

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = [cv2.imread(f'{folderPath}/{img}') for img in myList]
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        continue

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        fingers = detector.fingersUp()

        # Selection Mode: Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 126:
                if 100 < x1 < 220:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 250 < x1 < 350:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 380 < x1 < 500:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 520 < x1 < 640:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode: Index finger up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        # Clear Canvas: All fingers up
        if all(fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Combine canvas with webcam feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Add header image
    img[0:126, 0:640] = header

    cv2.imshow("Virtual Painter", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
