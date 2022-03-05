import cv2
import pytesseract
import numpy as np
import time

cap = cv2.VideoCapture("./videos/video01.mp4")

p_time = 0

while True:
    
    success, img = cap.read()

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {str(round(fps))}', (50, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(100)


    