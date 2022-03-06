import cv2
import easyocr
import numpy as np
import time

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

license_plate_directory = ['GKSB78']


reader = easyocr.Reader(['es'], gpu=True)

p_time = 0
license_plate_number = None

while True:
    
    success, img = cap.read()
    
    # Image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_plate = cascade.detectMultiScale(gray_img, 1.1, 5)

    if len(number_plate) != 0:

        # Make a rectangle where a license plate is
        x, y, w, h = number_plate[0, 0], number_plate[0, 1], number_plate[0, 2], number_plate[0, 3]
        
        # Show bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Crop license plate
        cropped_img = img[y: y+h, x: x + w]
        
        # Make a kernel, dilate and erode
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.erode(cropped_img, kernel, iterations=10)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Get a binary image
        (thresh, plate_binary) = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)

        # Read the text on the license plate
        lpn = reader.readtext(plate_binary)
        # cv2.imshow("License Plate", plate_binary)

        if len(lpn) != 0:
            _, _, confidence = lpn[0]

            if confidence > 0.7:
                cv2.imshow("License Plate", plate_binary)
                _, license_plate_number, _ = lpn[0]
                license_plate_number = ''.join(e for e in license_plate_number if e.isalnum())

                if license_plate_number in license_plate_directory:
                    cv2.putText(img, f'RECOGNIZED: {str(license_plate_number)}',
                                (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    cv2.putText(img, f'FPS:{str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


    