import cv2
import numpy as np
def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.bilateralFilter(imgGray, 9, 150, 150)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1], apertureSize=3)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgCanny)
    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    perimeter=0
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True) 
            approx = cv2.approxPolyDP(i, 0.001 * peri, True)  
            bbox = cv2.boundingRect(approx) 
            if draw:
                cv2.drawContours(img, [i], -1, (255, 0, 0), 5)
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                perimeter = peri
                cv2.putText(img, f"Perimeter: {peri:.2f} px", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i, peri])
            else:
                finalContours.append([len(approx), area, approx, bbox, i, peri])
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True) 
    return img, finalContours, perimeter