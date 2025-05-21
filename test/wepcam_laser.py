import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    circle = cv2.GaussianBlur()

    #색상범위설정(red)
    lower_red1 = np.array([0, 150, 150])    #테두리
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([170, 150, 150])

    
    upper_red2 = np.array([179, 255, 255]) #중앙 흰점
    lower_white = np.array([0, 0, 240]) 
    upper_white = np.array([180, 30, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_white, upper_white)
    mask4 = cv2.bitwise_or(mask1, mask2)

    mask = cv2.bitwise_or(mask3,mask4)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#houghcircles (허프원 변환 알고리즘)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 300:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
            if circularity > 0.7:  # 원형에 가까운 것만
                (x,y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius)+5, (0,255,0), 2)
                cv2.putText(frame, "laser", (center[0]+10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            
    cv2.imshow('laser detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()