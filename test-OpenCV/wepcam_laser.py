import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 허프 원 변환
    circles = cv2.HoughCircles(gray_blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=30,
                               param1=100,
                               param2=30,
                               minRadius=5,
                               maxRadius=100)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_white = np.array([0, 0, 240])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c
            center = (x, y)

            # 중심이 흰색인지 확인
            if white_mask[y, x] == 255:
                # 테두리에 빨간색이 있는지 확인 (원 경계 주변만 검사)
                ring_mask = np.zeros_like(red_mask)
                cv2.circle(ring_mask, center, r, 255, 5)  # 테두리만

                red_ring_overlap = cv2.bitwise_and(red_mask, ring_mask)
                red_pixels = cv2.countNonZero(red_ring_overlap)

                if red_pixels > 20:  # 일정 수 이상 빨간 픽셀 있을 경우
                    cv2.circle(frame, center, r, (0, 255, 0), 2)
                    cv2.putText(frame, "laser", (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('laser detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
