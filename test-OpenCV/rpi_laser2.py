from picamera2 import Picamera2
import cv2
import numpy as np

# 카메라 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    # 프레임 캡처
    frame = picam2.capture_array()
    bgr = frame.copy()

    # 색상 변환
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 밝고 진한 빨간색 HSV 범위 설정
    lower_red1 = np.array([0, 180, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 180, 200])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    laser_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 마스크 정제 (노이즈 제거 + 부드럽게)
    kernel = np.ones((3, 3), np.uint8)
    laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_OPEN, kernel)
    laser_mask = cv2.GaussianBlur(laser_mask, (5, 5), 0)

    # 컨투어 찾기
    contours, _ = cv2.findContours(laser_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 300:  # 작고 강한 점만 필터링
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)

            # 중심 밝기 확인 (V값 또는 gray 값 사용 가능)
            if gray[y, x] > 200:  # or: hsv[y, x][2] > 200
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.putText(frame, "laser", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 결과 출력
    cv2.imshow('laser detection', frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cv2.destroyAllWindows()
