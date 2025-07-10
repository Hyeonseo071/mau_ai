from picamera2 import Picamera2
import cv2
import numpy as np

# 카메라 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# CLAHE 객체 생성 (밝기 대비 향상)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    frame = picam2.capture_array()
    bgr = frame.copy()

    # grayscale + 대비 향상
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # 블러 처리로 노이즈 제거
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 원 검출 (작은 반지름도 탐지 가능하게 수정)
    circles = cv2.HoughCircles(gray_blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=30,
                               param1=100,
                               param2=25,        # 민감도 상승
                               minRadius=2,      # 더 작은 원도 인식
                               maxRadius=100)

    # 색상 공간 HSV로 변환
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 흰색 마스크 (배경 대비용)
    lower_white = np.array([0, 0, 240])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 빨간색 범위 정의 (양 끝 Hue 보정)
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = white_mask.shape

        for c in circles[0, :]:
            x, y, r = c
            center = (x, y)

            if 0 <= x < width and 0 <= y < height:
                # 중심이 흰 배경에 있는지 확인
                if white_mask[y, x] == 255:
                    ring_mask = np.zeros_like(red_mask)
                    cv2.circle(ring_mask, center, r, 255, thickness=7)  # 두께 증가

                    red_ring_overlap = cv2.bitwise_and(red_mask, ring_mask)
                    red_pixels = cv2.countNonZero(red_ring_overlap)

                    if red_pixels > 20:
                        label = "laser (near)" if r > 10 else "laser (far)"
                        cv2.circle(frame, center, r, (0, 255, 0), 2)
                        cv2.putText(frame, label, (x + 10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print(f"[!] 원 위치 화면 벗어남: x={x}, y={y}, 화면=({width}, {height})")

    cv2.imshow('laser detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows() #1
