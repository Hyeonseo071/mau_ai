from picamera2 import Picamera2
import cv2
import numpy as np

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(gray_blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=30,
                               param1=100,
                               param2=30,
                               minRadius=5,
                               maxRadius=100)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

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
        height, width = white_mask.shape

        for c in circles[0, :]:
            x, y, r = c
            center = (x, y)

            if 0 <= x < width and 0 <= y < height:
                if white_mask[y, x] == 255:
                    ring_mask = np.zeros_like(red_mask)
                    cv2.circle(ring_mask, center, r, 255, 5)
                    red_ring_overlap = cv2.bitwise_and(red_mask, ring_mask)
                    red_pixels = cv2.countNonZero(red_ring_overlap)

                    if red_pixels > 20:
                        cv2.circle(frame, center, r, (0, 255, 0), 2)
                        cv2.putText(frame, "laser", (x + 10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print(f"[!] 경계 밖 좌표 무시: x={x}, y={y}, 범위=({width}, {height})")

    cv2.imshow('laser detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
