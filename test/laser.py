import cv2 #이미지 처리
import numpy as np #수치계산

# #카메라 초기화
# cap = cv2.VideoCapture(0)

laser1_img = cv2.imread("../images/2.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(laser1_img, (5, 5), 0)

# 임계값으로 밝은 점만 남기기 (thresholding)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# 컨투어 찾기
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 원본 컬러 이미지도 불러와서 표시용으로 사용
laser1_color = cv2.imread("../images/2.jpg")

# 레이저 점 찾기 (작고 밝은 영역만)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 1 < area < 1000:  
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(laser1_color, center, int(radius)+5, (0, 255, 0), 2)
        cv2.putText(laser1_color, "Laser", (center[0]+10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# 결과 이미지 출력
cv2.imshow("Detected Laser", laser1_color)

cv2.waitKey(0)
cv2.destroyAllWindows()