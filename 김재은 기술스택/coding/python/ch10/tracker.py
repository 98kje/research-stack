import sys
import numpy as np
import cv2


# 동영상 열기
cap = cv2.VideoCapture('tracking1.mp4')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 트래커 객체 생성

# Kernelized Correlation Filters
#tracker = cv2.TrackerKCF_create()

# Minimum Output Sum of Squared Error
#tracker = cv2.TrackerMOSSE_create()

# Discriminative Correlation Filter with Channel and Spatial Reliability
tracker = cv2.TrackerCSRT_create()

# 첫 번째 프레임에서 추적 ROI 설정
ret, frame = cap.read()

if not ret:
    print('Frame read failed!')
    sys.exit()

rc = cv2.selectROI('frame', frame)
tracker.init(frame, rc)

# 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        print('Frame read failed!')
        sys.exit()

    # 추적 & ROI 사각형 업데이트
    ret, rc = tracker.update(frame)
    rc = tuple([int(_) for _ in rc])
    cv2.rectangle(frame, rc, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()
