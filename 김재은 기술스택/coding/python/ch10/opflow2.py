import sys
import numpy as np
import cv2


# 카메라 장치 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

# 설정 변수 정의
MAX_COUNT = 50
needToInit = False
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 128)]

ptSrc = None
ptDst = None

# 카메라 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if needToInit:
        ptSrc = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
        needToInit = False

    if ptSrc is not None:
        if prev is None:
            prev = gray.copy()

        ptDst, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, ptSrc, None)

        for i in range(ptDst.shape[0]):
            if status[i, 0] == 0:
                continue

            cv2.circle(img, tuple(ptDst[i, 0].astype(int)), 4, colors[i % 8], 2, cv2.LINE_AA)

    cv2.imshow('frame', img)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('r'):
        needToInit = not needToInit
    elif key == ord('c'):
        ptSrc = None
        ptDst = None

    ptDst, ptSrc = ptSrc, ptDst
    prev = gray

cv2.destroyAllWindows()
