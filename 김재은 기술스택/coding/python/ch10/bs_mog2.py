import sys
import numpy as np
import cv2


# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 차분 알고리즘 객체 생성
bs = cv2.createBackgroundSubtractorMOG2()
#bs = cv2.createBackgroundSubtractorKNN()
#bs.setDetectShadows(False)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = bs.apply(gray)
    back = bs.getBackgroundImage()

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 80:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('back', back)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
