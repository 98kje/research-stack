import sys
import numpy as np
import cv2


# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 차영상 구하기 & 이진화
    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
