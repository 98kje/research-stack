import sys
import numpy as np
import cv2


# 기준 영상 불러오기
src = cv2.imread('korea.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 카메라 장치 열기
cap1 = cv2.VideoCapture(0)

if not cap1.isOpened():
    print('Camera open failed!')
    sys.exit()

# 필요할 경우 카메라 해상도 변경
#cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# 카메라 프레임 화면에 출력할 동영상 파일 열기
cap2 = cv2.VideoCapture('korea.mp4')

if not cap2.isOpened():
    print('Video load failed!')
    sys.exit()

# AKAZE 특징점 알고리즘 객체 생성
detector = cv2.AKAZE_create()

# 기준 영상에서 특징점 검출 및 기술자 생성
kp1, desc1 = detector.detectAndCompute(src, None)

# 해밍 거리를 사용하는 매칭 객체 생성
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    # 매 프레임마다 특징점 검출 및 기술자 생성
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = detector.detectAndCompute(gray, None)

    # 특징점이 100개 이상 검출될 경우 매칭 수행
    if len(kp2) > 100:
        matches = matcher.match(desc1, desc2)

        # 좋은 매칭 선별
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:80]

        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

        # 호모그래피 계산
        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        inlier_cnt = cv2.countNonZero(inliers)

        # RANSAC 방법에서 정상적으로 매칭된 것의 개수가 20개 이상이면
        if inlier_cnt > 20:
            ret2, frame2 = cap2.read()

            if not ret2:
                break

            h, w = frame1.shape[:2]

            # 비디오 프레임을 투시 변환
            video_warp = cv2.warpPerspective(frame2, H, (w, h))

            white = np.full(frame2.shape[:2], 255, np.uint8)
            white = cv2.warpPerspective(white, H, (w, h))

            # 비디오 프레임을 카메라 프레임에 합성
            cv2.copyTo(video_warp, white, frame1)

    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
