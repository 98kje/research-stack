import sys
import numpy as np
import cv2


# 모델 & 설정 파일
model = 'openpose/pose_iter_440000.caffemodel'
config = 'openpose/pose_deploy_linevec.prototxt'

# 포즈 점 개수, 점 연결 개수, 연결 점 번호 쌍
nparts = 18
npairs = 17
pose_pairs = [(1, 2), (2, 3), (3, 4),  # 왼팔
              (1, 5), (5, 6), (6, 7),  # 오른팔
              (1, 8), (8, 9), (9, 10),  # 왼쪽다리
              (1, 11), (11, 12), (12, 13),  # 오른쪽다리
              (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # 얼굴

# 테스트 이미지 파일
img_files = ['pose1.jpg', 'pose2.jpg', 'pose3.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

for f in img_files:
    img = cv2.imread(f)

    if img is None:
        continue

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(img, 1/255., (368, 368))
    net.setInput(blob)
    out = net.forward()  # out.shape=(1, 57, 46, 46)

    h, w = img.shape[:2]

    # 검출된 점 추출
    points = []
    for i in range(nparts):
        heatMap = out[0, i, :, :]

        '''
        heatImg = cv2.normalize(heatMap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatImg = cv2.resize(heatImg, (w, h))
        heatImg = cv2.cvtColor(heatImg, cv2.COLOR_GRAY2BGR)
        heatImg = cv2.addWeighted(img, 0.5, heatImg, 0.5, 0)
        cv2.imshow('heatImg', heatImg)
        cv2.waitKey()
        '''

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int(w * point[0] / out.shape[3])
        y = int(h * point[1] / out.shape[2])

        points.append((x, y) if conf > 0.1 else None)  # heat map threshold=0.1

    # 검출 결과 영상 만들기
    for pair in pose_pairs:
        p1 = points[pair[0]]
        p2 = points[pair[1]]

        if p1 is None or p2 is None:
            continue

        cv2.line(img, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 4, (0, 0, 255), -1, cv2.LINE_AA)

    # 추론 시간 출력
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
