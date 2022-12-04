import sys
import math
import numpy as np
import cv2


def decode(scores, geometry, scoreThreshold):
    detections = []
    confidences = []

    # geometry.shape=(1, 5, 80, 80)
    # scores.shape=(1, 1, 80, 80)

    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]

        for x in range(0, width):
            score = scoresData[x]

            if(score < scoreThreshold):
                continue

            # feature map은 320x320 블롭의 1/4 크기이므로, 다시 4배 확대
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # (offsetX, offsetY) 위치에서 회전된 사각형 정보 추출
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # 회전된 사각형의 한쪽 모서리 점 좌표 계산
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                       offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # 회전된 사각형의 대각선에 위치한 두 모서리 점 좌표 계산
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = ((p1[0]+p3[0])/2, (p1[1]+p3[1])/2)

            detections.append((center, (w, h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]


# 모델 & 설정 파일
model = 'EAST/frozen_east_text_detection.pb'
confThreshold = 0.5
nmsThreshold = 0.4

# 테스트 이미지 파일
img_files = ['road_closed.jpg', 'patient.jpg', 'copy_center.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 출력 레이어 이름 받아오기
'''
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)
'''

# 실행

for f in img_files:
    img = cv2.imread(f)

    if img is None:
        continue

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(img, 1, (320, 320), (123.68, 116.78, 103.94), True)
    net.setInput(blob)
    scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    # scores.shape=(1, 1, 80, 80)
    # geometry.shape=(1, 5, 80, 80)

    # score가 confThreshold보다 큰 RBOX 정보를 RotatedRect 형식으로 변환하여 반환
    [boxes, confidences] = decode(scores, geometry, confThreshold)

    # 회전된 사각형에 대한 비최대 억제
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

    rw = img.shape[1] / 320
    rh = img.shape[0] / 320

    for i in indices:
        # 회전된 사각형의 네 모서리 점 좌표 계산 & 표시
        vertices = cv2.boxPoints(boxes[i[0]])

        for j in range(4):
            vertices[j][0] *= rw
            vertices[j][1] *= rh

        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
