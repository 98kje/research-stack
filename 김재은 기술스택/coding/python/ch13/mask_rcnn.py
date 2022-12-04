import sys
import numpy as np
import cv2


def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classes[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)


# 모델 & 설정 파일
model = 'mask_rcnn/frozen_inference_graph.pb'
config = 'mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
class_labels = 'mask_rcnn/coco_90.names'
confThreshold = 0.6
maskThreshold = 0.3

# 테스트 이미지 파일
img_files = ['dog.jpg', 'traffic.jpg', 'sheep.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 전체 레이어 이름 받아오기
'''
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
for name in layer_names:
    print(name)
'''

# 실행

for f in img_files:
    img = cv2.imread(f)

    if img is None:
        continue

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # boxes.shape=(1, 1, 100, 7)
    # masks.shape=(100, 90, 15, 15)

    h, w = img.shape[:2]
    numClasses = masks.shape[1]  # 90
    numDetections = boxes.shape[2]  # 100

    boxesToDraw = []
    for i in range(numDetections):
        box = boxes[0, 0, i]  # box.shape=(7,)
        mask = masks[i]  # mask.shape=(90, 15, 15)
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            #print(classId, classes[classId], score)

            x1 = int(w * box[3])
            y1 = int(h * box[4])
            x2 = int(w * box[5])
            y2 = int(h * box[6])

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
            classMask = mask[classId]

            # 객체별 15x15 마스크를 바운딩 박스 크기로 resize한 후, 불투명 컬러로 표시
            classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
            mask = (classMask > maskThreshold)

            roi = img[y1:y2+1, x1:x2+1][mask]
            img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

    # 객체별 바운딩 박스 그리기 & 클래스 이름 표시
    for box in boxesToDraw:
        drawBox(*box)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
