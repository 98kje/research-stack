import sys
import numpy as np
import cv2


oldx, oldy = -1, -1


def on_mouse(event, x, y, flags, _):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        oldx, oldy = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y),
                     (255, 255, 255), 24, cv2.LINE_AA)
            oldx, oldy = x, y
            cv2.imshow('img', img)


def norm_hangul(img):
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    dst = cv2.warpAffine(img, aff, (0, 0))
    return dst


# 네트워크 불러오기
net = cv2.dnn.readNet('tensorflow-hangul-recognition-master/korean_recognition.pb')

if net.empty():
    print('Network load failed!')
    sys.exit()

# 한글 파일 불러오기
classNames = None
with open('tensorflow-hangul-recognition-master/labels/256-common-hangul.txt', 'rt', encoding='utf-8') as f:
    classNames = f.read().rstrip('\n').split('\n')

# 마우스로 한글을 입력할 새 영상
img = np.zeros((400, 400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)

while True:
    c = cv2.waitKey()

    if c == 27:
        break
    elif c == ord(' '):
        img = norm_hangul(img)
        blob = cv2.dnn.blobFromImage(img, 1, (64, 64))
        net.setInput(blob)
        out = net.forward()  # out.shape=(1, 256)

        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        print(f'{classNames[classId]} ({confidence * 100:4.2f}%)')

        img.fill(0)
        cv2.imshow('img', img)

cv2.destroyAllWindows()
