import sys
import numpy as np
import cv2


face_name = ['conan', 'obama', 'sunkyoo']


def face_recognition(recognition_net, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blob = cv2.dnn.blobFromImage(gray, 1 / 255., (150, 200))
    recognition_net.setInput(blob)
    prob = recognition_net.forward()  # prob.shape=(1, 3)

    _, confidence, _, maxLoc = cv2.minMaxLoc(prob)
    face_idx = maxLoc[0]

    return face_idx, confidence


detection_net = cv2.dnn.readNet('opencv_face_detector/opencv_face_detector_uint8.pb',
                                'opencv_face_detector/opencv_face_detector.pbtxt')

if detection_net.empty():
    print('Detection Net open failed!')
    sys.exit()

recognition_net = cv2.dnn.readNet('face_rec/face_rec.pb')

if detection_net.empty():
    print('Recognition Net open failed!')
    sys.exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    detection_net.setInput(blob)
    detect = detection_net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        crop = frame[y1:y2, x1:x2]
        face_idx, confidence = face_recognition(recognition_net, crop)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = '{0}: {1:0.3f}'.format(face_name[face_idx], confidence)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
