import sys
import os
import glob
import numpy as np
import cv2


def norm_hangul(img):
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    dst = cv2.warpAffine(img, aff, (0, 0))
    return dst


img_files = glob.glob('tensorflow-hangul-recognition-master/image-data/hangul-images/*.jpeg')

if len(img_files) == 0:
    print('There is no jpeg file! Please check the directories & files!')
    sys.exit()

count = 0

for f in img_files:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

    if img is None:
        print('Image load failed:', f)
        continue

    img = norm_hangul(img)
    ret = cv2.imwrite(f, img)

    if not ret:
        print('Image write failed:', f)
        continue

    count += 1
    if count % 1000 == 0:
        print('{} images normalized...'.format(count))

print('Total {} images are normalized...'.format(count))
