import sys
import numpy as np
import cv2


def load_digits():
    img_digits = []

    for i in range(10):
        filename = './digits/digit{}.bmp'.format(i)
        img_digits.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

        if img_digits[i] is None:
            return None

    return img_digits


def find_digit(img, img_digits):
    max_idx = -1
    max_ccoeff = -1

    # 최대 NCC 찾기
    for i in range(10):
        img = cv2.resize(img, (100, 150))
        res = cv2.matchTemplate(img, img_digits[i], cv2.TM_CCOEFF_NORMED)

        if res[0, 0] > max_ccoeff:
            max_idx = i
            max_ccoeff = res[0, 0]

    return max_idx


def main():
    # 입력 영상 불러오기
    src = cv2.imread('digits_print.bmp')

    if src is None:
        print('Image load failed!')
        return

    # 100x150 숫자 영상 불러오기
    img_digits = load_digits()  # list of ndarray

    if img_digits is None:
        print('Digit image load failed!')
        return

    # 입력 영상 이진화 & 레이블링
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(src_bin)

    # 숫자 인식 결과 영상 생성
    dst = src.copy()
    for i in range(1, cnt):
        (x, y, w, h, s) = stats[i]

        if s < 1000:
            continue

        # 가장 유사한 숫자 이미지를 선택
        digit = find_digit(src_gray[y:y+h, x:x+w], img_digits)
        cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
        cv2.putText(dst, str(digit), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
