import sys
import glob
import cv2

img_files = glob.glob('.\\images\\*.jpg')

for f in img_files:
     print(f)
                        #루프함수로 파일 출력
cv2.namedWindow('image', cv2.WINDOW_NORMAL) # 전체화면띄우는 코드    
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN,cv2.WND_PROP_FULLSCREEN)

cnt = len(img_files) #파일개수
idx = 0 #인덱스값 

while True:

    img = cv2.imread(img_files[idx])
    if img is None:
        print("Image load failed!")
        break

    cv2.imshow('image', img)

    if cv2.waitKey(1000) == 27:
        break
    
    idx += 1
    if idx >= cnt:
        idx = 0


cv2.destroyAllWindows()