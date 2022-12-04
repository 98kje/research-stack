import sys
import cv2


print("Hello,opencv",cv2.__version__)
img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE) #현재폴더에있는 cat.bmp라는 폴더를 화면에 띄우겠다

if img is None:
    print("image load failed!.")
    sys.exit()

cv2.imwrite('cat_gray.png',img ) # 확장자를 보고 그대로 저장 즉 그레이로 만든거 저장한다는뜻. 이건 위에 그레이로 저장했으니 저장도 그레이로 저장
cv2.namedWindow('image') #opencv 지원함수로 이미지라는 창을 띄어옴. Autosize가 dafault
cv2.imshow('image', img) #image show 줄인말로 image 창에 img라는 영상을 보여줘라. 입력한거
print("종료 하시려면 q를 눌러주세요")
while True:
    if cv2.waitKey()== ord("q"): # 키보드 입력을 기다리는 역할 하면서 동시에 영상이 화면에 보여지도록 //아니면 == 뒤에 27 여기서 27은 esc의 아스키코드값 엔터는 13, 탭은 9
     break#ord함수는 특정키 입력받으면 종료되도록.


cv2.destroyAllWindows('image') #기존에 화면에 나와있는 모든창을 닫아라.