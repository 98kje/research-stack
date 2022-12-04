import matplotlib.pyplot as plt  #as는 plt로 바꿔준거
import cv2


# 컬러 영상 출력
imgBGR = cv2.imread('cat.bmp')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) #BGR을 RGB로 바꾸기위해 이렇게 사용한다..

plt.axis('off')
plt.imshow(imgRGB) # 혹은 plt.imshow(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
plt.show()

# 그레이스케일 영상 출력
imgGray = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

plt.axis('off') #얘는 없으면 좌표값 출력됨 x,y함수
plt.imshow(imgGray, cmap='gray') 
plt.show()

# 두 개의 영상을 함께 출력 
plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB) #서브 플롯을 나눈건데 1행 2행중에 1행에 넣어라
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray') #창을 나눠줄때 Matplotlib으로 사용
plt.show()
