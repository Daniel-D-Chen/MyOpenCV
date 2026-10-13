#123
import cv2
import numpy as np
cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('window',(640,480))
def callback(value):
    print(value)
cv2.createTrackbar('R','window',0,255,callback)
cv2.createTrackbar('G','window',0,255,callback)
cv2.createTrackbar('B','window',0,255,callback)
img = np.zeros((480,640,3),np.uint8)
while True:
    R = cv2.getTrackbarPos("R",'window')
    G = cv2.getTrackbarPos("G",'window')
    B = cv2.getTrackbarPos("B",'window')
    img[:] = [B,G,R]
    #img[:,:,0] = B
    #img[:, :, 1] = G
    #img[:, :, 2] = R

    cv2.imshow("window", img)
    # 按下'q'键退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# 销毁窗口
cv2.destroyAllWindows()