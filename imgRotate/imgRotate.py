import cv2
import math
cv2.namedWindow("window",cv2.WINDOW_NORMAL)
img = cv2.imread("./image2.png",cv2.IMREAD_COLOR)
H,W,Ch = img.shape
cv2.imshow("window", img)

mouseLButtonDown_x = 0
mouseLButtonDown_y = 0

# 获取图像中心
center = (W // 2, H // 2)

#恢复原图像
def mouseLButtonUp_callback(event,x,y,flags,userdata):
    if (cv2.EVENT_LBUTTONUP == event):
        cv2.imshow("window", img)


#记录单击鼠标左键位置
def mouseLButtonDOWN_callback(event,x,y,flags,userdata):
    if (cv2.EVENT_LBUTTONDOWN == event):
        mouseLButtonDown_x = x
        mouseLButtonDown_y = y

#计算斜率
def gradient(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)
#计算拖动夹角
def calculate_angle(x1, y1, x2, y2, x3, y3):
    # 计算向量BA和BC间的夹角B
    vector_BA = (x1 - x2, y1 - y2)
    vector_BC = (x3 - x2, y3 - y2)

    # 计算点积
    dot_product = vector_BA[0] * vector_BC[0] + vector_BA[1] * vector_BC[1]

    # 计算向量的模长
    magnitude_BA = math.sqrt(vector_BA[0] ** 2 + vector_BA[1] ** 2)
    magnitude_BC = math.sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)

    # 计算夹角（弧度转角度）
    angle = math.degrees(math.acos(cos_angle))

    gradient1= gradient(x1, y1, x2, y2)
    gradient2 = gradient(x3, y3, x2, y2)
    if gradient1<gradient2:
      angle = -angle
    return angle

#随拖动变换图像
def mouse_callback(event,x,y,flags,params):
    global mouseLButtonDown_x,mouseLButtonDown_y
    if (cv2.EVENT_LBUTTONDOWN == event):
        mouseLButtonDown_x = x
        mouseLButtonDown_y = y
        #params[0] = x
        #params[1] = y
    if (cv2.EVENT_FLAG_LBUTTON == flags) & (cv2.EVENT_MOUSEMOVE == event):
        angle = calculate_angle(mouseLButtonDown_x, mouseLButtonDown_y, center[0], center[1], x, y)
        # 设置旋转矩阵
        rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 第三个参数是旋转角度，第四个参数是缩放因子，最后一个参数是旋转中心
        # 执行旋转
        rotated_img = cv2.warpAffine(img, rotationMatrix, (img.shape[1], img.shape[0]))
        cv2.imshow("window", rotated_img)


        #print("mouseLButtonDown_x:" + str(params[0]) + ",mouseLButtonDown_y:" + str(params[1]))
        #print("center_x:"+str(center[0])+",center_y:"+str(center[1]))
        #print("x:"+str(x)+",y:"+str(y))
        print(angle)
    if (cv2.EVENT_LBUTTONUP == event):
        cv2.imshow("window", img)

#cv2.setMouseCallback("window", mouse_callback, [mouseLButtonDown_x,mouseLButtonDown_y])
cv2.setMouseCallback("window", mouse_callback, "")

while True:
    if cv2.waitKey(10) == ord('q'):
        break
# 销毁窗口
cv2.destroyAllWindows()