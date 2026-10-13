# 统计视频中待检测区域中，运动物体计数
# 掩膜减少计算，鼠标动作规定检测区域，IOU或非极大值抑制减少检测框

import cv2
import numpy as np
import sys



# 异常退出处理
def exceptionHandler():
    global cap
    try:
        sys.exit("程序结束")
    except SystemExit as e:
        print("捕获异常退出:", e)
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


def IoU(rectangle_A, rectangle_B, selfPropThresh):
    """
    这是计算两个矩形区域的交并比函数，rectangle格式为:[x_min,y_min,w,h]=[矩形左上角X坐标，矩形左上角Y坐标，矩形宽，矩形高]
    :param rectangle_A: 目标框1
    :param rectangle_B: 目标框2
    :param selfPropThresh: 自身占比异常阈值
    :return: 两个矩形区域的交并比
    """
    # 计算区域并集的面积
    rectArea_A = rectangle_A[2] * rectangle_A[3]
    rectArea_B = rectangle_B[2] * rectangle_B[3]

    # 计算区域交集的左上与右下坐标
    vertex_LU = np.maximum(rectangle_A[0:2], rectangle_B[0:2])
    vertex_RD = np.minimum(rectangle_A[0:2]+rectangle_A[2:4], rectangle_B[0:2]+rectangle_B[2:4])
    # 计算区域交集的面积
    intersection = np.maximum(0.0, vertex_RD - vertex_LU)
    interArea = intersection[0] * intersection[1]

    # 交集区域占任一目标框面积比例超过阈值，交并比值视为1.00
    if (interArea/rectArea_A > selfPropThresh) or (interArea/rectArea_B > selfPropThresh):
        return 1.00

    # 计算并集面积=两矩形面积之和-交集面积
    union_square = np.maximum(rectArea_A + rectArea_B - interArea, 1e-10)
    return np.clip(interArea / union_square, 0.0, 1.0)


def nonMaximumSuppression(rectangles, thresh):
    """
    这是NMS(非极大值抑制)去除重复目标框的函数
    :param rectangles: 目标框数组，目标框的格式为：[xin, ymin, width, height, area]
    :param thresh: 阈值
    :return: 不重复的目标框数组在原目标框数组中的下标数组
    """
    # 备选框不超过单个，返回所有目标框
    if len(rectangles) <= 1:
        return rectangles

    rect = rectangles[:, 0:4]  # 目标框
    scores = rectangles[:, 4]  # 目标框面积

    # 按面积从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，保留
        keep.append(order[0])
        # 计算窗口i与其他所有窗口的交叠部分的面积
        ious = np.array([IoU(rect[order[0]], rect[j], 0.5) for j in order[1:]])
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ious <= thresh)[0]
        # order里面只保留与窗口0交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含0)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return rectangles[keep]

# 计算对象包围框中心点
def center(x, y, w, h):
    x1 = x + w / 2
    y1 = y + h / 2
    return x1, y1

# 视频帧预处理
def preProcess(img):
    global mask
    # 将图像转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 保留检测区域原本灰度值，其余区域灰度置为0
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    # 使用高斯滤波去噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 去背景
    FGmask = bgsubmog.apply(blur)
    # 腐蚀去掉图中的小方块
    erode = cv2.erode(FGmask, kernel, iterations=1)
    # 膨胀放大
    dilate = cv2.dilate(erode, kernel, iterations=3)
    # 闭操作，去掉物体内部的小方块
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    # close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)
    return close


'''def drawContours(img, points):
    global objRect, objRect_area
    rectList, rectListArea = [],[]
    # 根据轮廓画矩形
    for (i, points) in enumerate(contours):
        # print(i, points)
        # 画最大外接矩形,太小就不画
        x, y, w, h = cv2.boundingRect(points)
        # if w < min_w and h < min_h:
        #    continue
        # 绘出矩形框
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rectList.append([(x, y), (x + w, y + h), w*h])
        #rectListArea.append()

    objRect = np.array(rectList)
    #objRect_area = np.array(rectListArea)'''


def mouse_callback(event, x, y, flags, params):
    # 拖动鼠标画出统计区域
    global detectArea_x, detectArea_y, detectArea_height, detectArea_width, firstFrame, mask, evaluation_line
    frame = firstFrame.copy()
    if (cv2.EVENT_LBUTTONDOWN == event):
        detectArea_x = x
        detectArea_y = y
    if (cv2.EVENT_FLAG_LBUTTON == flags) & (cv2.EVENT_MOUSEMOVE == event):
        # 画出统计区域
        cv2.rectangle(frame, (detectArea_x, detectArea_y), (x, y), (0, 0, 255), 2)
        # print((mouseLButtonDown_x, mouseLButtonDown_y), (x, y))
        cv2.imshow(windowName, frame)
    if (cv2.EVENT_LBUTTONUP == event):
        # 计算统计区域宽度
        detectArea_width = (x - detectArea_x) if x >= detectArea_x else (detectArea_x - x)
        # 获取区域左上点X坐标
        detectArea_x = detectArea_x if x >= detectArea_x else x
        # 计算统计区域高度
        detectArea_height= (y - detectArea_y) if y >= detectArea_y else (detectArea_y - y)
        # 获取区域左上点Y坐标
        detectArea_y = detectArea_y if y >= detectArea_y else y

        # 取检测区域下边框，作为车数量统计判別线
        evaluation_line = detectArea_y + detectArea_height - 200
        mask = np.zeros(firstFrame.shape[0:2], np.uint8)
        mask[detectArea_y:detectArea_y+detectArea_height, detectArea_x:detectArea_x+detectArea_width] = 255
        # cv2.imshow(windowName, firstFrame)

def main():
    global firstFrame, count, cap
    # 创建视频帧对象
    cap = cv2.VideoCapture('./video.mp4')
    # 检测视频是否被打开
    if not cap.isOpened():
        print('video open failed')
        sys.exit(0)

    ret, firstFrame = cap.read()
    if not ret:
        print('视频帧读取有误')
        exceptionHandler()
    cv2.putText(firstFrame, "press space to start video", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.imshow(windowName, firstFrame)
    # 鼠标事件绑定
    cv2.setMouseCallback(windowName, mouse_callback, "")
    while True:
        key = cv2.waitKey(0) & 0xff
        if key == 32:
            print('按下空格键，开始播放视频')
            break

    # 循环读取图片
    while True:
        ret, frame = cap.read()
        # 检测是否正确读取视频帧
        if not ret:
            print('视频帧读取有误')
            break
        # 显示待检测区域轮廓
        cv2.rectangle(frame, (detectArea_x, detectArea_y), (detectArea_x+detectArea_width, detectArea_y+detectArea_height), (0, 0, 255), 2)
        # 显示数量判定线
        cv2.line(frame, (10, evaluation_line), (1270, evaluation_line), (0, 0, 255), 2)
        # 进行形态学处理
        close = preProcess(frame)
        # 检测轮廓
        close, contours, hier = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 外轮廓转为外接包围框
        boundRects = []
        for (i, points) in enumerate(contours):
            # print(i, points)
            x, y, w, h = cv2.boundingRect(points)
            # 剔除宽或高度低于最小阈值的包围框
            if w < min_w and h < min_h:
                continue
            boundRects.append([x, y, w, h, w*h])

        # 采用非极大值抑制筛取预选框
        boundRects = nonMaximumSuppression(np.array(boundRects), 0.2)
        for (i, rectPoint) in enumerate(boundRects):
            x, y, w, h = rectPoint[0], rectPoint[1], rectPoint[2], rectPoint[3]
            # 画筛选后包围框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 包围框标号
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            # 求出车框的中心点
            cen_point = center(x, y, w, h)
            cars.append(cen_point)

        # 统计过线的车的数量
        for (x, y) in cars:
            if (y > evaluation_line - offest) and (y < evaluation_line + offest):
                count = count + 1
                cars.remove((x, y))
                # print(count)
        # 将count打印在屏幕上
        cv2.putText(frame, "count = " + str(count), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        cv2.imshow('Video', frame)


        # 每帧显示后键盘操作
        key = cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xff
        if key == 27:
            break  # 按ESC退出
        elif key == 32:  # 空格暂停播放
            while True:
                key = cv2.waitKey(0)
                if key == 32:  # 再次空格继续
                    break


if __name__ == "__main__":
    windowName = "Video"
    # 检测区域左上角坐标及宽、高
    detectArea_x, detectArea_y = 0, 0
    detectArea_width, detectArea_height = 0, 0
    # 检测区域遮罩掩膜
    mask = None
    # 预选框
    objRect = None
    # 矩形最小宽高
    min_w,min_h = 90, 90
    # 存储有效车辆的数组
    cars = []
    # 计数判別区域初始默认值
    evaluation_line = 600
    # 计数判別区域宽度
    offest = 4
    # 车的数量
    count = 0
    # 创建去背景的对象
    bgsubmog = cv2.createBackgroundSubtractorMOG2()
    # 卷积核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 视频首帧画面
    firstFrame = None
    # 视频对象
    cap = None
    main()
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()