'''import math
def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
angle = getAngle([[1,math.sqrt(3)],[0,0],[0,math.sqrt(3)]])
print(angle)'''

import numpy as np
def IoU(vertice1, vertice2):
    """
    这是计算两个矩形区域的交并比函数，verticle格式为:[xin,ymin,w,h]
    :param vertice1: 目标框1
    :param vertice2: 目标框2
    :return: 两个矩形区域的交并比
    """

    # vertice2包含于vertice1中
    x1_a,y1_a,x2_a,y2_a = vertice1[0],vertice1[1],vertice1[0]+vertice1[2],vertice1[1]+vertice1[3]
    x1_b, y1_b, x2_b, y2_b = vertice2[0], vertice2[1], vertice2[0] + vertice2[2], vertice2[1] + vertice2[3]
    if (x1_a <= x1_b) & (y1_a <= y1_b) & (x2_a >= x2_b) & (y2_a >= y2_b):
        return 1.00

    # 计算区域交集的左上与右下坐标
    lu = np.maximum(vertice1[0:2], vertice2[0:2])
    rd = np.minimum(vertice1[0:2]+vertice1[2:4], vertice2[0:2]+vertice2[2:4])
    # 计算区域交集的面积
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]
    # 计算区域并集的面积
    #square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    #square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    square1 = vertice1[2]*vertice1[3]
    square2 = vertice2[2]*vertice2[3]

    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    return np.clip(inter_square / union_square, 0.0, 1.0)

print(IoU(np.asarray([0,0,10,10]), np.asarray([2,2,5,5])))