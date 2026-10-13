import sys
import cv2
import numpy as np

def diagonal_intersection(corner_1, corner_2, corner_3, corner_4):
    """
    计算四边形对角线交点，返回交点坐标
    :param corner_1: 四边形左上角点坐标
    :param corner_2: 四边形左下角点坐标
    :param corner_3: 四边形右下角点坐标
    :param corner_4: 四边形右上角点坐标
    :return:
    """
    #(x1_a, y1_a) = ((img1_transform[0][0] + transform_dist[0], img1_transform[0][1] + transform_dist[1]))  # 左上
    #(x2_a, y2_a) = ((img1_transform[1][0] + transform_dist[0], img1_transform[1][1] + transform_dist[1]))  # 左下
    #(x3_a, y3_a) = ((img1_transform[2][0] + transform_dist[0], img1_transform[2][1] + transform_dist[1]))  # 右下
    #(x4_a, y4_a) = ((img1_transform[3][0] + transform_dist[0], img1_transform[3][1] + transform_dist[1]))  # 右上
    #cv2.line(result_img, (x1_a, y1_a), (x3_a, y3_a), (0, 0, 255), 2)
    #cv2.line(result_img, (x2_a, y2_a), (x4_a, y4_a), (0, 0, 255), 2)
    denominator = (corner_2[0] * corner_1[1]) - (corner_4[0] * corner_1[1]) - (corner_1[0] * corner_2[1]) + (corner_3[0] * corner_2[1]) - (corner_2[0] * corner_3[1]) \
                  + (corner_4[0] * corner_3[1]) + (corner_1[0] * corner_4[1]) - (corner_3[0] * corner_4[1])
    X_numerator = (corner_2[0] * corner_3[0] * corner_1[1]) - (corner_3[0] * corner_4[0] * corner_1[1]) - (corner_1[0] * corner_4[0] * corner_2[1]) + (corner_3[0] * corner_4[0] * corner_2[1]) \
                  - (corner_1[0] * corner_2[0] * corner_3[1]) + (corner_1[0] * corner_4[0] * corner_3[1]) + (corner_1[0] * corner_2[0] * corner_4[1]) - (corner_2[0] * corner_3[0] * corner_4[1])
    Y_numerator = (corner_3[0] * corner_1[1] * corner_2[1]) - (corner_4[0] * corner_1[1] * corner_2[1]) - (corner_1[0] * corner_2[1] * corner_3[1]) + (corner_4[0] * corner_2[1] * corner_3[1])\
                  + (corner_2[0] * corner_1[1] * corner_4[1]) - (corner_3[0] * corner_1[1] * corner_4[1]) + (corner_1[0] * corner_3[1] * corner_4[1]) - (corner_2[0] * corner_3[1] * corner_4[1])
    #Y_numerator = (x3_a * y1_a * y2_a) - (x4_a * y1_a * y2_a) - (x1_a * y2_a * y3_a) + (x4_a * y2_a * y3_a) + (
    #            x2_a * y1_a * y4_a) - (x3_a * y1_a * y4_a) + (x1_a * y3_a * y4_a) - (x2_a * y3_a * y4_a)
    # cv2.circle(result_img, (fengzi1 // fengmu, fengzi2 // fengmu), radius=5, color=(255, 0, 0), thickness=-1)
    intersection_point = (X_numerator//denominator, Y_numerator//denominator)  # 交点坐标
    return intersection_point

def stitch_image(img1, img2, H):
    """
    通过匹配图像的特征点，拼接两图像
    :param img1: 左侧图像，待透视变换后拼接
    :param img2: 右侧图像
    :param H: 特征点匹配后构建的单应性矩阵
    :return:
    """

    # 3. 创建一张大图，将两张图拼接到一起
    # 4. 将结果输出

    # 1. 获得原始图的高/宽
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_dims = np.float64([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float64([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # 2. 对图片进行透视变换（单应性矩阵使拼接图进行旋转，平移）
    img1_transform = cv2.perspectiveTransform(img1_dims, H)

    #拼接后图像区域对角坐标
    result_dims = np.concatenate((img2_dims, img1_transform), axis=0)
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 变换后界外的区域，即x、y负半轴的最远点
    # 平移进正半轴的距离
    transform_dist = [abs(x_min), abs(y_min)]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    #mat_src = np.float32([[0, 0], [0, h1], [w1, h1]])
    #mat_dst = np.float32([[0+transform_dist[0], 0+transform_dist[1]], [0+transform_dist[0], h1+transform_dist[1]], [w1+transform_dist[0], h1+transform_dist[1]]])
    #transform_array = cv2.getAffineTransform(mat_src, mat_dst)
    #homogeneousRow = np.array([[0.0, 0.0, 1.0]])
    #transform_array = np.concatenate((transform_array, homogeneousRow), axis=0)

    result_img = cv2.warpPerspective(img1, transform_array.dot(H), (x_max - x_min, y_max - y_min))

    result_img[transform_dist[1]:transform_dist[1]+h2, transform_dist[0]:transform_dist[0]+w2] = img2

    '''
    # 将拼接后的不规则图像转为矩形平面图 
    img1_transform = np.int32(img1_transform.reshape(-1, 2))
    img1_transform = img1_transform + transform_dist
    corner_A1 = (img1_transform[0][0], img1_transform[0][1])  # 左上
    corner_A2 = (img1_transform[1][0], img1_transform[1][1])  # 左下
    corner_A3 = (img1_transform[2][0], img1_transform[2][1])  # 右下
    corner_A4 = (img1_transform[3][0], img1_transform[3][1])  # 右上
    cv2.circle(result_img, corner_A1, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(result_img, corner_A2, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(result_img, corner_A3, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(result_img, corner_A4, radius=5, color=(0, 0, 255), thickness=-1)
    #cv2.line(result_img, corner_A1, corner_A3, (0, 0, 255), 2)
    #cv2.line(result_img, corner_A2, corner_A4, (0, 0, 255), 2)
    # intersection_point = diagonal_intersection(corner_A1, corner_A2, corner_A3, corner_A4)
    # cv2.circle(result_img, intersection_point, radius=5, color=(0, 0, 255), thickness=-1)

    img2_dims = np.int32(img2_dims.reshape(-1, 2))
    img2_dims = img2_dims + transform_dist
    corner_B1 = (img2_dims[0][0], img2_dims[0][1])
    corner_B2 = (img2_dims[1][0], img2_dims[1][1])
    corner_B3 = (img2_dims[2][0], img2_dims[2][1])
    corner_B4 = (img2_dims[3][0], img2_dims[3][1])
    cv2.circle(result_img, corner_B1, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(result_img, corner_B2, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(result_img, corner_B3, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(result_img, corner_B4, radius=5, color=(255, 0, 0), thickness=-1)
    #cv2.line(result_img, corner_B1, corner_B3, (255, 0, 0), 2)
    #cv2.line(result_img, corner_B2, corner_B4, (255, 0, 0), 2)
    # intersection_point = diagonal_intersection(corner_B1, corner_B2, corner_B3, corner_B4)
    # cv2.circle(result_img, intersection_point, radius=5, color=(255, 0, 0), thickness=-1)

    cv2.line(result_img, corner_A1, corner_A4, (0, 0, 255), 2)
    cv2.line(result_img, corner_B1, corner_B4, (0, 0, 255), 2)
    intersection_point_RU = diagonal_intersection(corner_A1, corner_B1, corner_A4, corner_B4)
    cv2.circle(result_img, intersection_point_RU, radius=5, color=(0, 0, 255), thickness=-1)

    cv2.line(result_img, corner_A2, corner_A3, (255, 0, 0), 2)
    cv2.line(result_img, corner_B2, corner_B3, (255, 0, 0), 2)
    intersection_point_RD = diagonal_intersection(corner_B2, corner_A2, corner_B3, corner_A3)
    cv2.circle(result_img, intersection_point_RD, radius=5, color=(0, 0, 255), thickness=-1)

    src_points = np.float32([corner_A1, corner_A2, intersection_point_RD, intersection_point_RU])
    distance_1 = cv2.norm(np.array(corner_A1), np.array(intersection_point_RU), cv2.NORM_L2)
    distance_2 = cv2.norm(np.array(corner_A2), np.array(intersection_point_RD), cv2.NORM_L2)
    distance = (distance_1+distance_2)/2
    dst_points = np.float32([(intersection_point_RU[0]-distance, intersection_point_RU[1]), (intersection_point_RD[0]-distance, intersection_point_RD[1]), intersection_point_RD, intersection_point_RU])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    new_img = cv2.warpPerspective(result_img[:], matrix, dsize=(distance+intersection_point_RU[0]))'''

    return result_img


def get_homo(img1, img2):
    # 1. 创建特征转换对象
    # 2. 通过特征转换对象获得特征点和描述子
    # 3. 创建特征匹配器
    # 4. 进行特征匹配
    # 5. 过滤特征，找出有效的特征匹配点

    sift = cv2.xfeatures2d.SIFT_create()

    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # 创建特征匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # 过滤特征，找出有效的特征匹配点
    verify_ratio = 0.8
    verify_matches = []
    for m1, m2 in matches:
        if m1.distance < verify_ratio * m2.distance:
            verify_matches.append(m1)

    min_matches = 8
    if len(verify_matches) > min_matches:

        # match_result = cv2.drawMatchesKnn(img1, k1, img2, k2, matches, None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #match_result = cv2.drawMatches(img1, k1, img2, k2, verify_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv2.imshow('match_result', match_result)

        img1_pts = []
        img2_pts = []
        # 得到筛选后特征点的匹配坐标
        for m in verify_matches:
            img1_pts.append(k1[m.queryIdx].pt)
            img2_pts.append(k2[m.trainIdx].pt)
        # [(x1, y1), (x2, y2), ...]
        # [[x1, y1], [x2, y2], ...]

        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
        # 获取透视变换的单应性矩阵
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return H
    else:
        print('err: Not enough matches!')
        exit()


# 第一步，读取文件，将图片设置成一样大小640x480
# 第二步，找特征点，描述子，计算单应性矩阵
# 第三步，根据单应性矩阵对图像进行变换，然后平移
# 第四步，拼接并输出最终结果
print(sys.version)
# 读取两张图片
img1 = cv2.imread('./imgA.png')
img2 = cv2.imread('./imgB.png')

# 将两张图片设置成同样大小
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# 获得单应性矩阵
H = get_homo(img1, img2)

# 进行图像拼接
result_image = stitch_image(img1, img2, H)

cv2.imshow('input img', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()