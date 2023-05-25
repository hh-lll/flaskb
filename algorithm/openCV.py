# \(^_^)/  /(^_^)\
"""
作者：ZhanMJ
时间: 2022年04月04日
"""
import math
from datetime import datetime
from operator import itemgetter


import cv2 as cv
import os

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from algorithm import UNet

# 图片缩放函数
def resize(image, width=700, inter=cv.INTER_AREA):
    OriginWidth = image.shape[1]
    OriginHeight = image.shape[0]
    ratio = float(width / OriginWidth)
    height = ratio * OriginHeight
    resized = cv.resize(image, (int(width), int(height)), interpolation=inter)
    return resized

# 透视变换函数
def four_points_transform(Origin, Points, ratio, FinalRatio=[7, 3], FinalHeight=300):
    Pts = []
    for i in range(len(Points)):
        Pts.append((int(ratio * Points[i][0]), int(ratio * Points[i][1])))
    tl, bl, br, tr = Pts[0], Pts[1], Pts[2], Pts[3]

    dst = np.array([
        [0, 0],
        [int(FinalHeight * FinalRatio[0] / FinalRatio[1]) - 1, 0],
        [int(FinalHeight * FinalRatio[0] / FinalRatio[1]) - 1, FinalHeight - 1],
        [0, FinalHeight - 1]], dtype="float32")

    rect = np.array([
        [tl[0], tl[1]],
        [tr[0], tr[1]],
        [br[0], br[1]],
        [bl[0], bl[1]]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(Origin, M, (int(FinalHeight * FinalRatio[0] / FinalRatio[1]), FinalHeight))
    return warped


# 轮廓识别函数
def recognize_contours(Origin, PreImage):
    contours, hierarchy = cv.findContours(PreImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    OriginSketch2 = Origin.copy()
    AllPoints = []
    for c in contours:
        for p in c:
            AllPoints.append((p[0][0], p[0][1]))
    w = PreImage.shape[1]
    h = PreImage.shape[0]
    min1 = w ** 2 + h ** 2
    min2 = w ** 2 + h ** 2
    min3 = w ** 2 + h ** 2
    min4 = w ** 2 + h ** 2
    for p in AllPoints:
        D2tl = (p[0] - 0) ** 2 + (p[1] - 0) ** 2
        D2bl = (p[0] - 0) ** 2 + (p[1] - h) ** 2
        D2tr = (p[0] - w) ** 2 + (p[1] - 0) ** 2
        D2br = (p[0] - w) ** 2 + (p[1] - h) ** 2
        if D2tl < min1:
            min1 = D2tl
            tl = p
        if D2bl < min2:
            min2 = D2bl
            bl = p
        if D2tr < min3:
            min3 = D2tr
            tr = p
        if D2br < min4:
            min4 = D2br
            br = p
    Corners = [tl, bl, br, tr]
    for center in Corners:
        cv.circle(OriginSketch2, center, 3, (255, 0, 0), 2)
    return Corners


# 最小二乘法拟合直线
def least_squares_fit(lines):
    """
    将lines中的线段根据最小二乘法拟合成一条线段
    :param lines: 线段集合，[np.array([[x1, y1, x2, y2]]), ..., np.array([[x1, y1, x2, y2]])]
    :return: 线段上的两点，np.array([[xmin, ymin], [xmax, ymax]])
    """
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, deg=1)
    point_min = (0, np.polyval(poly, 0))
    point_max = (700, np.polyval(poly, 700))
    return np.array([point_min, point_max], dtype=np.int64)


# 斜率平均值拟合直线
def average_slope_fit(lines):
    """
    将lines中的线段根据斜率平均值拟合成一条线段
    :param lines: 线段集合，[np.array([[x1, y1, x2, y2]]), ..., np.array([[x1, y1, x2, y2]])]
    :return: 线段上的两点，np.array([[xmin, ymin], [xmax, ymax]])
    """
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    x_avg = np.average(x_coords)
    y_avg = np.average(y_coords)
    slope_sum = 0
    for i in range(len(lines)):
        deltaX = lines[i][0][0] - lines[i][0][2]
        deltaY = lines[i][0][1] - lines[i][0][3]
        slope_sum += deltaX / deltaY
    slope = slope_sum / len(lines)
    b = x_avg - slope * y_avg
    point_start = (b, 0)
    point_end = (slope * 300 + b, 300)
    return np.array([point_start, point_end], dtype=np.int64)


# 求两直线交点坐标
def CrossPoint(L_hzt, L_Vtc):
    x1, y1, x2, y2 = L_hzt[0][0], L_hzt[0][1], L_hzt[1][0], L_hzt[1][1]
    x3, y3, x4, y4 = L_Vtc[0][0], L_Vtc[0][1], L_Vtc[1][0], L_Vtc[1][1]
    Y_coord = int(((x3 - x1) * (y1 - y2) * (y3 - y4) - y3 * (y1 - y2) * (x3 - x4) + y1 * (x1 - x2) * (y3 - y4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
    X_coord = int((x3 - x4) / (y3 - y4) * (Y_coord - y3) + x3)
    Coord = (X_coord, Y_coord)
    return Coord


# 伽马变换提高对比度
def gama_transfer(image, power1):
    image = 255 * np.power(image / 255, power1)
    image = np.round(image)
    image[image > 255] = 255
    out_img = image.astype(np.uint8)
    return out_img


def min(fold, filename):
    # fold 文件夹的路径
    # filename 图片的名称，包括后缀
    # for count in range(9):
    # 图像的导入与尺寸调整
    # fold = '../upload/'
    # filename = '12.jpg'
    path = os.path.join(fold)
    Origin = cv.imread(path + filename)
    img_270 = cv.flip(cv.transpose(Origin), 0)
    PreImage = resize(img_270, width=1000)
    ratio = Origin.shape[1] / PreImage.shape[1]

    # 高斯滤波
    Gauss = cv.GaussianBlur(PreImage, (9, 9), 0)

    # 自适应直方图均衡化
    # PreImageGray = cv.cvtColor(PreImage, cv.COLOR_BGR2GRAY)
    PreImageGray = Gauss[:, :, 2]
    clahe = cv.createCLAHE()
    res_clahe = clahe.apply(PreImageGray)

    # 梯度检测（Sobel算子）
    sobelx = cv.Sobel(PreImageGray, cv.CV_64F, 1, 0, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)

    # 中值滤波
    Blur = cv.medianBlur(sobelx, 3)

    # 二值化
    (_, thresh) = cv.threshold(Blur, 90, 255, cv.THRESH_BINARY)

    # 裁剪与透视变换
    Points = recognize_contours(PreImage, thresh)
    warped = four_points_transform(Origin, Points, ratio=ratio, FinalRatio=[7, 3], FinalHeight=600)
    Filename = "DstLIne_" + filename

    path = os.path.join("images/Dst/")
    cv.imwrite(path + Filename, warped)

    # Unet
    classes = 1  # exclude background

    img_path = path + Filename

    weights_path = "save_weights/best_model.pth"
    roi_mask_path = "save_weights/mask.jpg"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    roi_img = cv.flip(roi_img, -1)
    # load image

    original_img = Image.open(img_path).convert('RGB')
    original_img = np.array(original_img)
    original_img = cv.flip(original_img, -1)
    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        output = model(img.to(device))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不兴趣的区域像素设置成0(敢黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
    # try:
    # 托槽参考线识别
    # 基座图像读取与尺寸调整
    # warped = cv.resize(warped, (1400, 600))

    # prediction = np.rot90(prediction, 2)
    prediction = cv.flip(prediction, -1)
    prediction = cv.resize(prediction, (1400, 600))

    prediction = cv.cvtColor(prediction, cv.COLOR_GRAY2RGB)
    # image = cv.add(warped, prediction)
    bracket = resize(prediction, width=700)
    # 直接用Unet输出的图片进行边缘检测
    bracketBorder = cv.Canny(bracket, 20, 80)
    bracketContours, bracketHierarchy = cv.findContours(bracketBorder, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    min_size = 130
    max_size = 1000
    delete_list = []
    for i in range(len(bracketContours)):
        area = cv.contourArea(bracketContours[i]);
        if (cv.arcLength(bracketContours[i], True) < min_size) or (
                cv.arcLength(bracketContours[i], True) > max_size) or area < 10:
            delete_list.append(i)

    # 根据列表序号删除不符合要求的轮廓
    bracketContours = delet_contours(bracketContours, delete_list)
    temp = np.zeros(bracket.shape, dtype=np.uint8)
    for i in range(len(bracketContours)):
        # 6.2 凸包
        hull = cv.convexHull(bracketContours[i])
        cv.polylines(temp, [hull], True, (255, 255, 255), 1)


    res = temp

    # bracketMask = np.zeros_like(bracket)
    # bmask1 = [[100, 30], [170, 30], [170, 65], [100, 65]]
    #
    # bmask2 = [[180, 30], [240, 30], [240, 65], [180, 65]]
    #
    # bmask3 = [[260, 30], [350, 30], [350, 65], [260, 65]]
    #
    # bmask4 = [[370, 30], [430, 30], [430, 65], [370, 65]]
    #
    # bmask5 = [[450, 30], [520, 30], [520, 65], [450, 65]]
    #
    # bmask6 = [[530, 30], [610, 30], [610, 65], [530, 65]]
    #
    # # MaskPts6 = [[100, 30], [170, 30], [170, 70], [100, 70]]
    # bAllMask = np.array([bmask1, bmask2, bmask3, bmask4, bmask5, bmask6])
    # cv.fillPoly(bracketMask, bAllMask, color=(255, 255, 255))
    # bwithMask = cv.bitwise_and(res, bracketMask)
    # bwithMask = cv.cvtColor(bwithMask, cv.COLOR_BGR2GRAY)
    # bracketTop = cv.HoughLinesP(bwithMask, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=40)
    # bracketTop = quick_sort(bracketTop, 0, bracketTop.shape[0] - 1)
    # bracketTop = pick_six(bracketTop, 15)

    bracketPosition = []
    bracketLeftPoint = []
    bracketRightPoint = []
    for cnt in bracketContours:
        # 计算轮廓的上部最靠近图像顶部的点
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        #     cnt = sorted(cnt, key=itemgetter(0),reverse=True)
        cnt = sorted(cnt, key=(lambda x: x[0][0]))
        i = 0
        while cnt[i][0][1] - topmost[1] > 5:
            i = i + 1

        # 从这个点开始向左和向右扫描，找到最左边和最右边的两个点
        leftmost = tuple(cnt[i][0])
        bracketLeftPoint.append(leftmost)
        cnt = sorted(cnt, key=(lambda x: x[0][0]), reverse=True)
        i = 0
        while abs(cnt[i][0][1] - topmost[1]) > 5 :
            i = i + 1
        rightmost = tuple(cnt[i][0])
        bracketRightPoint.append(rightmost)
        x = int((leftmost[0]+rightmost[0])/2)
        y = int((leftmost[1]+rightmost[1])/2)
        xy = (x, y)
        bracketPosition.append(xy)

        # cv.line(img, leftmost, rightmost, (0, 255, 0), 1)


    image = resize(warped, width=700)
    Sketch = image.copy()

    # 灰度图获取
    # gray = image[:, :, 0]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 伽马变换对比度增强
    gama = gama_transfer(gray, 3)

    # Canny边缘检测
    BaseLines = cv.Canny(gama, 20, 80)

    Filename = "Canny_" + filename

    path = os.path.join("images/Canny/")
    cv.imwrite(path + Filename, BaseLines)

    # 基座水平线ROI提取
    BaseHztLineMask = np.zeros_like(BaseLines)
    LMaskPts = [[10, 40], [10, 80], [55, 80], [55, 40]]
    RMaskPts = [[645, 40], [645, 80], [690, 80], [690, 40]]
    AllHztMaskPts = np.array([LMaskPts, RMaskPts])
    cv.fillPoly(BaseHztLineMask, AllHztMaskPts, color=255)
    BaseHztLine = cv.bitwise_and(BaseLines, BaseHztLineMask)

    # 霍夫变换获取水平线方程
    HztLines = cv.HoughLinesP(BaseHztLine, 1, np.pi / 360, 15, minLineLength=20, maxLineGap=20)
    # print("{} lines detected!".format(len(HztLines)))

    # 平均值拟合水平线
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in HztLines])
    y_avg = int(np.average(y_coords))
    BaseImageSketch1 = Sketch.copy()
    cv.line(BaseImageSketch1, (0, y_avg), (700, y_avg), (255, 0, 0), 2)
    HorizontalLine = np.array([(0, y_avg), (700, y_avg)], dtype=np.int64)

    # 最小二乘法拟合水平线
    HorizontalLineSquare = least_squares_fit(HztLines)
    BaseImageSketch2 = Sketch.copy()
    cv.line(BaseImageSketch2, (HorizontalLineSquare[0]), (HorizontalLineSquare[1]), (0, 0, 255), 2)

    # 基座纵向线ROI提取
    B3MaskPts = [[100, 160], [100, 230], [150, 230], [150, 160]]
    B2MaskPts = [[180, 160], [180, 230], [230, 230], [230, 160]]
    B1MaskPts = [[270, 160], [270, 230], [320, 230], [320, 160]]
    A1MaskPts = [[380, 160], [380, 230], [430, 230], [430, 160]]
    A2MaskPts = [[470, 160], [470, 230], [520, 230], [520, 160]]
    A3MaskPts = [[550, 160], [550, 230], [600, 230], [600, 160]]
    AllVtcMaskPts = np.array([B3MaskPts, B2MaskPts, B1MaskPts, A1MaskPts, A2MaskPts, A3MaskPts])

    # 纵向线方程拟合
    VerticalLines = []
    i = 0
    BaseImageSketch3 = Sketch.copy()
    BaseImageSketch4 = Sketch.copy()
    for VtcPt in AllVtcMaskPts:
        BaseVtcLineMask = np.zeros_like(BaseLines)
        cv.fillPoly(BaseVtcLineMask, [VtcPt], color=255)
        BaseVtcLine = cv.bitwise_and(BaseLines, BaseVtcLineMask)

        # 霍夫变换获取纵向线方程
        VtcLines = cv.HoughLinesP(BaseVtcLine, 1, np.pi / 90, 30, minLineLength=20, maxLineGap=20)
        # print("{} lines detected!".format(len(VtcLines)))

        # 平均值拟合纵向线
        VerticalLines.append(average_slope_fit(VtcLines))
        cv.line(BaseImageSketch3, (VerticalLines[i][0]), (VerticalLines[i][1]), (255, 0, 0), 1)

        # 最小二乘法拟合纵向线
        VerticalLine = least_squares_fit(VtcLines)
        cv.line(BaseImageSketch4, (VerticalLine[0]), (VerticalLine[1]), (0, 0, 255), 1)

        i += 1


    # 标准托槽位置计算
    Standard_Position = []
    BaseImageSketch5 = Sketch.copy()
    BaseImageSketch6 = Sketch.copy()
    cv.line(BaseImageSketch5, (0, y_avg), (700, y_avg), (255, 0, 0), 2)
    cv.line(BaseImageSketch6, (HorizontalLineSquare[0]), (HorizontalLineSquare[1]), (0, 0, 255), 2)
    for Line in VerticalLines:
        coord = CrossPoint(HorizontalLine, Line)
        Standard_Position.append(coord)
        cv.line(BaseImageSketch5, Line[0], Line[1], (255, 0, 0), 2)
        cv.circle(BaseImageSketch5, coord, 2, (0, 255, 0), 3)
        cv.line(BaseImageSketch6, Line[0], Line[1], (255, 0, 0), 2)
        cv.circle(BaseImageSketch6, coord, 2, (0, 255, 0), 3)

    bracketPosition = sorted(bracketPosition, key=itemgetter(0))
    bracketLeftPoint = sorted(bracketLeftPoint, key=itemgetter(0))
    bracketRightPoint = sorted(bracketRightPoint, key=itemgetter(0))
    bracketPosition = pick_sixlist(bracketPosition,5)
    bracketLeftPoint = pick_sixlist(bracketLeftPoint,5)
    bracketRightPoint = pick_sixlist(bracketRightPoint,5)
    bracketPosition = sorted(bracketPosition, key=itemgetter(0))
    bracketLeftPoint = sorted(bracketLeftPoint, key=itemgetter(0))
    bracketRightPoint = sorted(bracketRightPoint, key=itemgetter(0))

    for i in range(0,len(bracketPosition)):
        cv.line(BaseImageSketch5, bracketLeftPoint[i], bracketRightPoint[i], (0, 255, 0), 2)
        cv.circle(BaseImageSketch5, bracketPosition[i], 2, (0, 0, 255), 3)
        cv.circle(BaseImageSketch5, bracketRightPoint[i], 2, (0, 165, 255), 3)
        cv.circle(BaseImageSketch5, bracketLeftPoint[i], 2, (255, 245, 0), 3)

    StandardPointsShow = np.vstack((BaseImageSketch5, BaseImageSketch6))
    cv.imwrite("Dst\\" + Filename + ".jpg", BaseImageSketch6)

    # path = os.path.join("../images/Base/")
    path = os.path.join("images/Base/")
    Filename = "Base_" + filename
    cv.imwrite(path + Filename, StandardPointsShow)



    # 算分
    eachScore = []
    totalScore = 0
    offset = []
    # 70mm->700fx
    # 0.5mm->5fx
    # 左上颌评分
    for i in range(0, 6):
        score = {
            "horizontal": "",
            "vertical": "",
            "shaftAngle": "",
            "score": ""
        }
        offSet = {
            "horizontal": "",
            "vertical": "",
            "shaftAngle": ""
        }
        hzt_d = Standard_Position[i][0] - bracketPosition[i][0]

        vct_d = Standard_Position[i][1] - bracketPosition[i][1]
        x_d = bracketRightPoint[i][0] - bracketLeftPoint[i][0]
        y_d = bracketRightPoint[i][1] - bracketLeftPoint[i][1]
        angle = math.atan(y_d / x_d)

        if hzt_d > 5 or hzt_d < -5:
            score["horizontal"] = int(abs(hzt_d / 5))
            offSet["horizontal"] = hzt_d
        else:
            score["horizontal"] = 0
            offSet["horizontal"] = 0

        if vct_d > 25 or vct_d < 15:
            score["vertical"] = int(abs((vct_d - 20) / 5))
            offSet["vertical"] = vct_d - 20
        else:
            score["vertical"] = 0
            offSet["vertical"] = 0

        if angle > 2 or angle < -2:
            score["shaftAngle"] = int(abs(angle / 2))
            offSet["shaftAngle"] = angle
        else:
            score["shaftAngle"] = 0
            offSet["shaftAngle"] = 0
        reverse = score["horizontal"] + score["vertical"] + score["shaftAngle"]
        score["score"] = 10 - reverse
        totalScore = totalScore + 10 - reverse
        eachScore.append(score)
        offset.append(offSet)

    now = datetime.now()  # current date and time
    totalScore = totalScore/6
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    time = now.strftime("%H:%M")
    print("time:", time)
    data = {
        "date": year+"年"+month+"月"+day+"日",
        "time": time,
        "cutFile": "DstLIne_"+filename,
        "finalFile": "Base_"+filename,
        "totalScore": totalScore,
        "eachScore": eachScore,
        "offset": offset
    }
    return data

#  自定义函数：用于删除列表指定序号的轮廓
#  输入 1：contours：原始轮廓
#  输入 2：delete_list：待删除轮廓序号列表
#  返回值：contours：筛选后轮廓
def delet_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


# 4. 绘制轮廓函数
# 自定义绘制轮廓的函数（为简化操作）
# 输入1：winName：窗口名
# 输入2：image：原图
# 输入3：contours：轮廓
# 输入4：draw_on_blank：绘制方式，True在白底上绘制，False:在原图image上绘制
def drawMyContours(image, contours, draw_on_blank):
    # cv2.drawContours(image, contours, index, color, line_width)
    # 输入参数：
    # image:与原始图像大小相同的画布图像（也可以为原始图像）
    # contours：轮廓（python列表）
    # index：轮廓的索引（当设置为-1时，绘制所有轮廓）
    # color：线条颜色，
    # line_width：线条粗细
    # 返回绘制了轮廓的图像image
    if (draw_on_blank):  # 在白底上绘制轮廓
        temp = np.zeros(image.shape, dtype=np.uint8)
        # temp = cv.UMat(temp)
        h = cv.drawContours(temp, contours, -1, (255, 255, 255), 1)
    else:
        temp = image.copy()
        h = cv.drawContours(temp, contours, -1, (255, 255, 255), 1)
    return h


# 先快速排序，然后选六个出来
# 问题：六个怎么选
def quick_sort(lists, i, j):
    if i >= j:
        return lists
    pivot = lists[i][0][0]
    pa = lists[i].copy()
    low = i
    high = j
    while i < j:
        while i < j and lists[j][0][0] >= pivot:
            j -= 1
        lists[i] = lists[j].copy()
        while i < j and lists[i][0][0] <= pivot:
            i += 1
        lists[j] = lists[i].copy()
    lists[j][0][0] = pivot
    lists[j] = pa
    quick_sort(lists, low, i - 1)
    quick_sort(lists, i + 1, high)
    return lists


def pick_six(array, interval):
    i = 0
    while i < array.shape[0]-1:
        i = i + 1
        if abs(array[i][0][0] - array[i-1][0][0]) < interval:
            a = int((array[i][0][0] + array[i-1][0][0])/2)
            b = int((array[i][0][1] + array[i-1][0][1])/2)
            c = int((array[i][0][2] + array[i-1][0][2])/2)
            d = int((array[i][0][3] + array[i-1][0][3])/2)
            a1 = np.array([[a, b, c, d]])
            array = np.append(array, [a1], axis=0)
            array = np.delete(array, i, axis=0)
            array = np.delete(array, i-1, axis=0)
            i = i - 2
    return array

def pick_sixlist(list, interval):
    i = 0
    while i < len(list)-1:
        i = i + 1
        if abs(list[i][0] - list[i-1][0]) < interval:
            a = int((list[i][0] + list[i-1][0])/2)
            b = int((list[i][1] + list[i-1][1])/2)
            a1 = (a, b)
            list.append(a1)
            list.pop(i)
            list.pop(i-1)
            i = max(i - 2, 0)

    print("pick_six后", list)
    return list


