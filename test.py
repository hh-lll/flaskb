# \(^_^)/  /(^_^)\
"""
作者：ZhanMJ
时间: 2022年04月04日
"""

import numpy as np
import cv2 as cv


# 图片显示函数
def show(WindowName, image, width=700):
    image = resize(image, width=width)
    cv.imshow(WindowName, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 图片缩放函数
def resize(image, width=700, inter=cv.INTER_AREA):
    OriginWidth = image.shape[1]
    OriginHeight = image.shape[0]
    ratio = float(width/OriginWidth)
    height = ratio * OriginHeight
    resized = cv.resize(image, (int(width), int(height)), interpolation=inter)
    return resized


# 透视变换函数
def four_points_transform(Origin, Points, ratio, FinalRatio=[7, 3], FinalHeight=300):
    Pts = []
    for i in range(len(Points)):
        Pts.append((int(ratio*Points[i][0]), int(ratio*Points[i][1])))
    tl, bl, br, tr = Pts[0], Pts[1], Pts[2], Pts[3]

    dst = np.array([
        [0, 0],
        [int(FinalHeight*FinalRatio[0]/FinalRatio[1]) - 1, 0],
        [int(FinalHeight*FinalRatio[0]/FinalRatio[1]) - 1, FinalHeight - 1],
        [0, FinalHeight - 1]], dtype="float32")

    rect = np.array([
        [tl[0], tl[1]],
        [tr[0], tr[1]],
        [br[0], br[1]],
        [bl[0], bl[1]]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(Origin, M, (int(FinalHeight*FinalRatio[0]/FinalRatio[1]), FinalHeight))
    return warped


# 轮廓识别函数
def recognize_contours(Origin, PreImage):
    contours, hierarchy = cv.findContours(PreImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key=cv.contourArea, reverse=True)
    #OriginSketch = Origin.copy()
    #cv.drawContours(OriginSketch, contours, -1, (0, 0, 255), 2)
    #show("drawContours", OriginSketch)
    OriginSketch2 = Origin.copy()
    Corners = []
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
    show("OriginSketch2", OriginSketch2)
    return Corners

def cut(OriginFilename):
    # 图像的导入与尺寸调整
    # OriginFilename = "Origin (16).jpg"
    Origin = cv.imread(OriginFilename)
    PreImage = resize(Origin, width=500)
    ratio = Origin.shape[1] / PreImage.shape[1]
    #show("PreImage", PreImage)


    # 高斯滤波
    Gauss = cv.GaussianBlur(PreImage, (9, 9), 0)
    #show("MedianBlur", PreImage)


    # 自适应直方图均衡化
    #PreImageGray = cv.cvtColor(PreImage, cv.COLOR_BGR2GRAY)
    PreImageGray = Gauss[:, :, 2]
    clahe = cv.createCLAHE()
    res_clahe = clahe.apply(PreImageGray)
    #show("clahe", PreImageGray)


    # 梯度检测（Sobel算子）
    sobelx = cv.Sobel(PreImageGray, cv.CV_64F, 1, 0, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    #show("sobelx", sobelx)


    # 中值滤波
    Blur = cv.medianBlur(sobelx, 3)
    #show("MedianBlur", Blur)


    # 二值化
    (_, thresh) = cv.threshold(Blur, 90, 255, cv.THRESH_BINARY)
    #show("thresh", thresh)


    # 裁剪与透视变换
    Points = recognize_contours(PreImage, thresh)
    warped = four_points_transform(Origin, Points, ratio=ratio, FinalRatio=[7, 3], FinalHeight=600)
    warpedshow = resize(warped, width=1000)
    show("warped", warpedshow)
    Filename = "Dst_{}".format(OriginFilename)
    cv.imwrite("Dst\\"+Filename, warped)


if __name__ == '__main__':
    cut("/upload/js.jpg")
