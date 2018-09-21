#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import cv2, os
import xlrd
import matplotlib.pyplot as plt
import readdicom
import dataread


# 读取dicom图像
def read_dicom(path):
    im = readdicom.getImage(path)
    return im


# 读取jpg图像
def read_jpg(path):
    im = cv2.imread(path)
    return im


# 起始点
y_start = 100
x_start = 100
def cut_center(im):
    ''' 截图图片的中心范围，去掉图片边缘的文字信息

    :param im: 要处理的图片
    :return: 处理后的图片
    '''
    return im[y_start: -100, x_start: -100]

# 小图截256 * 256，大图按原始比例截图
# min_size = 256
# max_size = 500
# def get_tophi_loc(list):
#     '''返回对图像进行切割的范围
#
#     :param list: 四个标记点的坐标信息
#     :param ratio: width／hight的比例
#     :return: 切割范围
#     '''
#     loc = np.array(list)
#     # 获取左上，右下两个点
#     x_min, y_min = np.min(loc, 0)
#     x_max, y_max = np.max(loc, 0)
#     # 获取最小范围宽度和高度
#     width = x_max - x_min
#     hight = y_max - y_min
#
#     # 默认扩大20个像素点的截取范围（上下左右各20）
#     change_width = 20
#     change_hight = 20
#     # 根据ratio对width或hight进行更正
#     if hight > max_size or width > max_size:
#         # 如果超过了最大范围，则直接按实际比例截图
#         pass
#     elif hight > width:
#         if hight > min_size:
#             new_width = hight
#             change_width = (new_width - width) / 2 + 20
#         else:
#             new_width = min_size
#             new_hight = min_size
#             change_width = (new_width - width) / 2 + 20
#             change_hight = (new_hight - hight) / 2 + 20
#     else:
#         if width > min_size:
#             new_hight = width
#             change_hight = (new_hight - hight) / 2 + 20
#         else:
#             new_width = min_size
#             new_hight = min_size
#             change_width = (new_width - width) / 2 + 20
#             change_hight = (new_hight - hight) / 2 + 20
#
#     # 对截取范围修正
#     x_min -= change_width
#     x_max += change_width
#     y_min -= change_hight
#     y_max += change_hight
#
#     return (int(x_min), int(x_max), int(y_min), int(y_max))
def get_tophi_loc(list):
    '''返回对图像进行切割的范围

    :param list: 四个标记点的坐标信息
    :param ratio: width／hight的比例
    :return: 切割范围
    '''
    loc = np.array(list)
    # 获取左上，右下两个点
    x_min, y_min = np.min(loc, 0)
    x_max, y_max = np.max(loc, 0)
    # 获取最小范围宽度和高度
    width = x_max - x_min
    hight = y_max - y_min

    # 默认扩大20个像素点的截取范围（上下左右各20）
    change_width = 20
    change_hight = 20
    if hight > width:
        change_width = (hight - width) / 2 + 20
    else:
        change_hight = (width - hight) / 2 + 20

    # 对截取范围修正
    x_min -= change_width
    x_max += change_width
    y_min -= change_hight
    y_max += change_hight

    return (int(x_min), int(x_max), int(y_min), int(y_max))

# 随机平移
def translation(x_min, x_max, y_min, y_max):
    x = -np.random.randint(30, 60)
    # y = np.random.randint(30, 60)
    x_min += x
    x_max += x
    # y_min += y
    # y_max += y

    return (x_min, x_max, y_min, y_max)


# 固定标记的坐标信息
sign_loc = [(1016, 457), (1017, 430), (1006, 457), (1007, 428), (1110, 455), (1111, 428)]
def is_sign(pt):
    '''判断坐标pt是否为固定标记
    :param pt: 点的坐标
    :return: True：是固定标记，False：不是
    '''
    for s_loc in sign_loc:
        if np.abs(s_loc[0] - x_start - pt[0]) <= 3 and np.abs(s_loc[1] - y_start - pt[1]) <= 3:
            return True
    return False


# 要处理的图像文件夹
root_dir = 'all_data'

result_dir = 'all_result/result/'
failed_dir = 'all_result/failed/'
wrong_dir = 'all_result/wrong/'

result_num = 1
def generate_cut_img(patient_no, label, path, filename):
    """对path对应的文件进行切割，同时保存处理失败的文件
        :param pt:
            filename ：文件名
            patient_no ：病例号
            path ：要处理的文件路径
            label ：良恶性标签 1 or 2
        :return:
          An `arg_scope` to use for the resnet models.
        """

    # 生成result的数量
    global result_num
    # 读取后裁剪出中心信息
    im = read_dicom(path)
    im = cut_center(im)
    # plt.imshow(im)
    # plt.show()
    # 保存原图信息，用于后面截取图片
    img_orig = np.copy(im)

    # 将图片二值化
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 180, 255, type=cv2.THRESH_BINARY)

    # 读取标记模版
    template1 = cv2.imread('model_img/model1.jpg', 0)
    template2 = cv2.imread('model_img/model2.jpg', 0)
    templates = [template1, template2]

    # 保存四个坐标的信息
    list = []
    # 发现固定标记的个数
    picture_sign_num = 0
    for template in templates:
        w, h = template.shape[::-1]
        # 使用matchTemplate对原始灰度图像和图像模板进行匹配
        res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)
        # 设定阈值
        threshold = 0.85
        # res大于85%
        loc = np.where(res >= threshold)

        # 使用灰度图像中的坐标对图像进行标记（画方框）
        for pt in zip(*loc[::-1]):
            # 过滤掉固定标记
            if is_sign(pt):
                picture_sign_num += 1
                continue
            cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 2)
            list.append(pt)
        # 显示图像

    if len(list) != 4:
        if picture_sign_num > 1:
            # plt.imshow(im)
            # plt.show()
            print(filename, ' process failed')
            if not os.path.exists(failed_dir):
                os.mkdir(failed_dir)
            cv2.imwrite(failed_dir + patient_no + '-' + os.path.splitext(filename)[0] + '.jpg', im)
        else:
            # plt.imshow(im)
            # plt.show()
            print(filename, ' data is wrong')
            if not os.path.exists(wrong_dir):
                os.mkdir(wrong_dir)
            cv2.imwrite(wrong_dir + patient_no + '-' + os.path.splitext(filename)[0] + '.jpg', img_orig)
    else:
        x_min, x_max, y_min, y_max = get_tophi_loc(list)
        # plt.imshow(img_orig[y_min:y_max, x_min:x_max])
        # plt.show()

        try:
            # 随机平移
            # x_min, x_max, y_min, y_max = translation(x_min, x_max, y_min, y_max)
            # 保存截取的结果
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            result_img = img_orig[y_min:y_max, x_min:x_max]
            result_img = cv2.resize(result_img, (dataread.ROWS, dataread.COLS), interpolation=cv2.INTER_CUBIC)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(result_dir + str(label) + patient_no + '_' + str(result_num) + '.jpg',
                        result_img)
            result_num += 1
        except:
            print(patient_no)


if __name__ == '__main__':
    data = pd.read_excel('2.xls', encoding='utf8')
    data = data.loc[:, ['num', 'label', 'dir', 'filename']]

    data = data.dropna(axis=0)

    for num, label, dir, filename in data.values:
        if os.path.exists(os.path.join(root_dir, dir)):
            for filename in os.listdir(os.path.join(root_dir, dir)):
                path = os.path.join(root_dir, dir, filename)
                generate_cut_img(num, label, path, filename)
