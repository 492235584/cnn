'''
1.均衡化前的直方图和累计直方图
2.均衡化后的直方图和累计直方图
均衡化的函数用的是opencv中的equalizaHist
计算直方图的函数用的是opencv中的calcHist
'''
import cv2,os
import numpy as np
import matplotlib.pyplot as plt

# 自适应灰度直方图均衡化处理
def hist(result_dir, filename):
    # 生成文件夹
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    im = cv2.imread(os.path.join(read_dir, filename))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    mri_max = np.amax(im)
    mri_min = np.amin(im)
    mri_img = ((im - mri_min) / (mri_max - mri_min)) * 255
    mri_img = mri_img.astype('uint8')

    r, c, h = mri_img.shape
    for k in range(h):
        temp = mri_img[:, :, k]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(temp)
        cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_hist.jpg', img)


# Laplace图像增强
def laplace(result_dir, filename):
    #生成文件夹
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    im = cv2.imread(os.path.join(read_dir, filename))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    dst = np.zeros(im.shape)
    dst = cv2.filter2D(im, im.shape[2], kernel, dst)
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_laplace.jpg', dst)

if __name__ == "__main__":
    read_dir = 'all_result/result/'

    for filename in os.listdir(read_dir):
        hist('all_result/result-hist/', filename)
        laplace('all_result/result-lap/', filename)
        print(filename + ' -------OK!!!')
