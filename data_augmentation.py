import tensorflow as tf
import cv2,os
import matplotlib.pyplot as plt


read_dir = 'all_result/result/'
result_dir = 'all_result/result-crop/'

def randomcrop(filename, suffix=1):
    img = cv2.imread(os.path.join(read_dir, filename))
    # 将图片进行随机裁剪为280×280
    crop_img = tf.random_crop(img, [192, 192, 3])

    sess = tf.InteractiveSession()
    # 将图片由BGR转成RGB
    crop_img = crop_img.eval()
    # 保存裁剪后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + str(suffix) + '.jpg',
                crop_img)
    sess.close()

NUM = 3
if __name__ == "__main__":
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for filename in os.listdir(read_dir):
        for suffix in range(1,NUM + 1):
            randomcrop(filename, suffix)
            print(filename + ' -------OK!!!')
