import tensorflow as tf
import cv2,os
import matplotlib.pyplot as plt


read_dir = 'all_result/result/'

# 随机剪切
def randomcrop(filename, result_dir, suffix):
    img = cv2.imread(os.path.join(read_dir, filename))
    # 将图片进行随机裁剪为280×280
    crop_img = tf.random_crop(img, [192, 192, 3])
    # 将图片由BGR转成RGB
    crop_img = crop_img.eval()
    # 保存裁剪后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + str(suffix) + '.jpg',
                crop_img)

# 图片翻转，上下，左右，对角
def flip(filename, result_dir):
    img = cv2.imread(os.path.join(read_dir, filename))
    # 将图片进行左右翻转
    l_r_img = tf.image.flip_left_right(img)
    l_r_img = l_r_img.eval()
    # 保存左右翻转后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + '_l-r.jpg',
                l_r_img)
        
                # 将图片进行上下翻转
                l_r_img = tf.image.flip_up_down(img)
                l_r_img = l_r_img.eval()
                # 保存上下翻转后的图片
                cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + '_u-d.jpg',
                            l_r_img)

NUM = 3
if __name__ == "__main__":
    result_dir = 'all_result/result/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    # with tf.Session() as sess:
    #     for filename in os.listdir(read_dir):
    #         for suffix in range(1,NUM + 1):
    #             randomcrop(filename, result_dir, suffix)
    #             print(filename + ' -------OK!!!')
    
    with tf.Session() as sess:
        for filename in os.listdir(read_dir):
            flip(filename, result_dir)
            print(filename + ' -------OK!!!')
    sess.close()

