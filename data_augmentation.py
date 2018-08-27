import tensorflow as tf
import cv2,os
import matplotlib.pyplot as plt

# 随机剪切
def randomcrop(filename, result_dir, suffix, crop):
    image = cv2.imread(os.path.join(read_dir, filename))
    # 将图片由BGR转成RGB
    crop_img = crop.eval(feed_dict = {img : image})
    # 保存裁剪后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + str(suffix) + '.jpg',
	        crop_img)

# 图片翻转，上下，左右，对角
def flip(filename, result_dir, lr_op, ud_op, lrud_op):
    image = cv2.imread(os.path.join(read_dir, filename))
    # 将图片进行左右翻转
    l_r_img = lr_op.eval(feed_dict = {img : image})
    # 保存左右翻转后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + '_l-r.jpg',
	        l_r_img)

    # 将图片进行上下翻转
    u_d_img = ud_op.eval(feed_dict = {img : image})
    # 保存上下翻转后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + '_u-d.jpg',
                u_d_img)

    # 将图片进行先左右再上下翻转
    l_r_u_d_img = lrud_op.eval(feed_dict = {img : image})
    # 保存上下翻转后的图片
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + 'l_r_u-d.jpg',
                l_r_u_d_img)

def changehue(filename, result_dir):
    img = cv2.imread(os.path.join(read_dir, filename))
    adjusted1 = tf.image.adjust_hue(img, 0.2)
    adjusted2 = tf.image.adjust_hue(img, 0.6)
    # 将图片的饱和度-5。
    adjusted3 = tf.image.adjust_saturation(img, -5)
    # 将图片的饱和度+5。
    adjusted4 = tf.image.adjust_saturation(img, 5)

    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + 'a1.jpg',
                adjusted1.eval())
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + 'a2.jpg',
                adjusted2.eval())
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + 'a3.jpg',
                adjusted3.eval())
    cv2.imwrite(result_dir + os.path.splitext(filename)[0] + '_' + 'a4.jpg',
                adjusted4.eval())


NUM = 3
if __name__ == "__main__":
    read_dir = './data/new/result/result-nx/'
    result_dir1 = './data/new/result/augment-nx/'
    result_dir2 = './data/new/result/augment-nx/'
    if not os.path.exists(result_dir1):
        os.mkdir(result_dir1)
    if not os.path.exists(result_dir2):
        os.mkdir(result_dir2)

    img = tf.placeholder(tf.float32, [256, 256, 3])
    crop = tf.random_crop(img, [192, 192, 3])

    lr_op = tf.image.flip_left_right(img)
    ud_op = tf.image.flip_up_down(img)
    lrud_op = tf.image.transpose_image(img)

    with tf.Session() as sess:
        for filename in os.listdir(read_dir):
            for suffix in range(1,NUM + 1):
                randomcrop(filename, result_dir1, suffix, crop)
                flip(filename, result_dir2, lr_op, ud_op, lrud_op)
                print(filename + ' -------OK!!!')

    # with tf.Session() as sess:
    #     for filename in os.listdir(read_dir):
    #         changehue(filename, result_dir)
    #         print(filename + ' -------OK!!!')
    sess.close()
