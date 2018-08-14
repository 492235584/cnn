import random, os, cv2
import numpy as np

#TRAIN_DIR = ['./result/', './result-crop/', './result-flip/']
TRAIN_DIR = ['./all_result/result/']
# TEST_DIR = './input/test/'
ROWS = 64 * 4
COLS = 64 * 4
CHANNELS = 3

def read():
    all_image = []
    for im_dir in TRAIN_DIR:
        all_image += [im_dir + filename for filename in os.listdir(im_dir)]

    # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
    random.shuffle(all_image)
    prct = 0
    train_num = int(len(all_image) * prct)
    train_img = all_image[0 : train_num]
    validation_img = all_image[train_num:]
    train = prep_data(train_img)
    validation = prep_data(validation_img)
    train_labels = get_labels(train_img)
    validation_labels = get_labels(validation_img)

    return (train,validation,train_labels,validation_labels)

# 按固定行列读取图片
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 500 == 0: print('Processed {} of {}'.format(i, count))

    return data

# 根据文件名生成label
def get_labels(image_path):
    labels = []
    for path in image_path:
        filename = os.path.split(path)[1]
        if filename[0] == '1':
            labels.append([[[1, 0]]])
        elif filename[0] == '2':
            labels.append([[[0, 1]]])
        else:
            print('label set failed')
            exit()

    return labels

if __name__=="__main__":
    read()
