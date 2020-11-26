import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm


def normalize_img(img_h, img_w):
    means, stdevs = [], []
    img_list = []

    TRAIN_DATASET_PATH = './data/train'
    TEST_DATASET_PATH = './data/test'
    image_fns = glob(os.path.join(TRAIN_DATASET_PATH, '*.png'))
    image_fns = np.array(image_fns)
    image_fns = np.concatenate(
        [np.array(glob(os.path.join(TEST_DATASET_PATH, '*.png'))), image_fns])

    for single_img_path in tqdm(image_fns):
        img = cv2.imread(single_img_path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

    file = open("normalize_value.txt", "w")
    file.write("first line is normMean, second line is normStd\n")
    means_str = str(means)[1:-2].replace(" ", "")
    stdevs_str = str(stdevs)[1:-2].replace(" ", "")
    file.write(str(means_str) + "\n")
    file.write(str(stdevs_str) + "\n")


if __name__ == "__main__":
    normalize_img(128, 128)
