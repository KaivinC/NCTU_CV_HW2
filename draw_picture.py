import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from PIL import Image
import os

def draw(processed_img, itr, bbox, label, score):
    img = Image.open("./data/test/" + str(itr) + ".png")
    img.convert('RGB')
    width, heigh = img.size

    img = np.array(img)

    bbox = np.float32(bbox)

    if(bbox.shape[0] > 0):
        if(heigh > width):
            bbox[:, 0] = bbox[:, 0] * (width / processed_img.shape[2])
            bbox[:, 2] = bbox[:, 2] * (width / processed_img.shape[2])
            bbox[:, 1] = bbox[:, 1] * (width / processed_img.shape[2])
            bbox[:, 3] = bbox[:, 3] * (width / processed_img.shape[2])
        else:
            bbox[:, 0] = bbox[:, 0] * (heigh / processed_img.shape[1])
            bbox[:, 2] = bbox[:, 2] * (heigh / processed_img.shape[1])
            bbox[:, 1] = bbox[:, 1] * (heigh / processed_img.shape[1])
            bbox[:, 3] = bbox[:, 3] * (heigh / processed_img.shape[1])

    for i in range(bbox.shape[0]):
        cv2.rectangle(img,
                      (bbox[i][0], bbox[i][1]),
                      (bbox[i][2], bbox[i][3]),
                      (220, 0, 0), 1)
        cv2.putText(img,
                    text=str(label[i]),
                    org=(int(bbox[i][0] - 1),
                         int(bbox[i][1] - 1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.2,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    color=(0,
                           0,
                           255))
    if not os.path.exists("./picture"):
        os.mkdir("./picture")
    cv2.imwrite("./picture/" + str(itr) + "_img.png", img)
