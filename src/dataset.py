from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
import cv2
import draw_picture
import copy

Normalizer = Normalize(mean=[0.44032887, 0.43167493, 0.4340605],
                       std=[0.21059683, 0.21733317, 0.2195521])
l = 33402

# pre-process image


def resize_image(x, y, SIZE):
    [H, W, C] = x.shape
    if H > W:
        scale = SIZE / W
        _H = int(H * scale) + 1
        _W = SIZE
        image = cv2.resize(x, (_W, _H))
        cropH = random.randint(0, (_H - SIZE + 1) // 2)
        image = image[cropH: cropH + SIZE]
        h = cropH
        w = 0
    else:
        scale = SIZE / H
        _H = SIZE
        _W = int(W * scale) + 1

        image = cv2.resize(x, (_W, _H))
        cropW = random.randint(0, (_W - SIZE + 1) // 2)
        image = image[:, cropW: cropW + SIZE]
        h = 0
        w = cropW

    annot = y.copy()
    annot[:, :4] = annot[:, :4] * scale
    annot[:, [0, 2]] = annot[:, [0, 2]] - w
    annot[:, [1, 3]] = annot[:, [1, 3]] - h

    return image, annot


class TrainDataset(Dataset):
    def __init__(self, SIZE):
        self.SIZE = SIZE
        self.x = ['data/train/' +
                  str(i + 1) + '.png' for i in range(int(l * 0.2), l)]
        self.y = ['data/train/train_label/' +
                  str(i + 1) + '.npy' for i in range(int(l * 0.2), l)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        SIZE = self.SIZE

        x = cv2.imread(self.x[item])
        y = np.load(self.y[item])

        image, annot = resize_image(x, y, SIZE)

        T_image = np.transpose(image, (2, 0, 1)) / 255.
        T_image = torch.tensor(T_image)

        annot = torch.tensor(annot)

        xx = Normalizer(T_image)
        sample = {'img': xx, 'annot': annot}

        return sample


class ValDataset(Dataset):
    def __init__(self, SIZE):
        self.SIZE = SIZE
        self.x = ['data/train/' +
                  str(i + 1) + '.png' for i in range(0, int(l * 0.2))]
        self.y = ['data/train/train_label/' +
                  str(i + 1) + '.npy' for i in range(0, int(l * 0.2))]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        SIZE = self.SIZE

        x = cv2.imread(self.x[item])
        y = np.load(self.y[item])

        image, annot = resize_image(x, y, SIZE)

        T_image = np.transpose(image, (2, 0, 1)) / 255.
        T_image = torch.tensor(T_image)

        annot = torch.tensor(annot)

        xx = Normalizer(T_image)
        sample = {'img': xx, 'annot': annot}

        return sample


def collater(data):
    imgs = [s['img'] for s in data]
    imgs = torch.stack(imgs)

    annots = [s['annot'] for s in data]

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'img': imgs, 'annot': annot_padded}


class TestDataset(Dataset):
    def __init__(self, SIZE):
        self.SIZE = SIZE
        self.x = ['data/test/' + str(i + 1) + '.png' for i in range(13068)]
        self.scale = [None] * 13068

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        SIZE = self.SIZE
        if isinstance(self.x[item], str):
            self.x[item] = cv2.imread(self.x[item])
            image = cv2.resize(self.x[item], (SIZE, SIZE))

        H, W, C = self.x[item].shape

        if H > W:
            scale = SIZE / W
            _H = int(H * scale) + 1
            _W = SIZE
            image = cv2.resize(self.x[item], (_W, _H))
            m = 32 - (_H % 32)
            image = np.pad(image, ((0, m), (0, 0), (0, 0)), 'constant')
        else:
            scale = SIZE / H
            _H = SIZE
            _W = int(W * scale) + 1
            image = cv2.resize(self.x[item], (_W, _H))
            m = 32 - (_W % 32)
            image = np.pad(image, ((0, 0), (0, m), (0, 0)), 'constant')

        image = np.transpose(image, (2, 0, 1)) / 255.
        image = torch.tensor(image)
        image = Normalizer(image)

        return image
