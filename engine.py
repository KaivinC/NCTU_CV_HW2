import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import *
from src.model import EfficientDet
import shutil
import numpy as np
# custom module
from util import *
import datetime
from draw_picture import draw
from PIL import Image
import copy
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
cpu_device = torch.device("cpu")


# train one epoch
def train_one_epoch(model, train_loader, optimizer, args):
    clsLoss = AverageMeter()
    regLoss = AverageMeter()
    Loss = AverageMeter()
    model.train()

    epoch_loss = []
    for i, sample in enumerate(train_loader):
        image = sample['img']
        annotation = sample['annot']
        batch_size = image.shape[0]

        optimizer.zero_grad()
        image = image.to(device).float()
        annotation = annotation.to(device)
        cls_loss, reg_loss = model([image, annotation])

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        clsLoss.update(cls_loss, batch_size)
        regLoss.update(reg_loss, batch_size)
        Loss.update(loss, batch_size)

        print('[{0}/{1}]  '
              'Loss {Loss.val:.4f} ({Loss.avg:.4f})  '
              'clsLoss {clsLoss.val:.4f} ({clsLoss.avg:.4f})  '
              'regLoss {regLoss.val:.4f} ({regLoss.avg:.4f})         '
              .format(i + 1, len(train_loader),
                      Loss=Loss,
                      clsLoss=clsLoss,
                      regLoss=regLoss), end='\r')

    print(' ' * 100, end='\r')

# val one epoch
def val_one_epoch(model, val_loader, args):
    clsLoss = AverageMeter()
    regLoss = AverageMeter()
    Loss = AverageMeter()

    model.eval()
    loss_regression_ls = []
    loss_classification_ls = []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['img']
            annotation = sample['annot']
            batch_size = image.shape[0]

            image = image.to(device).float()
            annotation = annotation.to(device)

            cls_loss, reg_loss = model([image, annotation])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()

            clsLoss.update(cls_loss, batch_size)
            regLoss.update(reg_loss, batch_size)
            Loss.update(cls_loss + reg_loss, batch_size)

            print('[{0}/{1}]  '
                  'Loss {Loss.val:.4f} ({Loss.avg:.4f})  '
                  'clsLoss {clsLoss.val:.4f} ({clsLoss.avg:.4f})  '
                  'regLoss {regLoss.val:.4f} ({regLoss.avg:.4f})  '
                  .format(i + 1, len(val_loader),
                          Loss=Loss,
                          clsLoss=clsLoss,
                          regLoss=regLoss), end='\r')

    print(' ' * 100, end='\r')

    return Loss.avg


# test one epoch
def test_one_epoch(model, data_loader):
    itr = 0
    results = []
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(data_loader):
            itr += 1
            image = images.to(device).float()

            batch_size = image.shape[0]
            result = {}

            [nms_scores, nms_class, transformed_anchors] = model([image])

            nms_scores = nms_scores.to(cpu_device).tolist()
            nms_class = nms_class.to(cpu_device).tolist()
            transformed_anchors = transformed_anchors.to(cpu_device).tolist()

            bbox = transformed_anchors
            result["bbox"] = transformed_anchors
            bbox = np.array(bbox)
            result["bbox"] = np.array(result["bbox"])
            result["score"] = nms_scores
            result["label"] = nms_class
            original_img = Image.open("./data/test/" + str(itr) + ".png")
            width, heigh = original_img.size

            for i in range(len(result["label"])):
                if result["label"][i] == 0:
                    result["label"][i] = 10

            processed_img = image.to(cpu_device)
            processed_img = np.array(processed_img[0])

            if(bbox.shape[0] > 0):
                if(heigh > width):
                    result["bbox"][:, 0] = copy.deepcopy(
                        bbox[:, 1] * (width / processed_img.shape[2]))
                    result["bbox"][:, 2] = copy.deepcopy(
                        bbox[:, 3] * (width / processed_img.shape[2]))
                    result["bbox"][:, 1] = copy.deepcopy(
                        bbox[:, 0] * (width / processed_img.shape[2]))
                    result["bbox"][:, 3] = copy.deepcopy(
                        bbox[:, 2] * (width / processed_img.shape[2]))
                else:
                    result["bbox"][:, 0] = copy.deepcopy(
                        bbox[:, 1] * (heigh / processed_img.shape[1]))
                    result["bbox"][:, 2] = copy.deepcopy(
                        bbox[:, 3] * (heigh / processed_img.shape[1]))
                    result["bbox"][:, 1] = copy.deepcopy(
                        bbox[:, 0] * (heigh / processed_img.shape[1]))
                    result["bbox"][:, 3] = copy.deepcopy(
                        bbox[:, 2] * (heigh / processed_img.shape[1]))

            result["bbox"] = (result["bbox"].tolist())
            results.append(result)

            print("itr: %d " % itr, end="\r")

        with open("result.json", "w") as f:
            f.write(str(results))
            f.close()
