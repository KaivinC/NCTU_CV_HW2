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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
# custom module
from util import *
import datetime
from engine import train_one_epoch, val_one_epoch, test_one_epoch
from torchsummary import summary
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda")


def get_args():
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--image_size", type=int, default=128,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--expname", type=str,
                        default='record', help="experiment name")
    parser.add_argument("--io", type=str, default="/run.log")
    parser.add_argument(
        "--resume",
        type=str,
        default="./checkpoint/14_resnet34/model_32_0.6379526853561401.pth")

    args = parser.parse_args()

    return args


def main(args):
    torch.cuda.manual_seed(1)

    # load dataset
    datasets = TestDataset(args.image_size)
    data_loader = DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1)

    # load model
    model = EfficientDet(num_classes=10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # resume
    if args.resume:
        print('Resume model')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    test_one_epoch(model, data_loader)


if __name__ == "__main__":
    args = get_args()
    main(args)
