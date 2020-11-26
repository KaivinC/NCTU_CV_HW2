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
import datetime
from engine import train_one_epoch, val_one_epoch

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


def get_args():
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--image_size", type=int, default=128,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--expname", type=str,
                        default='record', help="experiment name")
    parser.add_argument("--io", type=str, default="/run.log")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    return args


def main(args):
    torch.cuda.manual_seed(1)

    # Create chekpoint
    time = datetime.datetime.now()
    if not os.path.exists("./checkpoint"):
        os.mkdir("checkpoint")
    filename = str(time.month) + "_" + str(time.day) + "_" + str(time.hour)
    save_dir = os.path.join("./checkpoint", filename)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    datasets = {}
    data_loader = {}
    # load dataset
    datasets['train'] = TrainDataset(args.image_size)
    data_loader['train'] = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collater,
        num_workers=32)

    # prepare dataloader
    datasets['val'] = ValDataset(args.image_size)
    data_loader['val'] = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collater,
        num_workers=32)

    # load model
    model = EfficientDet(num_classes=10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # set optimizer and scheduler
    # if the loss can't decrease, use the SGD
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
    # weight_decay=1e-5)    scheduler = ReduceLROnPlateau(optimizer,
    # 'min',factor = 0.2, patience = 5,verbose = True,min_lr = 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 7, eta_min=1e-8, verbose=False)

    # resume
    if args.resume:
        print('Resume training')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    for epoch in range(args.num_epochs):
        # print and record lr
        for param_group in optimizer.param_groups:
            print('learning rate:', param_group['lr'])

        # train one epoch
        train_one_epoch(model, data_loader['train'], optimizer, args)

        # val one epoch
        loss = val_one_epoch(model, data_loader['val'], args)

        print("Svaing the model...")
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()},
                   os.path.join(save_dir,
                                'model_{}_{}.pth'.format(epoch,
                                                         loss)))
        print("Saving successfully!!")

        scheduler.step(loss)


if __name__ == "__main__":
    args = get_args()
    main(args)
