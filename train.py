#!/usr/bin/env python3

import os
import sys
import re
import time
import collections
import argparse
import random
import pdb
import debug
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import tensorboardX as tb
from torchvision import transforms
from torch.utils.data import DataLoader

import darktorch
from darktorch.nn import Darknet
from darktorch.nn.loss import RegionLossV2, RegionLossV3
from darktorch.transforms import RandomHorizontalFlip
from darktorch.transforms import LetterboxTrain
from darktorch.transforms import CollateTransform
from darktorch.transforms import BGR2RGB, CustomToTensor, HWC2CHW, Int2Float
from darktorch.utils.data import MultiScaleSampler
from darktorch.utils.data import ListDataset
from darktorch.utils import LRScheduler
from darktorch.utils import darknet_from_cfg
from darktorch.utils import parse_data, parse_configuration, parse_categories


def parse_args():
    parser = argparse.ArgumentParser('DarkTorch train')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--clipping-norm', type=int, default=100)
    parser.add_argument('--cfg', type=str, default='cfg/yolov2-voc.cfg')
    parser.add_argument('--data', type=str, default='cfg/voc.data')
    parser.add_argument('--no-shuffle', action='store_true', default=False)
    parser.add_argument('--nonrandom', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--once', action='store_true', default=False)
    parser.add_argument(
        '--weights', type=str, default='weights/darknet19_448.conv.23')
    args = parser.parse_args()
    args.use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    return args


def to_voc_label_path(im_path):
    path = im_path.replace('images', 'labels')
    path = path.replace('JPEGImages', 'labels')
    path = path.replace('.jpg', '.txt')
    return path


args = parse_args()
device = torch.device('cuda' if args.use_cuda else 'cpu')
shuffle = (not args.no_shuffle)
drop_last = True

# Use Darktorch helper function to initialize the model
model = darknet_from_cfg(args.cfg)
model.load_weights(args.weights)
model.to(device)
print(model)

# Optional fintuning
if args.finetune:
    for index, param in enumerate(model.parameters()):
        if index < 23:
            param.requires_grad = False


## Transforms
hprob = 0 if args.nonrandom == True else 0.5
horizontal_flip = RandomHorizontalFlip(p=hprob)
collate_transform = CollateTransform()

letterbox_transform = LetterboxTrain(
    jitter=model.jitter, w=model.width, h=model.height, nonrandom=args.nonrandom)

color_shift = transforms.ColorJitter(
    hue=model.hue, saturation=model.saturation, brightness=model.exposure)

transform = transforms.Compose(
    [Int2Float(), letterbox_transform, BGR2RGB(), HWC2CHW(), CustomToTensor()])

target_transform = transforms.Compose(
    [letterbox_transform, collate_transform])


# Parse Darknet configuration files
data_cfg = parse_data(args.data)
categories = parse_categories(data_cfg['names'])
net_cfg = parse_configuration(args.cfg)

data_train = ListDataset(
    data_cfg['train'],
    fp_transform_fn=to_voc_label_path,
    transform=transform,
    target_transform=target_transform,
    shuffle_labels=shuffle)

sampler = MultiScaleSampler(data_train, shuffle)

v = 3 if 'v3' in args.cfg else 2
if v == 2:
    criterion = RegionLossV2(
        model.priors.reshape(-1).tolist(),
        threshold=model.threshold,
        n_classes=model.num_classes)

elif v == 3:
    criterion = RegionLossV3(
        model.priors.reshape(-1).tolist(),
        threshold=model.threshold,
        n_classes=model.num_classes)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=model.learning_rate,
    weight_decay=model.weight_decay,
    momentum=model.momentum)

train_loader = DataLoader(
    data_train,
    batch_size=model.batch_size,
    pin_memory=args.use_cuda,
    drop_last=drop_last,
    num_workers=args.num_workers,
    sampler=sampler)#, collate_fn=darktorch.utils.data.shit_collate)

optim_dict = args.weights.replace(
    '.pt',
    '.optim.pt') if '.pt' in args.weights else args.weights + '.optim.pt'

if os.path.isfile(optim_dict):
    optimizer.load_state_dict(torch.load(optim_dict))

lr_scheduler = LRScheduler(optimizer, model.learning_rate, model.steps,
                           model.scales, model.burn_in)


def train(epoch, iteration):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        n_batches = iteration + batch_idx
        lr_scheduler.update(n_batches)

        optimizer.zero_grad()
        total_loss = 0.0

        for index in range(model.subdivisions):
            p0 = index * (model.batch_size // model.subdivisions)
            p1 = p0 + (model.batch_size // model.subdivisions)

            job_data = data[p0:p1].to(device)
            job_target = target[p0:p1].to(device)
            darktorch.utils.write_tensor(job_data.detach().cpu(), 'input.binlog', True, False)
            output = model(job_data)
            darktorch.utils.write_tensor(output.detach().cpu(), 'output.binlog', True, False)
            #sys.exit(0)
            loss = criterion(output, job_target)
            print(loss.item())

            total_loss += loss.item()
            loss.backward()
           

        total_loss = total_loss / model.batch_size
        nn.utils.clip_grad_norm_(model.parameters(), args.clipping_norm)
        optimizer.step()

        print('iter:{} epoch:{} loss:{}'.format(n_batches, epoch, total_loss))
        writer.add_scalar('Cost', total_loss, n_batches)

        if (n_batches % 1000 == 0
                or (n_batches % 100 == 0 and n_batches < 1000)):

            weights_dir = data_cfg['backup']
            os.makedirs(weights_dir, exist_ok=True)
            fp = os.path.join(weights_dir,
                              'yolov2-voc.' + str(n_batches) + '.pt')
            model.save_weights(fp)
            torch.save(optimizer.state_dict(), fp.replace('.pt', '.optim.pt'))
        
        if args.once == True:
            return 1

    return len(train_loader)


ip_epoch = len(train_loader)  # iterations per epoch
max_epochs = (model.max_batches // ip_epoch) + 1
n_iters = model.samples_processed // model.batch_size
start_epoch = n_iters // ip_epoch + 1

# Calculate samples needed to complete the current epoch
if n_iters > 0 and n_iters % ip_epoch != 0:
    index = (n_iters // ip_epoch) + 1
    ic_epoch = index * ip_epoch - n_iters  # Iterations to complete the current epoch
    n_complete_epoch = ic_epoch * model.batch_size
    sampler.complete_epoch_samples(n_complete_epoch)

with tb.SummaryWriter() as writer:
    for epoch in range(start_epoch, max_epochs + 1):
        n_iters += train(epoch, n_iters + 1)

