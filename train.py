#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MyDataset
from unet import UNet

import numpy as np

dataset_path = 'deepscores/deep_scores_v2_100p/images_png/'
gt_path = 'deepscores/deep_scores_v2_100p/pix_annotations_png/'

f = open("train_files.txt", 'r')
lines = f.readlines()
train_file_names =[]
for line in lines:
    train_file_names.append(line)
f.close()

batch_size = 2
epochs = 20
size_x = 512 #540 #1080
size_y = 768 #960 #1920

train_dataset = MyDataset(dataset_path, gt_path, train_file_names, size_x, size_y)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=159, padding=True, up_mode='upconv').to(device)
# model = nn.DataParallel(model)
optim = torch.optim.Adam(model.parameters())


for ep in range(epochs):
    loss_list = []
    for X, y in dataloader:
        X = X.to(device)  # [N, 1, H, W]
        y = y.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        loss_list.append(loss.item())
    print("[{}]:{}".format(ep,np.mean(loss_list)))
    
    torch.save(model.state_dict(), "pytorch_weight.pt")




