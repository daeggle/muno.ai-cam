import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, img_path, gt_path, file_names, size_x, size_y):
        self.img_path = img_path
        self.gt_path = gt_path
        self.file_names =file_names
        self.size_x = size_x
        self.size_y = size_y
    def __getitem__(self, index):
        img = cv2.imread(self.img_path+self.file_names[index][:-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST)
        img = ~img
        img = np.asarray(img, dtype=float)
        img = img / 255 
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.FloatTensor(img)
        
        gt = cv2.imread(self.gt_path+self.file_names[index][:-1])
        gt = cv2.resize(gt, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST)
        gt = gt[:,:,0]
        # gt = np.transpose(gt, (2, 0, 1))
        gt_tensor = torch.LongTensor(gt)
        
        return img_tensor, gt_tensor
    def __len__(self):
        return len(self.file_names)
