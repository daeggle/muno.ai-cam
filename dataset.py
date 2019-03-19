import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset): # pytorch의 Dataset 클래스 상속
    def __init__(self, img_path, gt_path, file_names, size_x, size_y):  # 클래스 생성자
        self.img_path = img_path                                        # 이미지 폴더 위치
        self.gt_path = gt_path                                          # ground true 폴더 위치
        self.file_names =file_names                                     # 사용할 파일 이름 리스트
        self.size_x = size_x                                            # resize x의 크기
        self.size_y = size_y                                            # resize y의 크기
    def __getitem__(self, index): # index번째 tensor를 불러오는 함수 오버라이딩
        img = cv2.imread(self.img_path+self.file_names[index][:-1])     # cv2로 path의 이미지를 로딩, 코드가 미흡해서 '\n'을 못지워 -1 슬라이싱 필요
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     # BGR을 GRAY로 변환
        img = cv2.resize(img, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST) # 지정된 크기로 resize
        img = ~img                                                      # 비트 반전
        img = np.asarray(img, dtype=float)                              # float 형 numpy 배열로 변경
        img = img / 255                                                 # 0 ~ ! 사이 값을 가지도록 255로 나눗셈
        img = np.expand_dims(img, axis=0)                               # (1, H, W)가 되도록 차원 확장
        img_tensor = torch.FloatTensor(img)                             # FloatTensor로 변환
        
        gt = cv2.imread(self.gt_path+self.file_names[index][:-1])       # 같은 방식으로 ground true 주소에서 이미지 로딩
        gt = cv2.resize(gt, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST) # resize인데 segmentation이므로 NEAREST 보간을 사용해야 함
        gt = gt[:,:,0]                                                  # 한개의 채널만 있으면 됨
        gt_tensor = torch.LongTensor(gt)                                # pytorch의 cross-entropy 함수를 사용하려면 target이 long타입이여야 함
        
        return img_tensor, gt_tensor
    def __len__(self):                                                  # 크기를 반환하는 함수 오버라이딩
        return len(self.file_names)
