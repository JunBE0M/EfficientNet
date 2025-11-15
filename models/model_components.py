# model_components.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models import create_model
import pandas as pd
from PIL import Image
import os
import numpy as np

NUM_CLASSES = 5 
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet 표준 정규화 평균.
STD = [0.229, 0.224, 0.225]   # ImageNet 표준 정규화 표준편차



class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_col = self.labels_frame.columns[0] # 파일명 컬럼명

    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        # 파일 경로 설정 및 이미지 로드
        img_name = self.labels_frame.iloc[idx][self.img_col]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 5개 질병 label 로드
        label = self.labels_frame.iloc[idx, 1:].values.astype('float32') 
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

# Efficient model 생성

def get_efficientnet_model(model_name='efficientnet_b4', num_classes=NUM_CLASSES, pretrained=True):
    model = create_model(
        model_name, 
        pretrained=pretrained,
        num_classes=num_classes 
    )
    return model

# 데이터 전처리. Transform
def get_transforms():
    """훈련 및 검증/테스트용 이미지 전처리 객체 반환"""
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE), # Data Augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    return transform_train, transform_val_test

# Dataloader 함수
def get_data_loaders(img_dir, batch_size=32):
    transform_train, transform_val_test = get_transforms()


    CSV_PATH_BASE = '../data/'
    
    train_dataset = ChestXrayDataset(csv_file=os.path.join(CSV_PATH_BASE, 'train_labels.csv'), img_dir=img_dir, transform=transform_train)
    val_dataset = ChestXrayDataset(csv_file=os.path.join(CSV_PATH_BASE, 'val_labels.csv'), img_dir=img_dir, transform=transform_val_test)
    test_dataset = ChestXrayDataset(csv_file=os.path.join(CSV_PATH_BASE, 'test_labels.csv'), img_dir=img_dir, transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"데이터 로더 준비 완료. 배치 크기: {batch_size}")
    return train_loader, val_loader, test_loader