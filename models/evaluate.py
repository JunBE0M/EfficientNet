import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models import create_model
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

# 경로 설정
IMG_FOLDER_PATH = '../data/images' 
CHECKPOINT_DIR = '../checkpoints' # 체크포인트 로드 폴더
TEST_CSV = '../data/test_labels.csv' 

NUM_CLASSES = 5 
IMAGE_SIZE = 224
MODEL_NAME = 'efficientnet_b4' 
BATCH_SIZE = 32              
LABELS = ['Edema', 'Effusion', 'Mass', 'Nodule', 'Pneumothorax']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_col = self.labels_frame.columns[0]
        self.labels_np = self.labels_frame.iloc[:, 1:].values.astype('float32')
    
    def __len__(self): return len(self.labels_frame)
    
    def __getitem__(self, idx):
        img_name = self.labels_frame.iloc[idx][self.img_col]
        img_path = os.path.join(self.img_dir, img_name)
        # 이미지 파일명을 확인하고, 파일이 없을 경우 예외 처리
        if not os.path.exists(img_path):
            # 파일이 없으면 에러를 발생시키거나 건너뛸 수 있으나, 여기서는 에러를 발생시켜 디버깅을 돕습니다.
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = self.labels_np[idx] 
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label)


def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(IMAGE_SIZE), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    # ⭐ 테스트/검증에 사용되는 전처리 (RandomCrop 미사용)
    transform_val_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transform_train, transform_val_test

def get_data_loaders(img_dir, test_csv, batch_size):

    _, transform_val_test = get_transforms()
    
    test_dataset = ChestXrayDataset(csv_file=test_csv, img_dir=img_dir, transform=transform_val_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    
    return test_loader

def get_efficientnet_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False):

    # pretrained=False로 설정하여 가중치 파일로 모델을 초기화
    model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

# 모델 평가 함수
def evaluate_model():
    print("\n테스트 평가 시작")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"평가 장치: {device}")
    
    # Test Dataloader
    test_loader = get_data_loaders(IMG_FOLDER_PATH, TEST_CSV, BATCH_SIZE)
    print(f"테스트 데이터 로드 완료. (총 {len(test_loader.dataset)}개)")
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_5class.pth")
    
    # Model load
    model = get_efficientnet_model(pretrained=False)
    
    if not os.path.exists(checkpoint_path):
        print(f"\n체크포인트 파일을 찾을 수 없음. {checkpoint_path}")
        return

    # 체크포인트 로드 및 가중치 적용

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"모델 로드 완료.")

    all_probabilities = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images) 
            probabilities = torch.sigmoid(logits) 

            all_probabilities.append(probabilities.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

    probabilities_np = np.concatenate(all_probabilities, axis=0)
    true_labels_np = np.concatenate(all_true_labels, axis=0)

    # ROC 계산
    avg_auc = roc_auc_score(true_labels_np, probabilities_np, average='macro')
    print(f"\n=======================================================")
    print(f"🌟 최종 테스트 AUC-ROC (Macro Avg): {avg_auc:.4f}")
    print(f"=======================================================")
        
    # 클래스별 AUC-ROC
    print("\n[클래스별 AUC-ROC]")
    for i, label in enumerate(LABELS):
        auc_i = roc_auc_score(true_labels_np[:, i], probabilities_np[:, i])
        print(f"   - {label}: {auc_i:.4f}")

    # 정확도
    threshold = 0.5
    predictions_np = (probabilities_np > threshold).astype(int)
    subset_accuracy = accuracy_score(true_labels_np, predictions_np)
    hamming_accuracy = np.mean(predictions_np == true_labels_np)

    print(f"\n[정확도 지표 (임계값 {threshold})]")
    print(f"   - Subset Accuracy (완벽 일치): {subset_accuracy:.4f}")
    print(f"   - Label Accuracy (개별 라벨 일치): {hamming_accuracy:.4f}")


if __name__ == '__main__':
    evaluate_model()