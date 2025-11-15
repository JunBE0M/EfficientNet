# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil

# model_components.py에서 정의한 함수 및 클래스 import
from model_components import get_efficientnet_model, get_data_loaders 

MODEL_NAME = 'efficientnet_b4'
IMG_FOLDER_PATH = './data/images/'  # Image PNG 파일 모음
CHECKPOINT_DIR = '../checkpoint'
BATCH_SIZE = 32      # 32, 64로 튜닝해보기 
LEARNING_RATE = 1e-4 
NUM_EPOCHS = 15        

# Model 학습 함수

def train_model():
    # 장치 설정 (GPU A40 사용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 장치: {device}")

    # 데이터 로더 준비
    train_loader, val_loader, _ = get_data_loaders(IMG_FOLDER_PATH, BATCH_SIZE)

    model = get_efficientnet_model(model_name=MODEL_NAME)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # BCE Loss 사용
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam Optimzer 사용
    
    # 체크포인트 폴더 생성
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"체크포인트 폴더 생성: {CHECKPOINT_DIR}")

    best_val_loss = float('inf')
    
    # Train loop
    print("\n학습 시작")
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- 훈련 단계 ---
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validaiton
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
            
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 모델 가중치 저장.
            modelCheckpoint = {
                'epoch': epoch,
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            save_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_5class.pth")
            torch.save(modelCheckpoint, save_path)
        
    print("\n학습 완료")

if __name__ == '__main__':
    train_model()