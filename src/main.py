# src/main.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
sys.path.append(MODELS_DIR)

try:
    from data_split import split_and_save_dataset, ORIGINAL_CSV_FILE as SPLIT_ORIGINAL_CSV
    from train import train_model
    from evaluate import evaluate_model

except ImportError as e:
    print(f"모듈 로드 오류: {e}")
    sys.exit(1)

def run_full_pipeline():
    """
    데이터 전처리, 학습, 평가 과정을 순차적으로 실행하는 함수.
    """

    # 1. 데이터 분할 단계 실행
    print("\n--- Data Split ---")
    
    try:
        split_and_save_dataset(SPLIT_ORIGINAL_CSV)

    except FileNotFoundError as e:
        print("Failed to split data")
        return

    # 2. 모델 학습 단계 실행
    train_model()

    # 3. 모델 평가 단계 실행
    evaluate_model()

if __name__ == '__main__':
    # 메인 함수 실행
    run_full_pipeline()