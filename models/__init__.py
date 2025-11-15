'''
model_components.py
   - 역할: 모델 구조(EfficientNet 등), 데이터셋(Dataset), 데이터 로더(DataLoader),
         이미지 전처리(Transforms) 등 재사용 가능한 핵심 구성 요소를 정의
   - 기능: 모델을 생성하고 데이터를 로드하는 기반을 제공

main_train.py
   - 역할: 모델 학습(Training) 파이프라인의 메인 로직을 담당
   - 기능: 모델 초기화, 옵티마이저/손실 함수 설정, 에폭(Epoch) 반복,
         검증(Validation) 및 최고 성능 체크포인트 저장 로직을 포함

evaluate_model.py
   - 역할: 학습된 모델의 성능을 테스트 셋을 사용해 평가(Evaluation)하는 로직을 담당
   - 기능: 체크포인트를 로드하여 최종 AUC-ROC, 정확도 등의 지표를 계산하고 출력
'''