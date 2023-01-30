# block_classifier
라벨링된 논문 초록 내의 문장들을 학습하여 논문 초록 내의 문장의 라벨을 예측하는 모델을 학습합니다.

## 실행방법
먼저 data 디렉토리 내부의 "prepare_dataset.ipynb"를 전부 실행하여 데이터셋을 생성합니다.  

### 예시
- 데이터셋이 준비되면 다음과 같이 해당 모델을 학습 및 테스트할 수 있습니다.
```bash
python block_classifier.py --gpu_id 1 --data_model ignore-5 --model scibert --batch_size 16 --learning_rate 1e-5
```
- 라벨링 방법에 따라 --data_model을 바꾸어서 여러 데이터셋들을 학습 및 테스트해볼 수 있습니다. 자세한 것은 data 디렉토리를 참고하세요.
