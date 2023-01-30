# about dataset

### 이용방법
"prepare_dataset.ipynb"을 실행하면 필요한 모든 데이터셋이 생성됩니다.

### 데이터셋 종류
- 기본적인 데이터의 라벨은 석사 김현지 분의 문서를 참고하면 됩니다.
- 라벨의 유형들을 조작할 때 성능 차이를 확인하고 싶어서 다양한 방식으로 라벨링을 하여 데이터셋을 생성하도록 했습니다. "prepare_dataset.ipynb"의 주석 및 "reduce_tag.py"를 참고하면 됩니다.
- 현재 문장이 주어질 때 다음 문장의 태그를 예측하는 모델을 위하여는 각 데이터셋에 추가적으로 파일 접미사로 _reformed가 붙습니다. "prepare_dataset.ipynb"의 주석 및 "reform_total.py"를 참고하면 됩니다.

