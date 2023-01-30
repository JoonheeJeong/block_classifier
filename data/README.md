# about dataset

### 이용방법
- "prepare_dataset.ipynb"을 실행하면 필요한 모든 데이터셋이 생성됩니다.
- paser 인자 --data_model에서 입력받은 정보를 바탕으로 block_classifier.py의 클래스 BlockDataModule의 함수 _get_fns에서 해당하는 데이터셋을 읽도록 되어있습니다. 가능한 데이터셋의 종류 (--data_model의 값)은 다음과 같습니다.
  - base
  - ignore
  - replace
  - ignore-56
  - replace-56
  - ignore-5
- 그밖에 자세한 내용은 "prepare_dataset.ipynb"의 주석 및 "reduce_tag.py"를 참고하시기 바랍니다.

### 추가 설명
- 기본적인 데이터의 라벨은 석사 김현지 분의 문서를 참고하면 됩니다.
- 라벨의 유형들을 조작할 때 성능 차이를 확인하고 싶어서 다양한 방식으로 라벨링을 하여 데이터셋을 생성하도록 했습니다. "prepare_dataset.ipynb"의 주석 및 "reduce_tag.py"를 참고하시기 바랍니다.
- +) 추가적으로 현재 문장이 주어질 때 다음 문장의 태그를 예측하는 모델을 위한 수정된 데이터셋도 함께 생성됩니다. 위에서 수정된 각 데이터셋에 추가적인 파일 접미사로 _reformed가 붙습니다. "prepare_dataset.ipynb"의 주석 및 "reform_total.py"를 참고하시기 바랍니다. 
  - 이 모델을 이용할 때는 parser 인자 --task의 값으로 predict_next_structure를 입력하면 됩니다.
  - 다만 본 실험은 미비한 성능을 보여 가설 검증이 실패한, 이용하기에 적합하지 않은 모델입니다. 참고만 하시기 바랍니다.

