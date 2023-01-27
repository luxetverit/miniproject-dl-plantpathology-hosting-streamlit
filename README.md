# Plant Pathology
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/uzunb/house-prices-prediction-lgbm/main/1_%F0%9F%92%BB_Enter_Page.py)

## Description
사과나무 질병 예측은 Kaggle Plant Pathology 2020 데이터 세트를 사용하여 개발되었습니다.

## 데이터
The dataset is available at [Kaggle](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7).

## 목표
이 프로젝트의 목표는 apple tree 잎사귀의 질병 유무를 확인하는 것입니다.

## 특징
데이터 세트 특징~~~~~~~ 몇개인지
시각화자료~~

## 설정
```
이미지 변환 : 사이즈, 대비, 밝기, 수평, 수직, 회전, 정규화
에폭 : 20회
배치사이즈 : 4
이미지 크기 : 224 x 224
```

## 모델
### 모델
이 모델은 ResNet50 신경망을 기본으로 합니다.

```python
model = models.resnet50(pretrained = True)
for param in model.parameters():
    param.require_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,512,bias=True),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(512,4, bias = True))

model = model.to(device)
```

### 손실함수 및 옵티마이저
```python
import torch.nn as nn
from torch import optim

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
```

### 평가
```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print('Epoch : {}/{}...'.format(epoch+1, epochs),
                      'Train Loss : {:.3f} / '.format(train_loss/len(trainloader)),
                      'Valid Loss : {:.3f} / '.format(valid_loss/len(validloader)),
                      'Valid AUC : {:.3f} / '.format(valid_auc),
                      'Valid Accuracy : {:.3f}'.format(valid_accuracy))
```

## 훈련 및 결과

