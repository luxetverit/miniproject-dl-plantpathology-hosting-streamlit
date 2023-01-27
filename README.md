# Plant Pathology
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/uzunb/house-prices-prediction-lgbm/main/1_%F0%9F%92%BB_Enter_Page.py)

## Description
사과나무 질병 예측은 Kaggle Plant Pathology 2020 데이터 세트를 사용하여 개발되었습니다.

## 데이터
The dataset is available at [Kaggle](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7).

## 목표
이 프로젝트의 목표는 apple tree 잎사귀의 질병 유무를 확인하는 것입니다.

## 특징
데이터 세트 특징~~~~~~~ 
시각화자료~~

## 모델
### 모델
ResNet50

```
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
### 이미지 변환
```
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_train = A.Compose([
    A.Resize(224, 224),
    A.RandomBrightnessContrast(brightness_limit=0.2, # 밝기 대비 조절
                               contrast_limit=0.2, p=0.3),
    A.VerticalFlip(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=30, p = 0.3),
    A.OneOf([A.Emboss(p=1), # 양각화, 날카로움, 블러 효과
             A.Sharpen(p=1),
             A.Blur(p=1)], p=0.3),
    A.PiecewiseAffine(p=0.3), # 어파인 변환
    A.Normalize(), # 정규화 변환
    ToTensorV2() # 텐서로 변환
])

transform_test = A.Compose([
    A.Resize(224,224),
    A.Normalize(),
    ToTensorV2()
])
```
### 훈련
