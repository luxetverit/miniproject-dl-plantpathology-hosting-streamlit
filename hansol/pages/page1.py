
import streamlit as st
import cv2
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import base64
import time

# 이미지 업로드 -> 바이너리 데이터 전환 = byte 단위로 읽는다는 의미 -> 사이즈 조절 & ToTensor & 정규화 -> 모델에 투입

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

st.header("app")
image_file = st.file_uploader("Choose a Image")
if image_file is not None:
    bytes_data = image_file.getvalue() # returns the entire contents of the stream regardless of current position.
    img = Image.open(BytesIO(bytes_data)).convert("RGB")
    # st.image(img, caption="업로드한 이미지", use_column_width=True, clamp=True)​


if image_file is not None:
    bytes_data = image_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    image = transforms.Resize([224, 224])(image)
    img_for_plot = np.array(image)
    image = transforms.ToTensor()(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    image = normalize(image).unsqueeze(dim=0)
    output = model(image)
    _, pred = torch.max(output, 1)
    pred = pred.item()
    st.image(img_for_plot, use_column_width=False)
    if pred == 0:
        st.success("이 잎사귀는 정상입니다.")
    elif pred == 1:
        st.error("이 잎사귀는 여러 가지 질병이 있습니다.")
    elif pred == 2:
        st.error("이 잎사귀는 rust 질병이 있습니다.")
    else:
        st.error("이 잎사귀는 scab 질병이 있습니다.")
else:
    st.warning("이미지를 업로드해주세요.")
    

# model

model.eval()
