
import streamlit as st
import cv2
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import base64
import time

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

st.header("picture")
img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    st.image(image, caption="웹캠에서 찍은 이미지", use_column_width=True)
    # Resize
    image = image.resize((224,224))
    # Normalization
    image = np.array(image)
    image = image/255.
    image = image.transpose(2,0,1)
    image = torch.tensor(image,dtype=torch.float32)
    image = image.unsqueeze(0)
    output = model(image)
    _, pred = torch.max(output, 1)
    pred = pred.item()
    if pred == 0:
        st.success("이 잎사귀는 정상입니다.")
    elif pred == 1:
        st.error("이 잎사귀는 여러 가지 질병이 있습니다.")
    elif pred == 2:
        st.error("이 잎사귀는 rust 질병이 있습니다.")
    else:
        st.error("이 잎사귀는 scab 질병이 있습니다.")
else:
    st.warning("웹캠에서 이미지를 찍어주세요.")

model.eval()

