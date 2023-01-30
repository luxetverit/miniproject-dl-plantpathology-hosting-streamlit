import streamlit as st
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
import cv2

st.set_page_config(
    page_icon='🎃',
    page_title='TITLE',
    layout='centered',
)

st.header('잎사귀 질병 유무 분류기🍀')

tab1, tab2 = st.tabs(["app", "picture"])
with tab1:
    st.header("app")
    uploaded_file = st.file_uploader("Choose a Image")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(BytesIO(bytes_data)).convert("RGB")
        image = transforms.Resize([224, 224])(image)
        img_for_plot = np.array(image)
        img = transforms.ToTensor()(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        img = normalize(img).unsqueeze(dim=0)
        result = model(img)
        prob = torch.softmax(result.cpu(), dim=1).squeeze().detach().numpy()
        predict_idx = result.argmax().item()
        predict_name = result.argmax()
        plant = ['healthy','multiple_desease','rust','scab']
        answer = str(plant[predict_name]) + ':' , str(round(prob[predict_idx], 4) * 100) + '%'
        st.image(img_for_plot, use_column_width=False)
        st.text(answer)
        answer = str(plant[predict_name]) + ':' , str(round(prob[predict_idx] * 100, 3)) + '%'
with tab2:
    st.header("picture")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        st.write(type(cv2_img))
        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        st.write(cv2_img.shape)

st.sidebar.markdown("# :green[**Click app or picture!**]")
button = st.sidebar.button("app")
button2 =st.sidebar.button("picture")