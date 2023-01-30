
import streamlit as st
import cv2
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import base64
import time
import pandas as pd
from tqdm import tqdm

#visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


st.set_page_config(layout="wide")

# container
st.title('반려식물 병원 🏥')

st.write("웹캠 사용 또는 이미지 업로드를 이용해 잎사귀 질병을 분류합니다.")

@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = """
        <style>
        * {
            margin: 0px;
            padding: 0px;
        }
        .e8zbici2 {
        background-image: url("data:image/png;base64,%s");
        background-size: 100vw;
        background-repeat: no-repeat;
        }
        
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: 100vw;
        background-repeat: no-repeat;
        }

        .css-zt5igj {
            overflow: hidden;
        }
    
        .st-bc {
        margin-top: 40px; 
        }

        #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-k1vhr4.egzxvld5 > div.block-container.css-k1ih3n.egzxvld4 > div:nth-child(1) > div > div.stTabs.css-0.exp6ofz0 > div > div:nth-child(1) > div{
            background-color : #EEEEEE;
        }

        </style>
    """ % (bin_str, bin_str)
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_bg('bg2.png')


tab1, tab2, tab3, tab4 = st.tabs(["health", "multi diseases", "rust", "scab"])
with tab1:
    st.header("health")
    image = Image.open('./datas/images/Train_2.jpg')
    st.image(image, width= 500)

with tab2:
    st.header("multi diseases")
    image = Image.open('./datas/images/Train_1.jpg')
    st.image(image, width= 500)

with tab3:
    st.header("rust")
    image = Image.open('./datas/images/Train_3.jpg')
    st.image(image, width= 500)

with tab4:
    st.header("scab")
    image = Image.open('./datas/images/Train_0.jpg')
    st.image(image, width= 500)


    col1, col2, col3 = st.columns(3)
