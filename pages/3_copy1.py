import streamlit as st
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
def main():
    st.set_page_config(page_title="잎사귀 질병 분류 웹앱", page_icon=":경비병:", layout="wide")
    st.sidebar.title("메뉴")
    menu = ["메인 화면", "웹캠 사용", "이미지 업로드"]
    app_mode = st.sidebar.selectbox("메뉴를 선택하세요", menu)
    if app_mode == "메인 화면":
        st.title("메인 화면")
        st.write("웹캠 사용 또는 이미지 업로드를 이용해 잎사귀 질병을 분류합니다.")
    elif app_mode == "웹캠 사용":
        st.title("웹캠 사용")
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
    else:
        st.title("이미지 업로드")
        image_file = st.file_uploader("이미지를 선택하세요", type=["jpg", "png","jpeg"])
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
# 이미 학습된 모델을 불러옵니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)
model.eval()
if __name__ == '__main__':
    main()