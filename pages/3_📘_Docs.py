import streamlit as st

st.set_page_config(
    page_title='4leaf',
    layout='centered',
)

st.header('Code Viewer🧱')

if st.button('app.py code view'):
    code = '''
        import streamlit as st
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title='4leaf',
    layout='centered',
)

st.header('잎사귀 질병 유무 분류기🍀')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

model.eval()

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
    plant = ['healthy','multiple_desease','rust','scrab']

    answer = str(plant[predict_name]) + ' : ' + str(round(prob[predict_idx] * 100, 3)) + '%'
    st.image(img_for_plot, use_column_width=False)
    st.text(answer)
    '''
    st.code(code, language='python')