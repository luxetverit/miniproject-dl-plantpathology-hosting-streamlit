import streamlit as st

st.set_page_config(
    page_title='4leaf',
    layout='centered',
)

st.header('Code Viewerπ§±')

if st.button('app.py code view'):
    code = '''
        import streamlit as st
import cv2
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import base64
import time

# μ΄λ―Έμ§ μλ‘λ -> λ°μ΄λλ¦¬ λ°μ΄ν° μ ν = byte λ¨μλ‘ μ½λλ€λ μλ―Έ -> μ¬μ΄μ¦ μ‘°μ  & ToTensor & μ κ·ν -> λͺ¨λΈμ ν¬μ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

st.header('μμ¬κ· μ§λ³ μ λ¬΄ λΆλ₯κΈ°π')
image_file = st.file_uploader("Choose a Image")
if image_file is not None:
    bytes_data = image_file.getvalue() # returns the entire contents of the stream regardless of current position.
    img = Image.open(BytesIO(bytes_data)).convert("RGB")
    # st.image(img, caption="μλ‘λν μ΄λ―Έμ§", use_column_width=True, clamp=True)β


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
        st.success("μ΄ μμ¬κ·λ μ μμλλ€.")
    elif pred == 1:
        st.error("μ΄ μμ¬κ·λ μ¬λ¬ κ°μ§ μ§λ³μ΄ μμ΅λλ€.")
    elif pred == 2:
        st.error("μ΄ μμ¬κ·λ rust μ§λ³μ΄ μμ΅λλ€.")
    else:
        st.error("μ΄ μμ¬κ·λ scab μ§λ³μ΄ μμ΅λλ€.")
else:
    st.warning("μ΄λ―Έμ§λ₯Ό μλ‘λν΄μ£ΌμΈμ.")
    

# model

model.eval()

    '''
    st.code(code, language='python')