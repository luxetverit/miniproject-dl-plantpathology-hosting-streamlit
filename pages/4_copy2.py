import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import plotly.express as px
import matplotlib.gridspec as gridspec
import cv2
# 페이지 기본 설정
st.set_page_config(
    page_icon=":압정:",
    page_title="Plant Pathology"
)
st.subheader("Data Distribution Amount")
plt.rcParams['font.family'] = 'pretandard'
plt.rcParams['axes.unicode_minus'] = False
data_path = './plant-pathology-2020-fgvc7/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
healthy = train.loc[train['healthy']==1]
multiple_diseases = train.loc[train['multiple_diseases'] == 1]
rust = train.loc[train['rust']==1]
scab = train.loc[train['scab']==1]
diseases = dict()
for column in ["healthy","multiple_diseases","rust","scab"]:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = counts.iloc[1,0]
#bar chart to show different diseases
fig, ax1 = plt.subplots()
#fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,25))
ax1.bar(diseases.keys(),diseases.values(), color=["#7CFC00","#008000","#F0E68C","#556B2F"])
st.pyplot(fig)
fig, ax2 = plt.subplots()
ax2.pie(diseases.values(),labels = diseases.keys(), colors=["#7CFC00","#008000","#F0E68C","#556B2F"], autopct='%1.1f%%')
ax2.axis('equal')
st.pyplot(fig)
st.write("healthy")

def show_image(img_ids, rows=2, cols=3):
    assert len(img_ids) <= rows * cols # 이미지가 행/열 개수보다 많으면 오류 발생
    plt.figure(figsize=(15,8))
    fig, ax3 = plt.subplots()
    grid = gridspec.GridSpec(rows, cols)
    for idx, img_id in enumerate(img_ids):
        img_path = f'{data_path}/images/{img_id}.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(grid[idx])
        ax.imshow(image)
        ax.axis('off')
        
    st.pyplot(fig)
        
num_of_imgs=6
last_healthy_img_ids = healthy['image_id'][-num_of_imgs:]
last_multiple_diseases_img_ids = multiple_diseases['image_id'][-num_of_imgs:]
last_rust_img_ids = rust['image_id'][-num_of_imgs:]
last_scab_img_ids = scab['image_id'][-num_of_imgs:]
show_image(last_healthy_img_ids)
st.image(show_image(last_healthy_img_ids), use_column_width=False)
st.write("multiple_diseases")
show_image(last_multiple_diseases_img_ids)