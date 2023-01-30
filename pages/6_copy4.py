import random
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
import cv2

def grad_cam():
    # 데이터
    test = pd.read_csv('./plant-pathology-2020-fgvc7/test.csv')
    # 데이터셋, 로더
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(0)
        class ImageDataset(Dataset):
            def __init__(self, df, img_dir='./', transform=None, is_test=False):
                super().__init__()
                self.df = df
                self.img_dir = img_dir
                self.transform = transform
                self.is_test = is_test
            def __len__(self):
                return len(self.df)
            def __getitem__(self, idx):
                img_id = self.df.iloc[idx, 0]
                img_path = self.img_dir + img_id + '.jpg'
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform is not None:
                    image = self.transform(image=image)['image']
                # 테스트 데이터이면 이미지 데이터만 반환, 그렇지 않으면 타깃값도 반환
                if self.is_test:
                    return image
                else:
                    # 타깃값 4개 중 가장 큰 값의 인덱스
                    label = np.argmax(self.df.iloc[idx, 1:5])
                    return image, label
        def Loader(img_path=None, uploaded_image=None, upload_state=False):
            img_dir = './plant-pathology-2020-fgvc7/images/'
            batch_size = 4
            test_dataset = ImageDataset(test, img_dir=img_dir)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=True, worker_init_fn = seed_worker,
                                        generator=g, num_workers=0)
            return test_loader
        def deploy(file_path=None, uploaded_image=uploaded_image, uploaded=False, demo=True):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load('model.pth', map_location=device)
            model.eval()
            # For Grad-cam features
            final_conv = model.layer4[2]._modules.get("conv3")
            fc_params = list(model._modules.get("fc").parameters())
            if uploaded:
                test_loader = Loader(
                    uploaded_image=uploaded_image, upload_state=True, demo_state=False
                )
                image_1 = file_path
            for img in test_loader:
                activated_features = SaveFeatures(final_conv)
                # Save weight from fc
                weight = np.squeeze(fc_params[0].cpu().data.numpy())
                # Inference
                logits, output = inference(model, states, img, device)
                pred_idx = output.to("cpu").numpy().argmax(1)
                # Grad-cam heatmap display
                heatmap = getCAM(activated_features.features, weight, pred_idx)
                ##Reverse the pytorch normalization
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
                image = img[0] * STD[:, None, None] + MEAN[:, None, None]
                # Display image + heatmap
                plt.imshow(image.permute(1, 2, 0))
                plt.imshow(
                    cv2.resize(
                        (heatmap * 255).astype("uint8"),
                        (328, 328),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    alpha=0.4,
                    cmap="jet",
                )
                plt.savefig(output_image)
                # Display the Grad-Cam image
                st.title("**Grad-cam visualization**")
                gram_im = cv2.imread(output_image)
                st.image(gram_im, width=528, channels="RGB")