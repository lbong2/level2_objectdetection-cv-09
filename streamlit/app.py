import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision

import random

from pytorch_grad_cam import AblationCAM, EigenCAM
#from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

import pandas as pd


st.set_page_config(initial_sidebar_state="collapsed")
st.title("Object Detection with CV-09 Team model")
st.text("Upload an image")

with st.sidebar:
    confidence_thresh = st.slider(
        'select Confidence Threshold',
        0.0, 1.0, 0.5
    )
    add_radio = st.radio(
            "Select GradCam",
            ("None", "GradCam")
        )
    
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
colors = [(random.randint(0, 255) / 255., random.randint(0, 255) / 255, random.randint(0, 255) / 255) for _ in range(len(classes))]

def detect_objects(image, net, confidence_threshold=0.5):
    images = list(img.to('cuda') for img in image)
    output = net(images)[0]
    return output['boxes'].tolist(), output['scores'].tolist(), output['labels'].tolist()

@st.cache()
def load_model():
    check_point = "30_1992_best_model.pth"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
    num_classes = 11  # 10 class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to('cuda')
    model.load_state_dict(torch.load(check_point))
    model.eval()
    return model


def gradCamImg(img, model, classIDs, boxes):
    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=classIDs, bounding_boxes=boxes)]
    cam = EigenCAM(model,
        target_layers, 
        use_cuda=torch.cuda.is_available(),
        reshape_transform=fasterrcnn_reshape_transform)
    grayscale_cam = cam(img, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    return show_cam_on_image(np.clip(img.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1), grayscale_cam, use_rgb=False)   


def main():
    uploaded_file = st.file_uploader("Choose an image...", accept_multiple_files=False)
    if uploaded_file is not None:

        model = load_model()

        # 이미지 업로드 및 전처리
        img = np.array(Image.open(uploaded_file)) / 255.
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0)
        # 객체 검출 수행
        pred_start_time = time.time()
        boxes, confidences, classIDs = detect_objects(img, model)
        pred_end_time = time.time()

        

        if add_radio == 'GradCam':
            gradcam_start_time = time.time()
            cam_image = gradCamImg(img, model, classIDs, boxes)
            gradcam_end_time = time.time()
           
            
        img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        
        
        detec_img = img.copy()
        df_list = []

        for i in range(len(boxes)):
            if confidences[i] > confidence_thresh:
                x1, y1, x2, y2 = map(int, boxes[i])

                label = classes[classIDs[i]-1]
                color = colors[classIDs[i]-1]
                confidence = confidences[i]
                text = f"{label} ({confidence:.2f})"
                cv2.rectangle(detec_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(detec_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                df_list.append([x1, y1, x2, y2, confidence, label])


        df_list.sort(key=lambda x: x[4], reverse=True)

        cols = st.columns(2)
        if add_radio == 'GradCam':
            st.header("GradCam Image")
            st.image(cam_image, use_column_width=True)
        with cols[0]:
            st.header("Original image")
            st.image(img, use_column_width=True)
        with cols[1]:
            st.header("Detection image")
            st.image(np.clip(detec_img, 0.0, 1.0), use_column_width=True)
        st.text(f"pred Time taken: {pred_end_time - pred_start_time:.2f} seconds")
        if add_radio == 'GradCam':
            st.text(f"gradCam Time taken: {gradcam_end_time - gradcam_start_time:.2f} seconds")


        df = pd.DataFrame(
            df_list,
            columns=('x1', 'y1', 'x2', 'y2', 'confidence', 'class')
        )
        st.header("Confidence Table")
        st.table(df)
if __name__ == '__main__':
    main()