from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

import config 
import wandb
from importlib import import_module
import dataset
from dataset import CustomDataset, get_split
from utils import Averager, collate_fn
from metric import mAPLogger, calculate_map
import argparse
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        A.Rotate(p=0.5, limit=(-45, 45), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
            A.GaussianBlur(p=1.0, blur_limit=(3, 7), sigma_limit=(0.0, 0))
        ], p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def train_fn(num_epochs, train_data_loader, valid_data_loader, optimizer, model, device, cfg):
    # best_loss = 1000
    best_mAP = 0
    loss_hist = Averager()
    val_mAP = []
    val_cls_ap = []
    
    for epoch in range(num_epochs):
        map_logger = mAPLogger()
        loss_hist.reset()
        model.train()
        for images, targets, image_ids in tqdm(train_data_loader, desc=f'train #{epoch+1}: ', leave=False):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for images, targets, image_ids in tqdm(valid_data_loader, desc=f'validate #{epoch+1}: ', leave=False):    
                images = list(image.float().to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas}
                outputs = model(images, targets)
                # outputs: [{boxes:[[]], labels:[], scores: []}, ...] 배치 개수 만큼 dict 존재 
                
                # [train_idx(image_num), class_pred, prob_score, x1, y1, x2, y2]
                pred_boxes = []
                true_boxes = []
                for i, target in enumerate(targets):
                    # [[train_idx(image_num), class_pred, prob_score, x1, y1, x2, y2], [train_idx(image_num), class_pred, prob_score, x1, y1, x2, y2], ...]
                    
                    pred_boxes.extend([[target['image_id'], label, score, *box] 
                                    for box, label, score in zip(outputs[i]['boxes'], outputs[i]['labels'], outputs[i]['scores'])])                
                    true_boxes.extend([[target['image_id'], label, score, *box] 
                                    for box, label, score in zip(target['boxes'], target['labels'], target['area'])]) 
  
                map_logger.update(pred_boxes, true_boxes)
                # mAP, aps = calculate_map(tensor_pred_boxes, tensor_true_boxes)
            mAP = map_logger.get_mAP()


        if cfg.project:
            wandb.log({
                "val_mAP": mAP,
                "train_loss": loss_hist.value
            })
        print(f"Epoch #{epoch+1} train loss: {loss_hist.value} mAP: {mAP}")
        if mAP > best_mAP:
            save_path = f'./{cfg.model_save_dir}/{cfg.name}'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            best_mAP = mAP
            torch.save(model.state_dict(), os.path.join(save_path, f"{cfg.num_epochs}_{best_mAP}_best_model.pth"))
            
    del model
    torch.cuda.empty_cache()
    wandb.log({"best_mAP": best_mAP})
    wandb.config.update(vars(cfg))
    wandb.finish()

def main(cfg):
    # 데이터셋 불러오기
    annotation = '../dataset/train.json' # annotation 경로
    data_dir = '../dataset' # data_dir 경로
    get_split(annotation)
    train_dataset = CustomDataset("../dataset/temp_train.json", data_dir, get_train_transform()) 
    val_dataset = CustomDataset("../dataset/temp_val.json", data_dir, get_valid_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    # torchvision model 불러오기
    model_module = getattr(import_module("torchvision.models.detection"), cfg.model)
    weights = getattr(import_module("torchvision.models.detection"), "FasterRCNN_ResNet50_FPN_V2_Weights").DEFAULT
    model = model_module(weights=weights, **cfg.rcnn_params)
    num_classes = 11 # class 개수= 10 + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt_module = getattr(import_module("torch.optim"), cfg.optimizer)
    optimizer = opt_module(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    
    # training
    train_fn(cfg.num_epochs, train_data_loader, valid_data_loader, optimizer, model, device, cfg)

if __name__ == '__main__':
    
    seed_everything(42)
    cfg = config.read_config()
    parser = argparse.Namespace(**cfg)

    if parser.project:
        wandb.init(
            config=cfg,
            entity="boost_cv_09"
        )

    main(parser)