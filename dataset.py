from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np
import torch
import os 
import json 

from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None, ):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image_id = int(image_info['file_name'].split('/')[1][:4])
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정 
        labels = np.array([x['category_id']+1 for x in anns]) 
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
                                
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas}

        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())

def get_split(annotation, val_ratio=0.2):
    print("data split...")
    with open(annotation, "r") as f:
        cfg = json.load(f)
    x = [i for i in range(len(cfg['images']))]
    y = [0] * 4883
    for i in cfg['annotations']:
        y[i['image_id']] += 1
    
    train, val, _, _ = train_test_split(x, y, test_size=val_ratio)
    train_images = [cfg['images'][i] for i in train]
    val_images = [cfg['images'][i] for i in val]

    for i, image in enumerate(train_images):
        image['id'] = i
    for i, image in enumerate(val_images):
        image['id'] = i 

    train_anno = [i for i in cfg['annotations'] if i['image_id'] in train]
    val_anno = [i for i in cfg['annotations'] if i['image_id'] in val]
    for i, image in enumerate(train_anno):
        image['id'] = i
    for i, image in enumerate(val_anno):
        image['id'] = i 

    with open("../dataset/temp_train.json", "w") as f:
        cfg['images'] = train_images
        cfg['annotations'] = train_anno
        json.dump(cfg, f, indent='\t')

    with open("../dataset/temp_val.json", "w") as f:
        cfg['images'] = val_images
        cfg['annotations'] = val_anno
        json.dump(cfg, f, indent='\t')




        