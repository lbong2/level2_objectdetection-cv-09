import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
from collections import defaultdict

def get_json_data(root_dir):
    """_summary_
    json 파일에서 데이터를 읽어오는 함수

    Args:
        root_dir (str): json 파일이 들어 있는 폴더

    Returns:
        json_data (dict): json data가 들어 있는 dictionary
    """
    json_path = os.path.join(root_dir, 'train.json')

    with open(json_path) as f:
        json_data = json.load(f)
    
    return json_data

def get_all_annotation(json_data):
    """_summary_
    json_data에서 annotation 정보들을 dict(list)로 추출

    Args:
        json_data (dict): json_data

    Returns:
        annotation (dict): key = img id, value = annotation 정보가 들어있는 list
    """
    anno = defaultdict(list)
    for i in json_data['annotations']:
        anno[i['image_id']].append(i)
    
    return anno


def split_train_valid(root_dir, json_data, val_ratio=0.2):
    """_summary_

    Args:
        root_dir (str): data가 들어있는 폴더 path
        json_data (dict): json data가 들어잇는 data
        val_ratio (float, optional): train, valid set을 나누는 비율. Defaults to 0.2.

    Returns:
        train, val path_info (list): [id, image_paths]의 데이터를 train과 validation info를 return
    """
    
    image_info = np.array([[i['id'], os.path.join(root_dir,i['file_name']) ] for i in json_data['images']])

    # validation set legnth
    length = int(len(image_info))
    split_length = int(val_ratio * length)

    # shuffle
    idx = np.random.permutation(image_info.shape[0])
    image_info = image_info[idx]
    
    # split info
    train_info = image_info[:length-split_length]
    val_info = image_info[length-split_length:]

    return train_info, val_info


class CustomDataset(Dataset):
    classes = ['General trash', 
               'Paper', 
               'Paper pack', 
               'Metal', 'Glass', 
               'Plastic', 
               'Styrofoam', 
               'Plastic bag', 
               'Battery', 
               'Clothing']
    
    classes_id = [i for i in range(len(classes))]

    def __init__(self, image_info, json_anno, transforms=None):
        self._transforms = transforms
        
        self.num_classes = len(self.classes)

        self.image_ids = image_info[0]
        self.image_paths = image_info[1]

        self.all_annotation = json_anno
        self.width = 1024
        self.height = 1024

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        id = self.image_ids[idx]

        w = self.width
        h = self.height
        objs = []
        for obj in self.annotation[id]:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((w - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((h - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                objs.append(obj)
        
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            boxes[ix,:] = obj['clean_bbox']
            gt_classes[ix] = obj['category_id']
            iscrowd.append(int(obj['iscrowd']))

        image_id = torch.tensor([idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": gt_classes, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


