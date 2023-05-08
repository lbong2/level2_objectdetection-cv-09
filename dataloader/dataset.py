import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image

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

    def __init__(self, root_dir, transforms=None):
        self._root_dir = root_dir
        self._transforms = transforms
        self._json_path = os.path.join(root_dir, 'train.json')

        with open(self._json_path) as f:
            self.json_data = json.load(f)
        
        self.num_classes = len(self.classes)

        self.image_paths = [i['file_name'] for i in self.json_data['images']]

        self.annotation = self._get_annotation()
        self.width = 1024
        self.height = 1024
    
    def _get_annotation(self):
        anno = [[] for _ in range(len(self.image_paths))]
        for i in self.json_data['annotations']:
            anno[i['image_id']].append(i)
        return anno


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        path = self.image_paths[idx]
        img_path = os.path.join(self._root_dir,path)
        img = Image.open(img_path)

        w = self.width
        h = self.height
        objs = []
        for obj in self.annotation[idx]:
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
