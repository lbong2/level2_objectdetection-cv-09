import torch
from collections import Counter

import time
import torch



def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class mAPLogger(object):
    """_summary_
    mAP를 계산하고 업데이트 하는 클래스

    """
    def __init__(self, iou_threshold=0.5, num_classes = 10):
        """_summary_

        Args:
            iou_threshold (float, optional): FP를 결정하는 iou_threshold. Defaults to 0.5.
            num_classes (int, optional): num_class. Defaults to 10.
        """
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.epsilon = 1e-6
        
        self.TPs = [torch.Tensor([]) for _ in range(self.num_classes)]
        self.FPs = [torch.Tensor([]) for _ in range(self.num_classes)]
        self.total_true_bboxes = [0]* self.num_classes
        self.recalls = [0]* self.num_classes
        self.precisions = [0]* self.num_classes
        self.APs = [0] * self.num_classes
        self.mAP = 0
        
    def update(self,pred_boxes, true_boxes):
        """_summary_
        update 마다 각 클래스의 TP,FP를 구하여 precision, recall 연산
        precision, recall을 가지고 각 클래스의 AP를 구한후 mAP연산

        Args:
            pred_boxes (list): pred_boxes (list):  [[train_idx(image_num), class_pred, prob_score, x1, y1, x2, y2], ...]
            true_boxes (list): true_boxes (list):  [[train_idx(image_num), class_id, area, x1, y1, x2, y2], ...]
        """
        for c in range(self.num_classes):
            detections = [] # 각 클래스의 detection이 담길 리스트
            ground_truths = [] # 각 클래스의 ground truth가 담길 리스트
            
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)
                    
            for true_box in true_boxes: 
                if true_box[1] == c:
                    ground_truths.append(true_box)
                    
            # img 0 has 3 bboxes
            # img 1 has 5 bboxes
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter(gt[0] for gt in ground_truths) # gt의 각 이미지(key)의 개수(value)를 셈
            
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val) # 개수를 1차원 tensor로 변환
            # amount_boxes = {0: torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
            
            detections.sort(key=lambda x: x[2], reverse=True) # detections의 confidence가 높은 순으로 정렬
            TP = torch.zeros((len(detections))) # detections 개수만큼 1차원 TP tensor를 초기화
            FP = torch.zeros((len(detections))) # 마찬가지로 1차원 FP tensor 초기화
            total_true_bboxes = len(ground_truths) # recall의 TP+FN으로 사용됨

            self.total_true_bboxes[c] +=total_true_bboxes # TP+FN update 
            
            for detection_idx, detection in enumerate(detections): # 정렬한 detections를 하나씩 뽑음
                # ground_truth_img : detection과 같은 이미지의 ground truth bbox들을 가져옴
                ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]         
                best_iou = 0 # 초기화
                
                for idx, gt in enumerate(ground_truth_img): # 현재 detection box를 이미지의 ground truth들과 비교
                    iou = intersection_over_union(
                            torch.Tensor(detection[3:]).view(1,-1),
                            torch.Tensor(gt[3:]).view(1,-1),
                            )
                    
                    if iou > best_iou: # ground truth들과의 iou중 가장 높은놈의 iou를 저장
                        best_iou = iou
                        best_gt_idx = idx # 인덱스도 저장
                
                if best_iou > self.iou_threshold: # 그 iou가 0.5 이상이면 헤당 인덱스에 TP = 1 저장, 이하면 FP = 1 저장 
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1 # 이미 해당 물체를 detect한 물체가 있다면 즉 인덱스 자리에 이미 TP가 1이라면 FP=1적용
                else:
                    FP[detection_idx] = 1

            # update
            self.TPs[c] = torch.cat((self.TPs[c],TP),dim=0)
            self.FPs[c] = torch.cat((self.FPs[c],FP),dim=0)

            TP_cumsum = torch.cumsum(self.TPs[c], dim=0)
            FP_cumsum = torch.cumsum(self.FPs[c], dim=0)
            self.recalls[c] = TP_cumsum / (self.total_true_bboxes[c] + self.epsilon)
            self.precisions[c] = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + self.epsilon)) # TP_cumsum + FP_cumsum을 하면 1씩 증가하게됨
            
            self.recalls[c] = torch.cat((torch.tensor([0]), self.recalls[c])) # x축의 시작은 0 이므로 맨앞에 0추가
            self.precisions[c] = torch.cat((torch.tensor([1]), self.precisions[c])) # y축의 시작은 1 이므로 맨앞에 1 추가
            self.APs[c] = torch.trapz(self.precisions[c], self.recalls[c]).item() # 현재 클래스에 대해 AP를 계산해줌, trapz(y,x) x에 대한 y의 적분
        
        self.mAP = round(sum(self.APs) / len(self.APs),4)

    def get_ap_list(self):
        return self.APs
    
    def get_mAP(self):
        return self.mAP


def iou(box1, box2):
    # box1, box2: [x1, y1, x2, y2]

    # Get the coordinates of the intersection rectangle
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Calculate intersection area
    intersection = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    # Calculate box areas
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union = area_box1 + area_box2 - intersection

    # Calculate IOU
    iou = intersection / union

    return iou

def calculate_ap(ground_truth, predictions, iou_threshold=0.5):
    # Sort predictions by confidence score in descending order
    predictions = predictions[torch.argsort(predictions[:, 4], descending=True)]

    true_positives = torch.zeros(len(predictions))
    false_positives = torch.zeros(len(predictions))
    num_ground_truth = len(ground_truth)

    for i, prediction in enumerate(predictions):
        pred_box = prediction[:4]
        pred_conf = prediction[4]

        iou_max = torch.tensor(0.0)
        ground_truth_idx = -1

        for j, ground_truth_box in enumerate(ground_truth):
            iou_val = iou(pred_box, ground_truth_box)
            if iou_val > iou_max:
                iou_max = iou_val
                ground_truth_idx = j

        if iou_max >= iou_threshold and ground_truth_idx >= 0:
            if ground_truth[ground_truth_idx][4] == 0:
                true_positives[i] = 1
                ground_truth[ground_truth_idx][4] = 1
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1

    cum_true_positives = torch.cumsum(true_positives, dim=0)
    cum_false_positives = torch.cumsum(false_positives, dim=0)
    precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
    recall = cum_true_positives / num_ground_truth

    # Add precision and recall values for calculation of AP
    precision = torch.cat((torch.tensor([1.0]), precision))
    recall = torch.cat((torch.tensor([0.0]), recall))

    # Calculate average precision (AP)
    ap = torch.trapz(precision, recall)

    return ap


def calculate_map(ground_truths, predictions, iou_threshold=0.5):
    num_classes = 10  # 예측값 텐서의 마지막 차원에서 클래스 개수 추출
    aps = torch.zeros(num_classes)

    for c in range(num_classes):
        gt_class = ground_truths[:, ground_truths[:, :, 4] == c]
        pred_class = predictions[:, predictions[:, :, 5] == c]

        ap = calculate_ap(gt_class, pred_class, iou_threshold)
        aps[c] = ap

    mean_ap = aps.mean()

    return mean_ap, aps
