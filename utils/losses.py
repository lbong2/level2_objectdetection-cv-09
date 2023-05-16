import torch
from utils.mAP import intersection_over_union
import math
import torch.nn as nn
import torch.nn.functional as F

def _loss_inter_union(boxes1,boxes2):

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return intsctk, unionk

def _diou_iou_loss(boxes1,boxes2,eps):
    iou = intersection_over_union(boxes1,boxes2,eps)
    # smallest enclosing box
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    # The diagonal distance of the smallest enclosing box squared
    diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)

    return loss, iou

def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _upcast_non_float(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype not in (torch.float32, torch.float64):
        return t.float()
    return t

class SmoothL1Loss(nn.Module):
    def __init__(self,beta: float=1./9, size_average=False):
        super(SmoothL1Loss,self).__init__()
        self.beta = beta
        self.size_average=size_average

    def forward(self, input, target):
        """
        smooth_l1_loss for bbox regression
        :param input:
        :param target:
        :param beta:
        :param size_average:
        :return:
        """

        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        if self.size_average:
            return loss.mean()
        return loss.sum()


class IoULoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps
        
    def forward(self,pred_box,target_box):
        """Calculate the IoU loss between predicted bounding boxes and target bounding boxes.
        
        Args:
            pred (torch.Tensor): Predicted bounding boxes of shape (N, 4).
            target (torch.Tensor): Target bounding boxes of shape (N, 4).
        
        Returns:
            torch.Tensor: IoU loss.
        """
        iou = intersection_over_union(pred_box,target_box,self.eps)
        iou_loss = 1 - iou.mean()
        return iou_loss


class GIoULoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(GIoULoss, self).__init__()
        self.eps = eps

    def forward(self,pred_box, target_box):
        pred_box = _upcast_non_float(pred_box)
        target_box = _upcast_non_float(target_box)
        # pred_box, target_box : (N, 4)
        # returns : scalar
        intsct, union = _loss_inter_union(pred_box, target_box)
        iou = intsct / (union + self.eps)

        x1, y1, x2, y2 = pred_box.unbind(dim=-1)
        x1g, y1g, x2g, y2g = target_box.unbind(dim=-1)

        # smallest enclosing box
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1)
        miouk = iou - ((area_c - union) / (area_c + self.eps))

        loss = 1 - miouk
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()

        return loss


class DIoULoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(DIoULoss, self).__init__()
        self.eps= eps

    def forward(self,pred_box, target_box):
        pred_box = _upcast_non_float(pred_box)
        target_box = _upcast_non_float(target_box)
        loss, _ = _diou_iou_loss(pred_box,target_box,self.eps)
        loss = loss.mean()

        return loss


class CIoULoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(CIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_box, target_box):
        pred_box = _upcast_non_float(pred_box)
        target_box = _upcast_non_float(target_box)

        diou_loss, iou = _diou_iou_loss(pred_box, target_box, self.eps)
        x1, y1, x2, y2 = pred_box.unbind(dim=-1)
        x1g, y1g, x2g, y2g = target_box.unbind(dim=-1)

        # width and height of boxes
        w_pred = x2 - x1
        h_pred = y2 - y1
        w_gt = x2g - x1g
        h_gt = y2g - y1g
        v = (4 / (math.pi **2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        loss = diou_loss + alpha * v

        loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """_summary_
    Focal Loss를 사용하기 위한 class

    """
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        """_summary_

        Args:
            weight (list, optional): 가중치를 더할 list. Defaults to None.
            gamma (float, optional): (1-p)**gamma . Defaults to 2..
            reduction (str, optional): mean,sum 등의 return할 loss 계산. Defaults to 'mean'.
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, labels):
        """_summary_

        Args:
            pred (tensor): 모델이 예측한 결과값
            labels (tensor): 정답 label

        Returns:
            tensor: Focal Loss로 계산한 loss 값
        """
        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            labels,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    """_summary_
    LabelSmoothing Loss를 사용하기 위한 class
    
    """
    def __init__(self, classes=11, smoothing=0.1, dim=-1):
        """_summary_

        Args:
            classes (int, optional): classes개수 만큼 smoothing. Defaults to 3.
            smoothing (float, optional): smoothing 할 비율. Defaults to 0.1.
            dim (int, optional): Defaults to -1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, labels):
        """_summary_

        Args:
            pred (tensor): 모델이 예측한 tensor
            labels (tensor): 정답 label

        Returns:
            loss (tensor): label smoothing을 계산한 loss
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

CRITERION_ENTRYPOINTS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'bce': nn.BCEWithLogitsLoss,
    'label_smoothing': LabelSmoothingLoss,
    'smoothl1loss': SmoothL1Loss,
    'iou_loss':IoULoss,
    'diou_loss':DIoULoss,
    'giou_loss':GIoULoss,
    'ciou_loss':CIoULoss
}


def criterion_entrypoint(criterion_name):
    """_summary_
    CRITERION_ENTRYPOINTS에 해당하는 Loss return
    Args:
        criterion_name (str): crtierion name

    Returns:
        criterion (nn.module): criterion 
    """
    return CRITERION_ENTRYPOINTS[criterion_name]


def is_criterion(criterion_name):
    """_summary_
    CRITERION_ENTRYPOINTS에 해당하는 Loss 인지 확인
    Args:
        criterion_name (str): crtierion name

    Returns:
        bool: 있다면 True, 없으면 False
    """
    return criterion_name in CRITERION_ENTRYPOINTS


def create_criterion(criterion_name, **kwargs):
    """_summary_

    Args:
        criterion_name (str): ['cross_entropy', 'focal', 'label_smoothing', 'f1', 'bce'] 사용가능

    Raises:
        RuntimeError: 해당 하는 loss가 없다면 raise error

    Returns:
        loss (Module): 해당 하는 loss return
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion