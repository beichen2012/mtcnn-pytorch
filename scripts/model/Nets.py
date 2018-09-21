# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    """
    pnet definition
    """

    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(
            # 3*12*12 -> 10*5*5
            nn.Conv2d(3, 10, kernel_size=3, stride=2, padding=0),
            nn.PReLU(),

            # 10*5*5 -> 16*3*3
            nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),

            # 16*3*3 -> 32*1*1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU()
        )

        # cls
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            nn.Softmax2d()
        )

        # regression
        self.regressioner = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, data):
        y = self.features(data)
        cls = self.classifier(y)
        bbox = self.regressioner(y)
        return cls, bbox


def AddClsLoss(pred, label):
    # pred: 5 * 2 * 1 * 1
    idx = label < 2
    label_use = label[idx]
    pred_use = pred[idx]
    pred_log = torch.log(pred_use)
    pred_freeze = torch.squeeze(pred_log)
    loss = F.nll_loss(pred_freeze, label_use)
    return loss


def AddClsAccuracy(pred, label):
    idx = label < 2
    label_use = label[idx]
    pred_use = pred[idx]
    pred_s = torch.squeeze(pred_use)
    pred_idx = torch.argmax(pred_s, dim=1)
    c = pred_idx.eq(label_use)
    s = torch.sum(c)
    n = pred.size(0)
    return float(s.item()) / float(n)


def AddRegLoss(pred, label, bbox):
    # pred: N * 4 * 1 * 1
    idx = label > 0
    bbox_use = bbox[idx]
    pred_use = pred[idx]
    pred_squeeze = torch.squeeze(pred_use)
    return F.smooth_l1_loss(pred_squeeze, bbox_use)

def AddBoxMap(pred, label, bbox, image_width, image_height):
    # pred: N * 4 * 1 * 1
    idx = label > 0
    bbox_use = bbox[idx]
    pred_use = pred[idx]
    pred_squeeze = torch.squeeze(pred_use)

    pred_squeeze = pred_squeeze.cpu().detach().numpy()
    bbox_use = bbox_use.cpu().detach().numpy()
    # calc ap
    map = 0.0
    pred_squeeze[:, 0] *= image_width
    pred_squeeze[:, 1] *= image_height
    bbox_use[:, 0] *= image_width
    bbox_use[:, 1] *= image_height
    num = bbox_use.shape[0]
    for i in range(num):
        b1 = pred_squeeze[i]
        b2 = bbox_use[i]
        b1[2] = np.exp(b1[2]) * image_width
        b1[3] = np.exp(b1[3]) * image_height

        b2[2] = np.exp(b2[2]) * image_width
        b2[3] = np.exp(b2[3]) * image_height
        map += IOU(b1, b2)

    return map / num

def IntersectBBox(bbox1, bbox2):
    if (bbox2[0] > bbox1[0] + bbox1[2] or bbox2[0] + bbox2[2] < bbox1[0] or
            bbox2[1] > bbox1[1] + bbox1[3] or bbox2[1] + bbox2[3] < bbox1[1]):
        return 0, 0, 0, 0
    #
    x = np.max((bbox1[0], bbox2[0]))
    y = np.max((bbox1[1], bbox2[1]))
    w = np.min((bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])) - x + 1
    h = np.min((bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])) - y + 1
    return x, y, w, h


def IOM(bbox1, bbox2):
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    area_intersect = intersect_bbox[3] * intersect_bbox[2]
    area_bbox1 = bbox1[2] * bbox1[3]
    area_bbox2 = bbox2[2] * bbox2[3]

    area_down = 0.0000001 + np.min((area_bbox2, area_bbox1))
    return area_intersect / area_down


def IOU(bbox1, bbox2):
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    if intersect_bbox[2] <= 0 or intersect_bbox[3] <= 0:
        return 0.0
    #
    area_intersect = intersect_bbox[2] * intersect_bbox[3]
    area_bbox1 = bbox1[2] * bbox1[3]
    area_bbox2 = bbox2[2] * bbox2[3]
    return float(area_intersect) / float(area_bbox1 + area_bbox2 - area_intersect)
