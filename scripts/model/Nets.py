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
