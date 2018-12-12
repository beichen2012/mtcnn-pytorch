# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utility import IOU


class PNet(nn.Module):
    """
    pnet definition
    """

    def __init__(self, test=False):
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
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        )

        # regression
        self.regressioner = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)
        )

        self.test = test
        if test:
            self.softmax = nn.Softmax2d()

    def forward(self, data):
        y = self.features(data)
        cls = self.classifier(y)
        bbox = self.regressioner(y)
        if self.test:
            cls = self.softmax(cls)
        return cls, bbox


class RNet(nn.Module):
    def __init__(self, test=False):
        super(RNet, self).__init__()
        self.test = test

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0),
            nn.PReLU(),

            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),
        )

        self.cls = nn.Linear(128, 2)
        self.reg = nn.Linear(128, 4)

        if test:
            self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.extractor(x)
        x = x.contiguous().view(x.size(0), -1)
        cls = self.cls(x)
        reg = self.reg(x)
        if self.test:
            cls = self.softmax(cls)
        return cls, reg

class ONet(nn.Module):
    def __init__(self, test):
        super(ONet, self).__init__()

        self.test = test
        if test:
            self.softmax = nn.Softmax()

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(),

            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )

        self.cls = nn.Linear(256, 2)
        self.reg = nn.Linear(256, 4)

    def forward(self, x):
        x = self.extractor(x)
        x = x.contiguous().view(x.size(0), -1)
        cls = self.cls(x)
        reg = self.reg(x)
        if self.test:
            cls = self.softmax(cls)
        return cls, reg 



def AddClsLoss(pred, targets, k):
    label = targets[:, -1].long()
    # pred: 5 * 2 * 1 * 1
    idx = label < 2
    label_use = label[idx]
    pred_use = pred[idx]
    pred_use = torch.squeeze(pred_use)
    loss = F.cross_entropy(pred_use, label_use, reduction='none')
    topk = int(k * loss.size(0))
    loss, _ = torch.topk(loss, topk)
    loss = torch.mean(loss)
    return loss


def AddClsAccuracy(pred, targets):
    label = targets[:, -1].long()
    idx = label < 2
    label_use = label[idx]
    pred_use = pred[idx]
    pred_s = torch.squeeze(pred_use)
    pred_idx = torch.argmax(pred_s, dim=1)
    c = pred_idx.eq(label_use)
    s = torch.sum(c)
    n = pred_use.size(0)
    return float(s.item()) / float(n)


def AddRegLoss(pred, targets):
    label = targets[:, -1].long()
    bbox = targets[:, 0:4]
    # pred: N * 4 * 1 * 1
    idx = label > 0
    bbox_use = bbox[idx]
    pred_use = pred[idx]
    pred_squeeze = torch.squeeze(pred_use)
    # loss = F.mse_loss(pred_squeeze, bbox_use)
    loss = F.smooth_l1_loss(pred_squeeze, bbox_use)
    return loss
    # return


def AddBoxMap(pred, target, image_width, image_height):
    label = target[:, -1].long()
    bbox = target[:, 0:4]
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
