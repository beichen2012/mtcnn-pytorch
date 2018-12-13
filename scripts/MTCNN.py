# coding: utf-8

import cv2
import torch
import time
import os
import numpy as np
from collections import OrderedDict
from Nets import *
from pylab import plt
from util.utility import pad_bbox, square_bbox, py_nms

SHOW_FIGURE = False

def Image2Tensor(img, MEANS):
    src = img.astype(np.float32) - np.array(MEANS, dtype=np.float32)
    src = src.swapaxes(1, 2).swapaxes(0, 1)

    input = torch.from_numpy(src).unsqueeze(0)
    return input

class MTCNN(object):

    def __init__(self, detectors=[None, None, None], min_face_size=40, scalor=0.79, threshold=[0.6, 0.7, 0.7],
                 device=torch.device("cpu")):
        self.pnet = detectors[0]
        self.rnet = detectors[1]
        self.onet = detectors[2]
        self.min_face_size = min_face_size
        self.scalor = scalor
        self.threshold = threshold
        self.device = device

    def detect(self, img):
        bboxes = None

        # pnet
        if not self.pnet:
            return None
        bboxes = self.detect_pnet(img)

        if bboxes is None:
            return None

        ## 可视化PNET的结果
        if SHOW_FIGURE:
            plt.figure()
            tmp = img.copy()
            for i in bboxes:
                x0 = int(i[0])
                y0 = int(i[1])
                x1 = x0 + int(i[2])
                y1 = y0 + int(i[3])
                cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
            plt.imshow(tmp[:, :, ::-1])
            plt.title("pnet result")

        # rnet
        if not self.rnet:
            return bboxes

        if not self.onet:
            return bboxes

        return bboxes


    def detect_pnet(self, im):
        h, w, c = im.shape
        net_size = 12
        minl = np.min((w, h))
        base_scale = net_size / float(self.min_face_size)
        scales = []
        face_count = 0
        while minl > net_size:
            s = base_scale * self.scalor ** face_count
            if np.floor(minl * s) <= 12:
                break
            scales += [s]
            face_count += 1

        # 对每个scale层做预测
        total_boxes = []
        for scale in scales:
            hs = np.ceil(h * scale)
            ws = np.ceil(w * scale)
            hs = int(hs)
            ws = int(ws)
            im_data = cv2.resize(im, (ws, hs))
            input = Image2Tensor(im_data, (127.5,127.5,127.5))
            input = input.to(self.device)

            output_cls, output_reg = self.pnet(input)
            output_cls = output_cls.squeeze_(0).cpu().detach().numpy()
            output_reg = output_reg.squeeze_(0).cpu().detach().numpy()

            bboxes = self.generate_bbox(output_cls, output_reg, scale, self.threshold[0])

            # inter-scale nms
            if len(bboxes) <= 0:
                continue
            keep = py_nms(bboxes, 0.5, 'Union')

            if len(keep) <= 0:
                continue

            bboxes = bboxes[keep]
            #
            total_boxes.extend(bboxes)

        # 金字塔所有层做完之后，再做一次NMS
        # NMS
        if len(total_boxes) <= 0:
            return None
        total_boxes = np.array(total_boxes)
        keep = py_nms(total_boxes, 0.7, 'Union')
        if len(keep) <= 0:
            return None
        return total_boxes[keep]

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            根据cls_map（预测结果图）与thread，选择相应的回归框，并将其映射回原图
        Parameters:
        ----------
            cls_map: numpy array , 2*h*w
                detect score for each position
            reg: numpy array , 4*h*w
                reg bbox
            scale: float number
                scale of this image pyramid from original image
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array in original image， num*5, [x,y,w,h,score]
        """
        stride = 2
        cellsize = 12
        face_map = cls_map[1, :, :]
        t_index = np.where(face_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        # dx, dy, dw, dh = [reg[t_index[0], t_index[1], i] for i in range(4)]
        dx, dy, dw, dh = [reg[i, t_index[0], t_index[1]] for i in range(4)]

        ## backwark for smooth l1 loss (RCNN)
        dx *= cellsize
        dy *= cellsize
        dw = np.exp(dw) * cellsize
        dh = np.exp(dh) * cellsize

        # 各回归框的分数
        score = face_map[t_index[0], t_index[1]]

        # 加上Gx, Gy，映射回原图
        Gx = np.round(stride * t_index[1] / scale)
        Gy = np.round(stride * t_index[0] / scale)
        dx = dx / scale + Gx
        dy = dy / scale + Gy
        dw = dw / scale
        dh = dh / scale
        # 组合结果
        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T
        return bbox
def LoadWeights(path, net):
    checkpoint = torch.load(path)
    weights = OrderedDict()
    for k, v in checkpoint["weights"].items():
        name = k[7:]
        weights[name] = v
    net.load_state_dict(weights)

if __name__ == "__main__":
    USE_CUDA = True
    GPU_ID = [0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")
    # pnet
    pnet_weight_path = "./models/pnet_20181212_final.pkl"
    pnet = PNet(test=True)
    LoadWeights(pnet_weight_path, pnet)
    pnet.to(device)

    mtcnn = MTCNN(detectors=[pnet, None, None], device=device)

    img_path = "~/dataset/faces2.jpg"
    img_path = os.path.expanduser(img_path)

    img =cv2.imread(img_path)

    mtcnn.detect(img)

    if SHOW_FIGURE:
        plt.show()










