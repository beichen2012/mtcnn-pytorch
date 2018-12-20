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
        bboxes = bboxes[:, 0:4].astype(np.int32)
        bboxes = self.detect_ronet(img, bboxes, 24)

        if bboxes is None:
            return None

        ## 可视化RNET的结果
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
            plt.title("rnet result")

        if not self.onet:
            return bboxes
        bboxes = bboxes[:, 0:4].astype(np.int32)
        bboxes = self.detect_ronet(img, bboxes, 48)

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
            self.pnet.eval()
            with torch.no_grad():
                output_cls, output_reg = self.pnet(input)
            output_cls = output_cls.squeeze_(0).detach().cpu().numpy()
            output_reg = output_reg.squeeze_(0).detach().cpu().numpy()

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

    def detect_ronet(self, img, bboxes, image_size):
        H, W, C = img.shape
        IMAGE_SIZE = image_size
        # 1, 先将bbox转换成矩形
        sb = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i, :]
            sq = square_bbox(box)
            sb += [sq]

        # 2，pad
        crops = []
        origin_bbox = []
        for i in sb:
            size = i[2]
            sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(i, W, H)
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                continue
            crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
            out = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            out = out.astype(np.float32) - np.array([127.5,127.5,127.5], dtype=np.float32)
            out = out.swapaxes(1, 2).swapaxes(0, 1)
            crops += [out]
            origin_bbox += [i]

        # 3, 预测
        origin_bbox = np.array(origin_bbox)
        crops = np.array(crops)
        input = torch.from_numpy(crops).to(self.device)
        detector = self.rnet
        threshold = self.threshold[1]
        if image_size == 48:
            detector = self.onet
            threshold = self.threshold[2]
        detector.eval()
        with torch.no_grad():
            out = detector(input)

        # 4，映射
        ## out[0] -> N * 2
        ## out[1] -> N * 4
        cls_map = out[0].detach().cpu().numpy()
        reg = out[1].detach().cpu().numpy()

        face_map = cls_map[:, 1]
        t_index = np.where(face_map > threshold)
        if t_index[0].shape[0] <= 0:
            return None

        # #
        origin_bbox = origin_bbox[t_index]
        score = face_map[t_index]
        reg_map = reg[t_index]

        dx = reg_map[:, 0]
        dy = reg_map[:, 1]
        dw = reg_map[:, 2]
        dh = reg_map[:, 3]

        # backward for smooth l1 loss(RCNN)
        dx *= IMAGE_SIZE
        dy *= IMAGE_SIZE
        dw = np.exp(dw) * IMAGE_SIZE
        dh = np.exp(dh) * IMAGE_SIZE

        # add Gx AND Gy
        G = origin_bbox
        G = G.astype(np.float32)

        dx = dx / (float(IMAGE_SIZE) / G[:, 2]) + G[:, 0]
        dy = dy / (float(IMAGE_SIZE) / G[:, 3]) + G[:, 1]
        dw = dw / (float(IMAGE_SIZE) / G[:, 2])
        dh = dh / (float(IMAGE_SIZE) / G[:, 3])

        # compose
        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T

        # do nms
        if image_size == 24:
            keep = py_nms(bbox, 0.6, "Union")
            if len(keep) <= 0:
                return None
            return bbox[keep]

        if image_size == 48:
            keep = py_nms(bbox, 0.6, "Minimum")
            if len(keep) <= 0:
                return None
            return bbox[keep]

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
    pnet_weight_path = "./models/pnet_20181218_final.pkl"
    pnet = PNet(test=True)
    LoadWeights(pnet_weight_path, pnet)
    pnet.to(device)

    # rnet
    rnet_weight_path = "./models/rnet_20181218_final.pkl"
    rnet = RNet(test=True)
    LoadWeights(rnet_weight_path, rnet)
    rnet.to(device)

    # onet
    onet_weight_path = "./models/onet_20181218_2_final.pkl"
    onet = ONet(test=True)
    LoadWeights(onet_weight_path, onet)
    onet.to(device)

    mtcnn = MTCNN(detectors=[pnet, rnet, onet], device=device, threshold=[0.6, 0.7, 0.7])


    #
    # img_path = "~/dataset/faces3.jpg"
    # # img_path = "~/dataset/WIDER_FACE/WIDER_train/images/14--Traffic/14_Traffic_Traffic_14_545.jpg"
    # # img_path = "~/dataset/WIDER_FACE/WIDER_val/images/14--Traffic/14_Traffic_Traffic_14_380.jpg"
    # img_path = os.path.expanduser(img_path)
    #
    # img =cv2.imread(img_path)
    # b = time.time()
    # bboxes = mtcnn.detect(img)
    # e = time.time()
    # print("time cost: {} ms".format((e-b) * 1000.0))
    #
    # if SHOW_FIGURE:
    #     if bboxes is not None:
    #         plt.figure()
    #         tmp = img.copy()
    #         for i in bboxes:
    #             x0 = int(i[0])
    #             y0 = int(i[1])
    #             x1 = x0 + int(i[2])
    #             y1 = y0 + int(i[3])
    #             cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
    #         plt.imshow(tmp[:, :, ::-1])
    #         plt.title("result")
    #     plt.show()

    fp = "~/dataset/GC_FACE_VAL"
    fp = os.path.expanduser(fp)
    txt = "file_list.txt"
    lines = []
    with open(os.path.join(fp, txt)) as f:
        lines = f.readlines()

    f = open("res.txt", "w")
    for i in lines:
        j = i.strip()
        print("now process -> {}".format(j))
        f.write("{}".format(j))
        sample = os.path.join(fp, j)
        img = cv2.imread(sample)
        bboxes = mtcnn.detect(img)

        if bboxes is not None:
            f.write(" {} ".format(bboxes.shape[0]))
            for b in bboxes:
                x = int(b[0])
                y = int(b[1])
                w = int(b[2])
                h = int(b[3])
                f.write("{} {} {} {} ".format(x, y, w, h))
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.imwrite("output/{}".format(j), img)
        else:
            f.write(" 0")
        f.write("\n")
    f.close()
    print("done")








