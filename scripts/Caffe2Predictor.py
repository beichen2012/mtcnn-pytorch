# coding: utf-8
"""
加载caffe2 的模型，并进行预测
"""
import os
from caffe2.python import core, workspace, model_helper
from caffe2.proto import caffe2_pb2
import numpy as np
import glog as log


def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x,y,w,h, score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def SpatialSoftmax(logits):
    """
    calc spatial softmax
    :param logits: shape(N,C,H,W)
    :return: the softmax map, shape(N,1,H,W)
    """
    # 1, calc the max value per c channel
    N,C,H,W = logits.shape
    cmax = np.max(logits, axis=1)
    for i in range(C):
        logits[:,i,:,:] -= cmax
        logits[:,i,:,:] = np.exp(logits[:,i,:,:])
    csum = np.sum(logits, axis=1)
    for i in range(C):
        logits[:,i,:,:] /= csum
    return logits




class C2Predictor(object):
    def __init__(self, init_net_path, predict_net_path, mean_val, scalor, device_option):
        self.device_option = device_option
        self.init_def = caffe2_pb2.NetDef()
        with open(init_net_path, "rb") as f:
            self.init_def.ParseFromString(f.read())
            for op in self.init_def.op:
                op.ClearField("device_option")

        self.net_def = caffe2_pb2.NetDef()
        with open(predict_net_path, "rb") as f:
            self.net_def.ParseFromString(f.read())
            for op in self.net_def.op:
                op.ClearField("device_option")


        self.net = self.net_def.name

        workspace.SwitchWorkspace(name=self.net, create_if_missing=True)
        self.init_def.device_option.CopyFrom(device_option)
        self.net_def.device_option.CopyFrom(device_option)
        workspace.RunNetOnce(self.init_def)
        workspace.CreateNet(self.net_def)
        self.mean_val = mean_val
        self.scalor = scalor

    def predict(self, img, output_blobs, input_blob_name="data"):
        """
        预测一幅图
        :param img: H,W,C的图像
        :param output_blobs: 需要输出的blob
        :param input_blob_name: 输入blob的名字
        :return:
        """
        src = img.swapaxes(1, 2).swapaxes(0, 1)
        src = src.astype(np.float32)
        src -= self.mean_val
        src *= self.scalor
        src = src[np.newaxis, ...]

        # forward
        workspace.SwitchWorkspace(name=self.net, create_if_missing=False)
        workspace.FeedBlob(input_blob_name, src, device_option=self.device_option)
        workspace.RunNet(self.net)

        # get result
        out = []
        for i in output_blobs:
            out += [workspace.FetchBlob(i)]

        return out

    def predict2(self, imgs, output_blobs, input_blob_name="data"):
        """
        预测多幅图
        :param img: N,C,H,W
        :param output_blobs: 需要输出的blob
        :param input_blob_name: 输入blob的名字
        :return:
        """
        src = imgs.astype(np.float32)
        src -= self.mean_val
        src *= self.scalor

        # forward
        workspace.SwitchWorkspace(name=self.net, create_if_missing=False)
        workspace.FeedBlob(input_blob_name, src, device_option=self.device_option)
        workspace.RunNet(self.net)

        # get result
        out = []
        for i in output_blobs:
            out += [workspace.FetchBlob(i)]

        return out




if __name__ == "__main__":
    log.info("process done...")
