# coding: utf-8
"""
实用功能函数
"""
import numpy as np



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


def Rectrect(w,h,rect):
    r = rect
    if r[0] < 0:
        r[0] = 0
    if r[1] < 0:
        r[1] = 0
    if r[0] + r[2] > w - 1:
        r[2] = w - 1 - r[0]
    if r[1] + r[3] > h - 1:
        r[3] = h - 1 - r[1]
    return r


def square_bbox(bbox):
    """
    把bbox，变成方形（以bbox的中心为中心，最长边为边）
    :param bbox:
    :return:
    """
    h = bbox[3]
    w = bbox[2]
    max_size = np.max((w,h))

    sq = bbox.copy()
    sq[0] = bbox[0] + w * 0.5 - max_size * 0.5
    sq[1] = bbox[1] + h * 0.5 - max_size * 0.5
    sq[2] = max_size
    sq[3] = max_size
    return sq

def pad_bbox(bbox, W, H):
    """
    计算bbox在原图中的坐标，及拷贝到小图的坐标
    :param bbox:
    :return:
    """
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2] + x0
    y1 = bbox[3] + y0

    # src dst
    sx0 = x0
    dx0 = 0
    sy0 = y0
    dy0 = 0
    sx1 = x1
    dx1 = bbox[2]
    sy1 = y1
    dy1 = bbox[3]

    # 如果x小于0
    if x0 < 0:
        sx0 = 0
        dx0 = -x0
    if y0 < 0:
        sy0 = 0
        dy0 = -y0

    if x1 > W - 1:
        sx1 = W - 1
        dx1 = sx1 - sx0
    if y1 > H - 1:
        sy1 = H - 1
        dy1 = sy1 - sy0

    #
    miny = np.min((dy1-dy0, sy1-sy0))
    dy1 = dy0 + miny
    sy1 = sy0 + miny
    minx = np.min((dx1-dx0, sx1-sx0))
    dx1 = dx0 + minx
    sx1 = sx0 + minx

    return sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1


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
