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
