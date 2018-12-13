# coding: utf-8
"""
读取wider face 数据集
利用 P(+R)NET 生成正样本和负样本。
正样本标签为：1
负样本标签为：0
部分人脸样本标签为：2
综合标签为： img_path xmin, ymin, xmax, ymax, label
例如： 1.jpg 1 -1 2 10 11
注意：保存的图片缩放到了N*N，bbox坐标也是相对于小图的，但是未做scale
"""
import sys
sys.path.append(sys.path[0] + "/../")
import os
import numpy as np
import random
import cv2
from util.Logger import Logger
import time
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger
from MTCNN import *


# 小于该人脸的就不要了
MIN_FACE_SIZE = 40
IOU_POS_THRES = 0.65
IOU_NEG_THRES = 0.3
IOU_PART_THRES = 0.4

net_type = "RNET"
if net_type == "RNET":
    OUT_IMAGE_SIZE = 24
    post_fix = 'r'
else:
    OUT_IMAGE_SIZE = 48
    post_fix = 'o'

# path to wider face
root_dir = r'~/dataset/WIDER_FACE'
root_dir = os.path.expanduser(root_dir)

train_dir = os.path.join(root_dir, 'WIDER_train/images')
val_dir = os.path.join(root_dir, 'WIDER_val/images')
anno_dir = os.path.join(root_dir, 'wider_face_split')

# path to output root dir
output_root_dir = r"../dataset/train_faces_{}".format(post_fix)
if not os.path.exists(output_root_dir):
    os.mkdir(output_root_dir)

# output dirs: pos and neg
output_pos_dir = os.path.join(output_root_dir, "pos")
output_neg_dir = os.path.join(output_root_dir, "neg")
output_part_dir = os.path.join(output_root_dir, "part")
if not os.path.exists(output_pos_dir):
    os.mkdir(output_pos_dir)
if not os.path.exists(output_neg_dir):
    os.mkdir(output_neg_dir)
if not os.path.exists(output_part_dir):
    os.mkdir(output_part_dir)

# ouput labels file
label_pos = os.path.join(output_pos_dir, "anno_pos.txt")
label_neg = os.path.join(output_neg_dir, "anno_neg.txt")
label_part = os.path.join(output_part_dir, "anno_part.txt")


def GenerateData(mt):
    fanno_pos = open(label_pos, "w")
    fanno_neg = open(label_neg, "w")
    fanno_part = open(label_part, "w")
    anno_file = os.path.join(anno_dir, "wider_face_train_bbx_gt.txt")
    not_reg_file = open('no_reg.txt', 'w')
    with open(anno_file, "r") as f:
        while True:
            filename = f.readline()
            if not filename:
                break
            filename = filename.strip('\n')
            log.info("now process -> {}".format(filename))
            face_num = f.readline()
            face_num = int(face_num)


            # 读取真值 bbox
            gt_bbox = []
            for i in range(face_num):
                line = f.readline()
                line = line.split()
                x = int(line[0])
                y = int(line[1])
                w = int(line[2])
                h = int(line[3])
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                gt_bbox += [(x, y, w, h)]

            # 读取图片
            img = cv2.imread(os.path.join(train_dir, filename))
            if img is None:
                log.warning("error to load image {}", filename)
                continue
            H, W, C = img.shape

            # 预测
            bbox = mt.detect(img)
            if bbox is None:
                log.warning("this is pci -> {}, detected no faces out!".format(filename))
                not_reg_file.write(filename+'\n')
                continue
            bbox = bbox.astype(np.int32)
            bbox = bbox[:, 0:4]

            # 真值与预测值比较
            neg_idx = 0
            pos_idx = 0
            part_idx = 0
            if len(gt_bbox) <= 0:
                # 极端情况，所有的bbox都是负样本
                for i in bbox:
                    # squrare
                    r = square_bbox(i)
                    # pad
                    size = r[2]
                    crop = np.zeros((size, size, 3), dtype=np.uint8)
                    sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(r, W, H)
                    if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                        log.warning("img shape is: {},{}".format(img.shape[0], img.shape[1]))
                        continue
                    crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
                    out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
                    # 保存
                    output_filename = os.path.splitext(filename)[0]
                    output_filename = output_filename.replace('/', '-')
                    output_filename = output_filename + "_" + str(neg_idx) + ".jpg"
                    output_path = os.path.join(output_neg_dir, output_filename)
                    ret = cv2.imwrite(output_path, out)
                    if ret:
                        neg_idx += 1
                        # line: output_filename 0 0 0 0 0
                        fanno_neg.write(output_filename + ' 0 0 1 1 0\n')
                continue

            # 遍历并判断交并比，来确定是正样本，负样本还是部分人脸样本
            for i in bbox:
                ious = []
                for b in gt_bbox:
                    ious += [IOU(b, i)]

                # 负样本
                if np.max(ious) < IOU_NEG_THRES:
                    r = square_bbox(i)
                    # pad
                    size = r[2]
                    crop = np.zeros((size, size, 3), dtype=np.uint8)
                    sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(r, W, H)
                    if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                        log.warning("img shape is: {},{}".format(img.shape[0], img.shape[1]))
                        continue
                    crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
                    out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
                    # 保存
                    output_filename = os.path.splitext(filename)[0]
                    output_filename = output_filename.replace('/', '-')
                    output_filename = output_filename + "_" + str(neg_idx) + ".jpg"
                    output_path = os.path.join(output_neg_dir, output_filename)
                    ret = cv2.imwrite(output_path, out)
                    if ret:
                        neg_idx += 1
                        # line: output_filename 0 0 0 0 0
                        fanno_neg.write(output_filename + ' 0 0 1 1 0\n')
                    continue
                # 正样本与部分人脸
                ## iou最大的真值框
                max_idx = np.argmax(ious)
                ground_truth = gt_bbox[max_idx]

                ## square and pad
                r = square_bbox(i)
                ## pad(对于超出图像范围的bbox而言）
                size = r[2]
                crop = np.zeros((size, size, 3), dtype=np.uint8)
                sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(r, W, H)
                if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                    log.warning("img shape is: {},{}".format(img.shape[0], img.shape[1]))
                    continue
                crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
                out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
                # ground truth 坐标变换
                ## 变换到crop为原点的坐标
                ox = ground_truth[0] - r[0]
                oy = ground_truth[1] - r[1]
                ow = ground_truth[2]
                oh = ground_truth[3]

                ## 变换到out
                scalor = float(size) / float(OUT_IMAGE_SIZE)
                ox = ox / scalor
                oy = oy / scalor
                ow = ow / scalor
                oh = oh / scalor

                # # 这里保存成左上角点和右下角点
                xmin = ox
                ymin = oy
                xmax = xmin + ow
                ymax = ymin + oh

                ## 判断交并比
                if np.max(ious) > IOU_POS_THRES:
                    #### 正样本
                    output_filename = os.path.splitext(filename)[0]
                    output_filename = output_filename.replace('/', '-')
                    output_filename = output_filename + "_" + str(pos_idx) + ".jpg"
                    output_path = os.path.join(output_pos_dir, output_filename)
                    ret = cv2.imwrite(output_path, out)
                    if ret:
                        pos_idx += 1
                        # line: output_filename 0 0 0 0 0
                        fanno_pos.write(output_filename + ' {} {} {} {} 1\n'.format(xmin, ymin, xmax, ymax))
                elif np.max(ious) > IOU_PART_THRES:
                    #### 部分样本
                    output_filename = os.path.splitext(filename)[0]
                    output_filename = output_filename.replace('/', '-')
                    output_filename = output_filename + "_" + str(part_idx) + ".jpg"
                    output_path = os.path.join(output_part_dir, output_filename)
                    ret = cv2.imwrite(output_path, out)
                    if ret:
                        part_idx += 1
                        # line: output_filename 0 0 0 0 0
                        fanno_part.write(output_filename + ' {} {} {} {} 2\n'.format(xmin, ymin, xmax, ymax))

    # over
    fanno_pos.close()
    fanno_neg.close()
    fanno_part.close()
    not_reg_file.close()

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

    # rnet
    rnet = None
    if net_type == "ONET":
        rnet_weight_path = "./models/rnet_20181212_final.pkl"
        rnet = RNet(test=True)
        LoadWeights(rnet_weight_path, rnet)
        rnet.to(device)

    mt = MTCNN(detectors=[pnet, rnet, None], min_face_size=24, threshold=[0.5, 0.5, 0.5], device=device)
    GenerateData(mt)
    log.info("over...")
