# coding: utf-8
"""
读取wider face 数据集
生成正样本和负样本。
正样本标签为：1
负样本标签为：0
部分人脸样本标签为：2
综合标签为： img_path xmin ymin xmax ymax label

注意：保存的图片缩放到了12*12，bbox的坐标也是相对于12*12的
"""
import sys
sys.path.append(sys.path[0] + "/../")
import os
import numpy as np
import random
import cv2
from pylab import plt
from util.utility import *
from util.Logger import Logger
import time
import lmdb
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger

# 小于该人脸的就不要了
# 太小的话，会有太多的误检
MIN_FACE_SIZE = 40
IOU_POS_THRES = 0.65
IOU_NEG_THRES = 0.3
IOU_PART_THRES = 0.4

## 负样本个数
neg_samples = 100
## 正样本个数
pos_samples = 20

OUT_IMAGE_SIZE = 12

# path to wider face'
root_dir = r'~/dataset/WIDER_FACE'
root_dir = os.path.expanduser(root_dir)

train_dir = os.path.join(root_dir, 'WIDER_train/images')
val_dir = os.path.join(root_dir, 'WIDER_val/images')
anno_dir = os.path.join(root_dir, 'wider_face_split')

# path to output root dir
output_root_dir = r"../dataset/train_faces_p"
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
# label_pos = os.path.join(output_pos_dir, "anno_pos.txt")
# label_neg = os.path.join(output_neg_dir, "anno_neg.txt")
# label_part = os.path.join(output_part_dir, "anno_part.txt")

#
# fanno_pos = open(label_pos, "w")
# fanno_neg = open(label_neg, "w")
# fanno_part = open(label_part, "w")

# output lmdb
# 1 TB
LMDB_MAP_SIZE = 1099511627776
env_pos_image = lmdb.open(os.path.join(output_pos_dir, "image_pos"), map_size=LMDB_MAP_SIZE)
env_pos_label = lmdb.open(os.path.join(output_pos_dir, "label_pos"), map_size=LMDB_MAP_SIZE)

env_part_image = lmdb.open(os.path.join(output_part_dir, "image_part"), map_size=LMDB_MAP_SIZE)
env_part_label = lmdb.open(os.path.join(output_part_dir, "label_part"), map_size=LMDB_MAP_SIZE)

env_neg_image = lmdb.open(os.path.join(output_neg_dir, "image_neg"), map_size=LMDB_MAP_SIZE)
env_neg_label = lmdb.open(os.path.join(output_neg_dir, "label_neg"), map_size=LMDB_MAP_SIZE)

global_idx_pos = 0
global_idx_part = 0
global_idx_neg = 0

# lmdb
txn_pos_image = env_pos_image.begin(write=True)
txn_pos_label = env_pos_label.begin(write=True)

txn_part_image = env_part_image.begin(write=True)
txn_part_label = env_part_label.begin(write=True)

txn_neg_image = env_neg_image.begin(write=True)
txn_neg_label = env_neg_label.begin(write=True)


# 1, 读取标签(从train库里）
anno_file = os.path.join(anno_dir, "wider_face_train_bbx_gt.txt")
with open(anno_file, "r") as f:
    inner_neg_idx = 0
    inner_pos_idx = 0
    inner_part_idx = 0
    while True:
        filename = f.readline()
        if not filename:
            break
        filename = filename.strip('\n')
        log.info("now process -> {}".format(filename))
        face_num = f.readline()
        face_num = int(face_num)
        # img
        img = cv2.imread(os.path.join(train_dir, filename))
        if img is None:
            log.warning("error to load image {}", filename)
            continue
        # 读取真值 bbox
        H, W, C = img.shape
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
        if len(gt_bbox) <= 0:
            continue

        # 下面随机生成正样本和负样本
        ## 负样本生成
        neg_idx = 0
        while neg_idx <= neg_samples:
            size = random.randrange(OUT_IMAGE_SIZE, int(np.min((H, W)) / 2))
            nx = random.randrange(1, W - size)
            ny = random.randrange(1, H - size)
            crop_box = (nx, ny, size, size)
            score = []
            for i in gt_bbox:
                score_1 = IOU(crop_box, i)
                score += [score_1]
            # score_iom = []
            # for i in gt_bbox:
            #     score_iom += [IOM(crop_box, i)]
            score = np.asarray(score)
            #score_iom = np.asarray(score_iom)
            # and np.max(score_iom) < IOU_PART_THRES:
            if np.max(score) < IOU_NEG_THRES:
                crop = img[crop_box[1]: crop_box[1] + crop_box[3], crop_box[0]:crop_box[0] + crop_box[2]]
                # 保存原始图片
                out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
                label = np.array([0,0,1,1,0], dtype=np.float32)
                bo = out.tostring()
                bl = label.tostring()

                txn_neg_image.put("{}".format(global_idx_neg).encode("ascii"), bo)
                txn_neg_label.put("{}".format(global_idx_neg).encode("ascii"), bl)
                neg_idx += 1
                global_idx_neg += 1
                inner_neg_idx += 1

        ## 正样本、部分人脸样本生成
        pos_idx = 0
        part_idx = 0
        for box in gt_bbox:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            ### 如果人脸太小，就不要了（所说不容易训练）
            if np.max((w,h)) < MIN_FACE_SIZE or x < 0 or y < 0:
                continue
            ### 生成与box有重叠的负样本（个数随机，不用太多）
            for i in range(10):
                size = random.randrange(OUT_IMAGE_SIZE, int(np.min((W, H)) / 2))
                dx = random.randrange(int(np.max((-size, -x))), w)
                dy = random.randrange(int(np.max((-size, -y))), h)
                nx = np.max((0, x + dx))
                ny = np.max((0, y + dy))

                if nx + size > W or ny + size > H:
                    continue

                crop_box = np.array([nx, ny, size, size])
                ious = []
                for b in gt_bbox:
                    ious += [IOU(b, crop_box)]
                ious = np.array(ious)
                #score_iom = []
                #for i in gt_bbox:
                #    score_iom += [IOM(crop_box, i)]
                # and np.max(score_iom) < IOU_PART_THRES
                if np.max(ious) < IOU_NEG_THRES :
                    crop = img[ny:ny+size, nx:nx+size]
                    out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
                    label = np.array([0,0,1,1,0], dtype=np.float32)
                    txn_neg_image.put("{}".format(global_idx_neg).encode("ascii"), out.tostring())
                    txn_neg_label.put("{}".format(global_idx_neg).encode("ascii"), label.tostring())
                    neg_idx += 1
                    global_idx_neg += 1
                    inner_neg_idx += 1

            ### 生成正样本
            for i in range(pos_samples):
                size = random.randrange(int(np.min((w,h)) * 0.8), int(np.ceil(1.25 * np.max((w,h)))))
                dx = random.randrange(int(-w * 0.2), int(w * 0.2))
                dy = random.randrange(int(-h * 0.2), int(h * 0.2))
                nx = np.max((x + w / 2 + dx - size / 2), 0)
                ny = np.max((y + h / 2 + dy - size / 2), 0)
                nx = int(nx)
                ny = int(ny)
                if nx < 0:
                    nx = 0
                if ny < 0:
                    ny = 0
                if nx + size > W or ny + size > H:
                    continue
                if size < OUT_IMAGE_SIZE / 2:
                    continue

                #iou
                crop_box = np.array([nx, ny, size, size])
                iou = IOU(box, crop_box)
                # log.info("{} {} {} {}".format(nx, ny, size, size))
                crop = img[ny: ny+size, nx:nx+size]
                out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))

                # plt.figure()
                # plt.subplot(121)
                # plt.imshow(crop[:,:,::-1])
                # plt.subplot(122)
                # plt.imshow(out[:,:,::-1])
                # plt.title(str(iou))
                # plt.show()

                scalor = float(size) / float(OUT_IMAGE_SIZE)

                # ground truth 坐标变换
                ## 变换到crop
                ox = x - nx
                oy = y - ny
                ow = w
                oh = h
                # ## 变换到out
                ox = ox / scalor
                oy = oy / scalor
                ow = ow / scalor
                oh = oh / scalor
                #
                # # 这里保存成左上角点和右下角点
                xmin = ox
                ymin = oy
                xmax = xmin + ow
                ymax = ymin + oh

                if iou > IOU_POS_THRES:
                    #### 正样本
                    label = np.array([xmin, ymin, xmax, ymax, 1], dtype=np.float32)
                    txn_pos_image.put("{}".format(global_idx_pos).encode("ascii"), out.tostring())
                    txn_pos_label.put("{}".format(global_idx_pos).encode("ascii"), label.tostring())
                    pos_idx += 1
                    global_idx_pos += 1
                    inner_pos_idx += 1
                elif iou > IOU_PART_THRES:
                    #### 部分样本
                    label = np.array([xmin, ymin, xmax, ymax, 2], dtype=np.float32)
                    txn_part_image.put("{}".format(global_idx_part).encode("ascii"), out.tostring())
                    txn_part_label.put("{}".format(global_idx_part).encode("ascii"), label.tostring())
                    part_idx += 1
                    global_idx_part += 1
                    inner_part_idx += 1
        log.info("neg num: {}, pos num:{}, part num: {}".format(global_idx_neg, global_idx_pos, global_idx_part))
        if inner_neg_idx > 1000:
            txn_neg_image.commit()
            txn_neg_label.commit()
            txn_neg_image = env_neg_image.begin(write=True)
            txn_neg_label = env_neg_label.begin(write=True)
            inner_neg_idx = 0
            log.info("now commit netative lmdb")
        if inner_part_idx > 1000:
            txn_part_image.commit()
            txn_part_label.commit()
            txn_part_image = env_part_image.begin(write=True)
            txn_part_label = env_part_label.begin(write=True)
            inner_part_idx = 0
            log.info("now commit part lmdb")

        if inner_pos_idx > 1000:
            txn_pos_image.commit()
            txn_pos_label.commit()
            txn_pos_image = env_pos_image.begin(write=True)
            txn_pos_label = env_pos_label.begin(write=True)
            inner_pos_idx = 0
            log.info("now commit pos lmdb")
# over

log.info("process done!")
txn_neg_image.commit()
txn_neg_label.commit()
txn_part_image.commit()
txn_part_label.commit()
txn_pos_image.commit()
txn_pos_label.commit()

env_neg_image.close()
env_neg_label.close()
env_part_image.close()
env_part_label.close()
env_pos_image.close()
env_pos_label.close()