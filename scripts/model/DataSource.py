# coding: utf-8
"""
根据标签列表，生成训练样本和测试样本
label规则：
负样本：0
正样本：1
部分正样本：2
正负样本参与 分类，
正和部分正参与回归
"""
import numpy as np
import cv2
import random
import os


class DataSource:
    def __init__(self, root_dir, train_list, val_list, mean_val=127.5, scale_factor=0.0078125, image_size=12):
        '''
        :param root_dir:  根目录
        :param train_list:  neg.txt, pos.txt, part.txt, 列表，长度为3
        :param val_list:  neg.txt, pos.txt, part.txt, 列表，长度为3
        :param mean_val:    平均值
        :param scale_factor:    系数
        :param image_size:      图像尺寸(12,24,48)
        :return:
        '''
        self.root_dir = root_dir

        # train,val
        self.train_anno = []
        self.test_anno = []
        for i in range(0,3):
            with open(train_list[i], 'r') as f:
                self.train_anno += [f.readlines()]
            with open(val_list[i], 'r') as f:
                self.test_anno += [f.readlines()]

        #
        self.mean_val = mean_val
        self.scale_factor = scale_factor
        self.train_idx = [0,0,0]
        self.test_idx = [0,0,0]
        self.image_size = np.float(image_size)


    def prepareData(self, anno, current_idx, batch_size):
        '''
        准备数据
        :param anno: train_anno or val_anno
        :param current_idx:  列表，标识anno当前的索引
        :param batch_size:  N
        :return:
        '''
        #0，分割样本个数
        neg_num = batch_size // 5 * 3
        pos_num = batch_size // 5
        part_num = batch_size - neg_num - pos_num

        # 1, 准备样本标注文件
        ## 1.1 负样本
        neg_anno = []
        out_neg_idx = 0
        if current_idx[0] + neg_num >= len(anno[0]):
            neg_anno = anno[0][current_idx[0]:]
            out_neg_idx = neg_num - len(neg_anno)
            neg_anno.extend(anno[0][0:out_neg_idx])
            random.shuffle(anno[0])
        else:
            neg_anno = anno[0][current_idx[0] : current_idx[0] + neg_num]
            out_neg_idx = current_idx[0] + neg_num

        ## 1.2 正样本
        pos_anno = []
        out_pos_idx = 0
        if current_idx[1] + pos_num >= len(anno[1]):
            pos_anno = anno[1][current_idx[1]:]
            out_pos_idx = pos_num - len(pos_anno)
            pos_anno.extend(anno[1][0:out_pos_idx])
            random.shuffle(anno[1])
        else:
            pos_anno = anno[1][current_idx[1]: current_idx[1] + pos_num]
            out_pos_idx = current_idx[1] + pos_num
        ## 1.3 部分样本
        part_anno = []
        out_part_idx = 0
        if current_idx[2] + part_num >= len(anno[2]):
            part_anno = anno[2][current_idx[2]:]
            out_part_idx = part_num - len(part_anno)
            part_anno.extend(anno[2][0:out_part_idx])
            random.shuffle(anno[2])
        else:
            part_anno = anno[2][current_idx[2]: current_idx[2] + part_num]
            out_part_idx = current_idx[2] + part_num

        ## 2. 组合样本
        batch_anno = neg_anno
        batch_anno.extend(pos_anno)
        batch_anno.extend(part_anno)
        random.shuffle(batch_anno)

        ## 3. split to data, label, bbox
        data = []
        label = []
        bbox = []
        for line in batch_anno:
            i = line.strip()
            i = i.split()
            img_path = os.path.join(self.root_dir, i[0])

            # label
            dl = np.int32(i[1])
            # bbox(original size)
            x0 = np.float32(i[2])
            y0 = np.float32(i[3])
            w0 = np.float32(i[4])
            h0 = np.float32(i[5])

            # img
            img = cv2.imread(img_path, 1)
            if img is None:
                AssertionError("image [{}] cannot be readed!".format(img_path))

            # add to list(original image)
            data += [img]
            label += [dl]

            # add bbox to aug
            # bbox_pts += [ia.BoundingBoxesOnImage(
            #     [ia.BoundingBox(x0, y0, x0 + w0, y0 + h0)], shape=img.shape)]
            bbox += [(x0, y0, w0, h0)]

        # data augment
        for i in range(len(data)):
            if random.random() > 0.5:
                data[i] = cv2.flip(data[i], 1)
                x0, y0, w0, h0 = bbox[i]
                y0 = int(self.image_size - y0)
                bbox[i] = (x0, y0, w0, h0)

        # scale and resize
        assert (len(data) == len(bbox))
        for i in range(len(data)):
            # convert to float and preprocess
            src = data[i].astype(np.float32)
            src -= self.mean_val
            src *= self.scale_factor
            # to CHW
            src = src.swapaxes(1, 2).swapaxes(0, 1)
            data[i] = src

            # scale coordinate
            ele = (bbox[i][0] / self.image_size,
                   bbox[i][1] / self.image_size,
                   np.log(bbox[i][2] / self.image_size),
                   np.log(bbox[i][3] / self.image_size))

            bbox[i] = ele

        # convert to ndarray
        data = np.array(data)
        label = np.array(label, dtype=np.int64)
        bbox = np.array(bbox)
        bbox = bbox.astype(np.float32)

        return data, label, bbox, [out_neg_idx, out_pos_idx, out_part_idx]

    def getTrainData(self, batch_size):
        data, label, bbox, out_idx = self.prepareData(
            self.train_anno, self.train_idx, batch_size)
        self.train_idx = out_idx
        return data, label, bbox

    def getTestData(self, batch_size):
        data, label, bbox, out_idx = self.prepareData(
            self.test_anno, self.test_idx,  batch_size)
        self.test_idx = out_idx
        return data, label, bbox
