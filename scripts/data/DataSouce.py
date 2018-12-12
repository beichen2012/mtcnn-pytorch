# coding: utf-8
import torch
import numpy as np
import os
import cv2
import random
import torch.utils.data as data


class DataSource(data.Dataset):
    def __init__(self, root_dir, data_file, transform=None, shuffle=True):
        self.root_dir = root_dir
        self.transform = transform
        self.shuffle=shuffle

        self.pos = []
        self.part = []
        self.neg = []
        self.pos_idx = 0
        self.part_idx = 0
        self.neg_idx = 0
        with open(data_file[0]) as f:
            self.pos = f.readlines()
        with open(data_file[1]) as f:
            self.part = f.readlines()
        with open(data_file[2]) as f:
            self.neg = f.readlines()
        if shuffle:
            random.shuffle(self.pos)
            random.shuffle(self.part)
            random.shuffle(self.neg)

    def prepare_batch_sample(self, batch_size):
        # 0，分割样本个数
        neg_num = batch_size // 5 * 3
        part_num = batch_size // 5
        pos_num = batch_size - neg_num - part_num

        # neg
        neg_anno = []
        out_neg_idx = 0
        if self.neg_idx + neg_num >= len(self.neg):
            neg_anno = self.neg[self.neg_idx:]
            out_neg_idx = neg_num - len(neg_anno)
            neg_anno.extend(self.neg[0:out_neg_idx])
            self.neg_idx = out_neg_idx
            if self.shuffle:
                random.shuffle(self.neg)
        else:
            neg_anno = self.neg[self.neg_idx: self.neg_idx + neg_num]
            self.neg_idx += neg_num

        # pos
        pos_anno = []
        out_pos_idx = 0
        if self.pos_idx + pos_num >= len(self.pos):
            pos_anno = self.pos[self.pos_idx:]
            out_pos_idx = pos_num - len(pos_anno)
            pos_anno.extend(self.pos[0:out_pos_idx])
            self.pos_idx = out_pos_idx
            if self.shuffle:
                random.shuffle(self.pos)
        else:
            pos_anno = self.pos[self.pos_idx: self.pos_idx + pos_num]
            self.pos_idx += pos_num

        # part
        part_anno = []
        out_part_idx = 0
        if self.part_idx + part_num > len(self.part):
            part_anno = self.part[self.part_idx:]
            out_part_idx = part_num - len(part_anno)
            part_anno.extend(self.part[0:out_part_idx])
            self.part_idx = out_part_idx
            if self.shuffle:
                random.shuffle(self.part)
        else:
            part_anno = self.part[self.part_idx: self.part_idx + part_num]
            self.part_idx += part_num

        # out
        batch_anno = neg_anno
        batch_anno.extend(pos_anno)
        batch_anno.extend(part_anno)
        random.shuffle(batch_anno)
        return batch_anno

    def get_sample(self, line):
        sample = line.strip().split()
        img = cv2.imread(os.path.join(self.root_dir, sample[0]), 1)
        target = np.array([float(i) for i in sample[1:]])

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def getbatch(self, batch_size):
        batch_anno = self.prepare_batch_sample(batch_size)

        imgs = []
        targets = []
        for sample in batch_anno:
            img, target = self.get_sample(sample)
            imgs += [img]
            targets += [target]

        imgs = np.array(imgs)
        targets = np.array(targets).astype(np.float32)

        return torch.from_numpy(imgs), torch.from_numpy(targets)
