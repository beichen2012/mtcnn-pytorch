# coding: utf-8
import torch
import numpy as np
import os
import cv2
import random
import lmdb
import torch.utils.data as data


class DataSource(data.Dataset):
    def __init__(self, data_file, transform=None, shuffle=True, image_shape=(12,12,3), ratio=3):
        self.transform = transform
        self.shuffle = shuffle
        self.image_shape = image_shape
        self.ratio = int(ratio)

        self.pos = []
        self.part = []
        self.neg = []
        self.pos_idx = 0
        self.part_idx = 0
        self.neg_idx = 0

        # pos
        self.pos_env_image = lmdb.open(data_file[0],
                                       max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.pos_env_image.begin(write=False) as txn:
            self.pos = [key for key, _ in txn.cursor()]
        self.pos_env_label = lmdb.open(data_file[1],
                                       max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        # part
        self.part_env_image = lmdb.open(data_file[2],
                                        max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.part_env_image.begin(write=False) as txn:
            self.part = [key for key, _ in txn.cursor()]
        self.part_env_label = lmdb.open(data_file[3],
                                        max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        # neg
        self.neg_env_image = lmdb.open(data_file[4],
                                       max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.neg_env_image.begin(write=False) as txn:
            self.neg = [key for key, _ in txn.cursor()]
        self.neg_env_label = lmdb.open(data_file[5],
                                       max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        if shuffle:
            random.shuffle(self.pos)
            random.shuffle(self.part)
            random.shuffle(self.neg)

    def prepare_batch_sample(self, batch_size):
        # 0，分割样本个数
        neg_num = batch_size // (self.ratio + 2) * self.ratio
        part_num = batch_size // (self.ratio + 2)
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
        batch_anno = [(i, "0") for i in neg_anno]
        batch_anno.extend([(i, "1") for i in pos_anno])
        batch_anno.extend([(i, "2") for i in part_anno])
        random.shuffle(batch_anno)
        return batch_anno

    def get_sample(self, line):
        label = line[1]
        img = None
        target = None
        if line[1] == "0":
            with self.neg_env_image.begin(write=False) as txn:
                img = txn.get(line[0])
            with self.neg_env_label.begin(write=False) as txn:
                target = txn.get(line[0])
        elif line[1] == "1":
            with self.pos_env_image.begin(write=False) as txn:
                img = txn.get(line[0])
            with self.pos_env_label.begin(write=False) as txn:
                target = txn.get(line[0])
        else:
            with self.part_env_image.begin(write=False) as txn:
                img = txn.get(line[0])
            with self.part_env_label.begin(write=False) as txn:
                target = txn.get(line[0])

        img = np.frombuffer(img, dtype=np.uint8).copy().reshape(self.image_shape)
        target = np.frombuffer(target, dtype=np.float32).copy()

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
