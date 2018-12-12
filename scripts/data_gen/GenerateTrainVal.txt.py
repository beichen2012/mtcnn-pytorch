# coding: utf-8
"""
compose and split the whole dataset to train,val
"""
import sys
sys.path.append(sys.path[0] + "/../")
import os
import random
import time
from util.Logger import Logger
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger
import numpy as np

# PNET, RNET , ONET
net_type = 'PNET'

if net_type == 'PNET':
    postfix = 'p'
elif net_type == 'RNET':
    postfix = 'r'
else:
    postfix = 'o'


root_dir = "../../dataset"
pos_re_dir = "train_faces_{}/pos/".format(postfix)
neg_re_dir = "train_faces_{}/neg/".format(postfix)
part_re_dir = "train_faces_{}/part/".format(postfix)

neg_train_path = 'train_neg_{}.txt'.format(postfix)
pos_train_path = 'train_pos_{}.txt'.format(postfix)
part_train_path = 'train_part_{}.txt'.format(postfix)

neg_val_path = 'val_neg_{}.txt'.format(postfix)
pos_val_path = 'val_pos_{}.txt'.format(postfix)
part_val_path = 'val_part_{}.txt'.format(postfix)


f = open(os.path.join(root_dir, pos_re_dir + "anno_pos.txt"))
pos = f.readlines()
f.close()

f = open(os.path.join(root_dir, neg_re_dir + "anno_neg.txt"))
neg = f.readlines()
f.close()

f = open(os.path.join(root_dir, part_re_dir + "anno_part.txt"))
part = f.readlines()
f.close()

#
npos = len(pos)
nneg = len(neg)
npart = len(part)

log.info("number: pos -> {}, neg -> {}, part -> {}".format(npos, nneg, npart))


# 85:15的比例
for i in range(4):
    random.shuffle(pos)
    random.shuffle(neg)
    random.shuffle(part)

#
pos = [pos_re_dir + i.strip() + '\n' for i in pos]
neg = [neg_re_dir + i.strip() + '\n' for i in neg]
part = [part_re_dir + i.strip() + '\n' for i in part]


pos_train_num = int(npos * 0.85)
neg_train_num = int(nneg * 0.85)
part_train_num = int(npart * 0.85)

log.info("train: neg num -> {}, pos num -> {}, part num -> {} val: neg num -> {}, pos num -> {}, part num -> {}".format(
    neg_train_num, pos_train_num, part_train_num,
    nneg - neg_train_num, npos - pos_train_num, npart - part_train_num
))

#
def save2File(path, anno):
    with open(path, 'w') as f:
        for i in anno:
            f.write(i)


neg_train = neg[0:neg_train_num]
neg_val = neg[neg_train_num:]

pos_train = pos[0:pos_train_num]
pos_val = pos[pos_train_num:]

part_train = part[0:part_train_num]
part_val = part[part_train_num:]

save2File(os.path.join(root_dir, neg_train_path), neg_train)
save2File(os.path.join(root_dir, pos_train_path), pos_train)
save2File(os.path.join(root_dir, part_train_path), part_train)

save2File(os.path.join(root_dir, neg_val_path), neg_val)
save2File(os.path.join(root_dir, pos_val_path), pos_val)
save2File(os.path.join(root_dir, part_val_path), part_val)


log.info("done")


