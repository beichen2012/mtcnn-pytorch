# coding: utf-8
import os
import torch
from Nets import *
from DataSource import *
from train import *
from Logger import *
log = Logger("pnet.log", level='debug').logger

max_iter = 800000
train_batch = 768
test_batch = 384
test_iter = 10000

test_interval = 50000
display = 500

# learning rate
base_lr = 0.1
momentum = 0.9

stepsize = [10000, 50000, 100000, 250000, 400000,500000,600000]
gamma = 0.1

save_interval = 50000
#50000
save_prefix = "./models/pnet_20180926"

device = torch.device('cuda:0')
model = PNet()
# 数据源
root_dir = r"../../dataset/"
INPUT_IMAGE_SIZE = 12

topk = 0.7

train_anno_path = []
val_anno_path = []
train_anno_path += [os.path.join(root_dir, "train_neg_p.txt")]
train_anno_path += [os.path.join(root_dir, "train_pos_p.txt")]
train_anno_path += [os.path.join(root_dir, "train_part_p.txt")]

val_anno_path += [os.path.join(root_dir, "val_neg_p.txt")]
val_anno_path += [os.path.join(root_dir, "val_pos_p.txt")]
val_anno_path += [os.path.join(root_dir, "val_part_p.txt")]


if __name__ == '__main__':

    ds = DataSource(root_dir, train_anno_path, val_anno_path, 127.5, 0.0078125, INPUT_IMAGE_SIZE)

    sp = SolverParam(base_lr, momentum, stepsize, gamma, test_batch,
                     test_iter, test_interval,
                     train_batch, max_iter, display, save_interval, save_prefix, topk)

    t = TrainInst(model, device, ds, sp, log, INPUT_IMAGE_SIZE)
    t.run()
