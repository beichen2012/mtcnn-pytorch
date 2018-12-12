# coding: utf-8
# coding: utf-8
from data.DataSouce import DataSource
from data.augmentation import *
import os
import random
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from util.torchutil import SaveCheckPoint
from util.Logger import Logger
from Nets import *


if not os.path.exists("./log/"):
    os.mkdir("./log/")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                             time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger

USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

pre_checkpoint = None
resume = False

train_batch = 5
display = 100

base_lr = 0.01
clip_grad = 120.0
momentum = 0.9
gamma = 0.1
weight_decay = 0.0005
stepsize = [50000, 90000, 120000, 150000]
max_iter = 170000

save_interval = 10000

save_dir = "./models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_prefix = save_dir + "/pnet_20181212"


root_dir = r"../../dataset/"
INPUT_IMAGE_SIZE = 12

topk = 0.7
MEANS = [127.5,127.5,127.5]
train_anno_path = []
val_anno_path = []

train_anno_path += [os.path.join(root_dir, "train_pos_p.txt")]
train_anno_path += [os.path.join(root_dir, "train_part_p.txt")]
train_anno_path += [os.path.join(root_dir, "train_neg_p.txt")]


val_anno_path += [os.path.join(root_dir, "val_pos_p.txt")]
val_anno_path += [os.path.join(root_dir, "val_part_p.txt")]
val_anno_path += [os.path.join(root_dir, "val_neg_p.txt")]

def train():
    start_epoch = 0
    # dataset
    train_dataset = DataSource(root_dir, train_anno_path, transform=Compose([
        RandomMirror(0.5), SubtractFloatMeans(MEANS), ToPercentCoords()
    ]))

    val_dataset = DataSource(root_dir, val_anno_path, transform=None, shuffle=False)

    # net
    net = PNet()

    # optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, stepsize, gamma)

    # device
    if USE_CUDA:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if pre_checkpoint:
        cp = torch.load(pre_checkpoint)
        net.load_state_dict(cp['weights'])
        log.info("=> load state dict from {}...".format(pre_checkpoint))
        if resume:
            optimizer.load_state_dict(cp['optimizer'])
            scheduler.load_state_dict(cp['scheduler'])
            start_epoch = cp['epoch']
            log.info("=> resume from epoch: {}, now the lr is: {}".format(start_epoch, optimizer.param_groups[0]['lr']))

    net.to(device)

    k = 0
    loss = 0
    for epoch in range(start_epoch, max_iter + 1):
        net.train()
        images, targets = train_dataset.getbatch(train_batch)
        images = images.to(device)
        targets = targets.to(device)

        out = net(images)

        optimizer.zero_grad()

        loss_cls = AddClsLoss(out, targets, topk)
        loss_reg = AddRegLoss(out, targets)

        loss = loss_cls + loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        # if k % display == 0:
        #     log.info(
        #         "iter/epoch: {}/{}, lr: {}, loss: {:.4f},arm_loc: {:.4f}, arm_conf: {:.4f},odm_loc: {:.4f}, odm_conf: {:.4f},time/iter: {:.3f} s".format(
        #             k,
        #             epoch,
        #             optimizer.param_groups[0]['lr'],
        #             loss.item(),
        #             arm_loss_l.item(),
        #             arm_loss_c.item(),
        #             odm_loss_l.item(),
        #             odm_loss_c.item(),
        #             t1 - t0))
        # if k % save_interval == 0:
        #     path = save_prefix + "_iter_{}.pkl".format(k)
        #     SaveCheckPoint(path, net, optimizer, scheduler, epoch)
        #     log.info("=> save model: {}".format(path))
        # k += 1


    log.info("optimize done...")
    path = save_prefix + "_final.pkl"
    SaveCheckPoint(path, net, optimizer, scheduler, max_iter)
    log.info("=> save model: {} ...".format(path))


if __name__ == '__main__':
    train()


















#
#
# import os
# import torch
# from Nets import *
# from DataSource import *
# from train import *
# from Logger import *
# log = Logger("pnet.log", level='debug').logger
#
# max_iter = 800000
# train_batch = 1000
# test_batch = 500
# test_iter = 10000
#
# test_interval = 50000
# display = 500
#
# # learning rate
# base_lr = 0.1
# momentum = 0.9
#
# stepsize = [10000, 50000, 100000, 250000, 400000,500000,600000]
# gamma = 0.1
#
# save_interval = 50000
# #50000
# save_prefix = "./models/pnet_20180926"
#
# device = torch.device('cuda:0')
# model = PNet()
# # 数据源
# root_dir = r"../../dataset/"
# INPUT_IMAGE_SIZE = 12
#
# topk = 0.7
#
# train_anno_path = []
# val_anno_path = []
# train_anno_path += [os.path.join(root_dir, "train_neg_p.txt")]
# train_anno_path += [os.path.join(root_dir, "train_pos_p.txt")]
# train_anno_path += [os.path.join(root_dir, "train_part_p.txt")]
#
# val_anno_path += [os.path.join(root_dir, "val_neg_p.txt")]
# val_anno_path += [os.path.join(root_dir, "val_pos_p.txt")]
# val_anno_path += [os.path.join(root_dir, "val_part_p.txt")]
#
#
# if __name__ == '__main__':
#
#     ds = DataSource(root_dir, train_anno_path, val_anno_path, 127.5, 0.0078125, INPUT_IMAGE_SIZE)
#
#     sp = SolverParam(base_lr, momentum, stepsize, gamma, test_batch,
#                      test_iter, test_interval,
#                      train_batch, max_iter, display, save_interval, save_prefix, topk)
#
#     t = TrainInst(model, device, ds, sp, log, INPUT_IMAGE_SIZE)
#     t.run()
