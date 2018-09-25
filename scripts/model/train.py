# coding: utf-8
import torch
import torch.optim as optim
from DataSource import *
from Nets import *
import numpy as np


class SolverParam:
    def __init__(self, base_lr, momentum, stepsize,
                 gamma, test_batch, test_iter, test_interval,
                 train_batch, max_iter, display,
                 save_interval, save_prefix, topk):
        self.base_lr = base_lr
        self.momentum = momentum
        self.stepsize = stepsize
        self.gamma = gamma

        self.test_batch = test_batch
        self.test_iter = test_iter
        self.test_interval = test_interval

        self.train_batch = train_batch
        self.max_iter = max_iter
        self.display = display
        self.save_interval = save_interval
        self.save_prefix = save_prefix
        self.topk = topk


class TrainInst:
    """
    train instance class
    """

    def __init__(self, model, device, dataset, sp, log, image_size):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.sp = sp
        self.log = log
        self.image_size = image_size

    def train(self, idx):
        self.model.train()
        data, label, bbox = self.dataset.getTrainData(self.sp.train_batch)
        # to tensor
        data = torch.from_numpy(data).to(self.device)
        label = torch.from_numpy(label).to(self.device)
        bbox = torch.from_numpy(bbox).to(self.device)

        self.opter.zero_grad()
        pred_cls, pred_bbox = self.model(data)
        loss_cls = AddClsLoss(pred_cls, label, self.sp.topk)
        loss_box = AddRegLoss(pred_bbox, label, bbox)
        loss = loss_cls + loss_box
        loss.backward()
        self.opter.step()
        self.scheduler.step()
        # display train
        if idx % self.sp.display == 0:
            acc = AddClsAccuracy(pred_cls, label)
            mmap = AddBoxMap(pred_bbox, label, bbox, self.image_size, self.image_size)
            self.log.info(
                "train net -> iter: {}, lr: {}, cls loss: {:.4f}, cls acc: {:.4f}, bbox loss: {:.4f}, bbox map: {:.4f}".format(
                    idx, self.opter.param_groups[0]['lr'], loss_cls.item(), acc, loss_box, mmap))

        return

    def validation(self, idx):
        self.model.eval()
        test_cls_loss = []
        test_box_loss = []
        test_cls_acc = []
        test_box_map = []
        with torch.no_grad():
            for i in range(0, self.sp.test_iter):
                data, label, bbox = self.dataset.getTestData(self.sp.test_batch)
                # to tensor
                data = torch.from_numpy(data).to(self.device)
                label = torch.from_numpy(label).to(self.device)
                bbox = torch.from_numpy(bbox).to(self.device)
                pred_cls, pred_bbox = self.model(data)
                loss_cls = AddClsLoss(pred_cls, label, self.sp.topk)
                loss_box = AddRegLoss(pred_bbox, label, bbox)
                acc = AddClsAccuracy(pred_cls, label)
                map = AddBoxMap(pred_bbox, label, bbox, self.image_size, self.image_size)

                test_cls_acc += [acc]
                test_cls_loss += [loss_cls.item()]
                test_box_loss += [loss_box.item()]
                test_box_map += [map]
        #
        test_cls_loss = np.array(test_cls_loss)
        test_box_loss = np.array(test_box_loss)
        test_cls_acc = np.array(test_cls_acc)
        test_box_map = np.array(test_box_map)
        self.log.info(
            "test net -> iter: {}, lr: {}, cls loss: {:.4f}, cls acc: {:.4f}, bbox loss: {:.4f}, bbox map: {:.4f}".format(
                idx, self.opter.param_groups[0]['lr'], np.mean(test_cls_loss), np.mean(test_cls_acc),
                np.mean(test_box_loss), np.mean(test_box_map)))

    def run(self):
        # the model definition
        self.model = self.model.to(self.device)
        self.opter = optim.SGD(self.model.parameters(), lr=self.sp.base_lr, momentum=self.sp.momentum)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opter, self.sp.stepsize, gamma=self.sp.gamma)

        # run epoch
        for i in range(0, self.sp.max_iter):
            self.train(i)
            # fot test
            if i % self.sp.test_interval == 0 and i != 0:
                self.validation(i)

            # for save
            if i % self.sp.save_interval == 0 and i != 0:
                path = self.sp.save_prefix + "_iter_{}.pkl".format(i)
                torch.save(self.model.state_dict(), path)
                self.log.info("save model: {}".format(path))
        self.log.info("optimize done...")
        path = self.sp.save_prefix + "_final.pkl"
        torch.save(self.model.state_dict(), path)
        self.log.info("save final pkl...")


if __name__ == '__main__':
    print('cannot run by self!')
