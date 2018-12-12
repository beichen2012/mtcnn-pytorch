# coding: utf-8
import torch

def SaveCheckPoint(path, net, optimizer, scheduler, epoch):
    checkpoint = {
        'weights': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
