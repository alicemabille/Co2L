'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/util.py
'''

from __future__ import print_function

import math
import PIL
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler
from torchvision import transforms
from scipy.stats import multivariate_normal

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def transform(opt, type=PIL.Image) :
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    #augment inputs
    augments = [
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0)
    ]
    if type==PIL.Image :
        augments.append(transforms.ToTensor())
    augments.append(normalize)
    train_transform = transforms.Compose(augments)
    return train_transform


def apply_transform(x: torch.Tensor, transform, autosqueeze=False, y:torch.Tensor=None) -> torch.Tensor:
    """Applies a transform to a batch of images.

    Args:
        x: a batch of images.
        transform: the transform to apply.
        autosqueeze: whether to automatically squeeze the output tensor.

    Returns:
        The transformed batch of images.
    """
    if transform is None:
        return x
    if isinstance(x, PIL.Image.Image):
        return transform(x)
    
    # Check if the transform is an instance of TwoCropTransform
    if isinstance(transform, TwoCropTransform):
        # Apply the transform to each image in the batch
        out = [torch.stack([transform(xi)[i] for xi in x.cpu()]).to(x.device) for i in range(2)]
        
        # If autosqueeze is True and the batch size is 1, squeeze the output
        if autosqueeze and out[0].shape[0] == 1:
            out = [o.squeeze(0) for o in out]
        out = torch.cat(out)
        print('shape of transformed batch of images : ', out.shape)

        if y != None :
            print('doubling labels')
            y = y.repeat(2)
            print('shape of labels', y.shape)
            return (out, y)

        return out
    
    # Handle the case where the transform is not TwoCropTransform
    out = torch.stack([transform(xi) for xi in x.cpu()]).to(x.device)
        
    if autosqueeze and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("learning rate : ", lr)


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        print("learning rate : ", lr)


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(model, optimizer, save_file):
    print('==> Loading...' + save_file)
    loaded = torch.load(save_file)

    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    del loaded

    return model, optimizer

def worker_init_cuda_debug(worker_id):
    print(f"Worker {worker_id} is using CUDA device: {torch.cuda.current_device()}")