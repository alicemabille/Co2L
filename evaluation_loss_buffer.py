'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_linear.py
'''

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils import conf
from utils.lws_buffer import Buffer
from utils.ce_buffer import set_loader
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from utils.util import set_optimizer

from networks.resnet_big import SupConResNet, LinearClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0, help='Use all classes if None else learned tasks so far')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny-imagenet'], help='dataset')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--mem_size', type=int, default=200) #buffer size
    parser.add_argument('--n_bin', type=int, default=4) #buffer bins

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--logpt', type=str, default='',
                        help='path to logs')
    parser.add_argument('--tensorboard', action='store_true',
                    help='use tensorboard for logging and visualization')

    opt = parser.parse_args()

    # set the path according to the environment
    #opt.data_folder = '~/data/'
    opt.data_folder = '/data/alicmabi11/datasets'

    opt.tb_folder = os.path.join(opt.ckpt, 'tensorboard', 'linear_eval')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))



    opt.origin_ckpt = opt.ckpt
    opt.ckpt = os.path.join(opt.ckpt, 'last_{target_task}.pth'.format(target_task=opt.target_task))
    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    acc = 0.0
    cnt = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images) # train classifier on features, the head of SupConResNet is only used for computing its loss
        output = classifier(features.detach()) #detach to train only the classifier without computing gradient of ResNet's parameters
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc += (output.argmax(1) == labels).float().sum().item()
        cnt += bsz

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print("predictions :",output.argmax(1))
            print("labels :",labels)
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=acc/cnt*100.))
            sys.stdout.flush()

    return losses.avg, acc/cnt*100.


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0


    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #
            cls_list = np.unique(labels.cpu())
            correct_all = (output.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (output[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

            for c in cls_list:
                #print("class ", c)
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))

    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
    return losses.avg, top1.avg, corr, cnt, correct_task/np.sum(cnt)*100.


def main():
    best_acc = 0
    task_acc = 0
    opt = parse_option()

    #initialize buffer
    device = conf.get_device()
    buffer = Buffer(opt.mem_size, device, n_tasks=opt.target_task,
                            attributes=['examples', 'labels', 'logits', 'task_labels', 'loss_values'],
                            n_bin=opt.n_bin)

    if opt.target_task is not None:
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            buffer.load(os.path.join(opt.logpt, f'loss_buffer_{opt.target_task}.pth'))
            buffer_data = buffer.get_data(opt.mem_size)
            replay_indices = buffer_data[0].to(torch.int)
            print('replay_indices type: ', type(replay_indices))
            print('replay_indices length: ', len(replay_indices))
            print('replay_indices first element: ', replay_indices[0])
            print('replay_indices first element type: ', type(replay_indices[0]))
            print('number of buffered samples : ', buffer.num_examples)

    print('number of samples to replay : ', len(replay_indices))

    # build data loader
    train_loader, val_loader, cls_num_list = set_loader(opt, replay_indices)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    print( optimizer.param_groups[0]['lr'])

    ########################## TENSORBOARD #############################
    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=opt.tb_folder)
    ####################################################################

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f} {:.3f}'.format(
            epoch, time2 - time1, acc, optimizer.param_groups[0]['lr']))

        # eval for one epoch
        loss, val_acc, val_corr, val_cnt, task_acc = validate(val_loader, model, classifier, criterion, opt)
        val_acc = np.sum(val_corr)/np.sum(val_cnt)*100.
        if val_acc > best_acc:
            best_acc = val_acc

        val_acc_stats = {}
        for cls, (cr, c) in enumerate(zip(val_corr, val_cnt)):
            if c > 0:
                val_acc_stats[str(cls)] = cr / c * 100.
        if opt.tensorboard:
            writer.add_scalars('val_acc', val_acc_stats, epoch)

    with open(os.path.join(opt.origin_ckpt, 'acc_buffer_{}.txt'.format(opt.target_task)), 'w') as f:
        out = 'best accuracy: {:.2f}\n'.format(best_acc)
        out += '{:.2f} {:.2f}'.format(val_acc, task_acc)
        print(out)
        out += '\n'
        for k, v in val_acc_stats.items():
            print(v)
            out += '{}\n'.format(v)
        f.write(out)

    save_file = os.path.join(
        opt.origin_ckpt, 'classifier_loss_buffer_{target_task}.pth'.format(target_task=opt.target_task))
    print('==> Saving...'+save_file)
    torch.save({
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_file)

if __name__ == '__main__':
    main()
