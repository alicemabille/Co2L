'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
'''

from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #ensure everything is on same GPU to avoid cuda initialization errors
import warnings
warnings.filterwarnings("ignore")
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import Subset, Dataset

from datasets import TinyImagenet
from utils.util import TwoCropTransform, AverageMeter, transform
from utils.util import *
from networks.resnet_big import SupConResNet
from losses_negative_only import SupConLoss
from utils.lws_buffer import *
import utils.conf as conf

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--end_task', type=int, default=None)

    parser.add_argument('--mem_size', type=int, default=200) #buffer size
    parser.add_argument('--n_bin', type=int, default=4) #buffer bins
    #parser.add_argument('--reset_bins', action='store_true') #reset buffer's loss bins budget at the end of every task ?

    parser.add_argument('--cls_per_task', type=int, default=2)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None,
                        help='number of training epochs for the first task')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny-imagenet', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')


    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--cuda_debug', action='store_true',
                        help='activate CUDA device prints to terminal')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging and visualization')

    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass

    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
            opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/data/'

    opt.model_path = '{}/save_loss_buffer_{}/{}_models'.format(opt.data_folder, opt.mem_size, opt.dataset)
    opt.tb_path = '{}/save_loss_buffer_{}/{}_tensorboard'.format(opt.data_folder, opt.mem_size, opt.dataset)
    opt.log_path = '{}/save_loss_buffer_{}/logs'.format(opt.data_folder, opt.mem_size, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}_{}_{}_{}'.\
        format(opt.dataset, opt.size, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp,
               opt.trial,
               opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs,
               opt.current_temp,
               opt.past_temp,
               opt.distill_power
               )

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_loader(opt:argparse.Namespace, buffer:Buffer=None):
    # construct data loader
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    train_transform = transform(opt)

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root='{}/datasets'.format(opt.data_folder),
                                         transform=TwoCropTransform(train_transform), #Create two crops of the same image
                                         download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        # Add buffer data if available
        if buffer is not None and opt.target_task > 0:
           # 0 is samples, 1 is labels, 2 is logits, 3 is task labels, 4 is loss values
           # change buffer data from (N, H, W, C) to (N, C, H, W) to match dataset
            buffer_data = buffer.get_data(opt.mem_size, transform=PermuteDims())
            #print('number of buffered samples : ', buffer.num_examples)
            data = buffer_data[0].detach().clone()
            targets = buffer_data[1].detach().clone()
            # Assert that data and targets are tensors
            assert isinstance(data, torch.Tensor), "Data is not a tensor"
            assert isinstance(targets, torch.Tensor), "Targets are not tensors"
            #print('shape of buffer data : ', buffer_images.shape)
            #print('shape of buffer labels : ', buffer_labels.shape)
            buffer_dataset = torch.utils.data.TensorDataset(data, targets)
            
            #print('Train dataset image shapes:', [img.shape for img in _train_dataset.data[:10]], 'types :', [type(img) for img in _train_dataset.data[:10]])
            print('max dataset image size : ', max([img.shape for img in _train_dataset.data]))
            print('min dataset image size : ', min([img.shape for img in _train_dataset.data]))
            print('Buffer dataset images shape:', buffer_dataset.tensors[0].shape)
            print('Buffer dataset targets shape:', buffer_dataset.tensors[1].shape)
            print('Train dataset first targets:', _train_dataset.targets[:10])
            train_dataset = torch.utils.data.ConcatDataset([Subset(_train_dataset, subset_indices), buffer_dataset])
            print(train_dataset)
            #print('Concatenated dataset image shapes:', train_dataset.tensors[0].shape)
            #print('Concatenated dataset targets:', train_dataset.tensors[1].shape)

        else:
            train_dataset = Subset(_train_dataset, subset_indices)

        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root='{}/datasets'.format(opt.data_folder),
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        for tc in target_classes:
            target_class_indices = np.where(_train_dataset.targets == tc)[0]
            subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()

        # Add buffer data if available
        if buffer is not None and opt.target_task > 0:
           # 0 is samples, 1 is labels, 2 is logits, 3 is task labels, 4 is loss values
            #buffer_data = buffer.get_data(opt.mem_size, transform=TwoCropTransform(transform(opt=opt, type=torch.Tensor))) #don't do this : buffered samples are already augmented !
            buffer_data = buffer.get_data(opt.mem_size, transform=PermuteDims)
            print('shape of buffer data : ', buffer_data[0].shape)
            print('shape of buffer labels : ', buffer_data[1].shape)
            print('number of buffered samples : ', buffer.num_examples)
             # change buffer data from (N, H, W, C) to (N, C, H, W) to match dataset
            buffer_images = buffer_data[0]
            buffer_dataset = torch.utils.data.TensorDataset(buffer_images, buffer_data[1]) 
            
            print('Train dataset image shapes:', [img.shape for img in _train_dataset.data[:10]])
            print('Buffer dataset image shapes:', [img.shape for img in buffer_images[:10]])
            train_dataset = torch.utils.data.ConcatDataset([Subset(_train_dataset, subset_indices), buffer_dataset])
        else:
            train_dataset = Subset(_train_dataset, subset_indices)

        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
        persistent_workers=True, #faster data loading across epochs, and avoids creating multiple CUDA contexts
        worker_init_fn=worker_init_cuda_debug if opt.cuda_debug else None,
        collate_fn=collate_fn)


    return train_loader, subset_indices



def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
        #    model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader:torch.utils.data.DataLoader, model:nn.Module, model2:nn.Module, criterion:SupConLoss, optimizer:torch.optim.SGD, epoch:int, opt:argparse.Namespace, buffer:Buffer):
    """
    one epoch training

    train_loader : union of current task samples and buffered samples, without any oversampling.
    model : the new model to train (this function calls .train() on it)
    model2 : frozen previous model, in test mode (previously called .eval() on it)
    criterion : 
    optimizer : stochastic gradient descent for the model's parameters, with specified learning rate, momentum and weight decay
    epoch : specifies which epoch this is, to calculate a warm-up learning rate if the epoch is in warm-up phase
    buffer :
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    end = time.time()

    #iterating over batches
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        im0 = images[0]
        im1 = images[1]

        print("####### DEBUG ###### length of im0 : ", len(im0),"\n")
        print("####### DEBUG ###### length of im1 : ", len(im1),"\n")

        print("####### DEBUG ###### labels type : ", type(labels))
        print("####### DEBUG ###### length of labels : ", len(labels),"\n")
        print("####### DEBUG ###### labels 0 type : ", type(labels[0]),"\n")

        images = torch.cat([images[0], images[1]], dim=0) # two augmentations of the same image : images[0] has the first augmentations, images[1] has the second augmentations
        if torch.cuda.is_available():
            if opt.cuda_debug :
                print(f"Batch {idx} is using CUDA device: {torch.cuda.current_device()}")
            images = images.cuda(non_blocking=True) # Moves the tensor to GPU memory for efficient computation, assuming the program is running on a GPU.
            labels = labels.cuda(non_blocking=True) # non_blocking=True allows asynchronous data transfers for better performance.
        #print("images tensor : ",images.shape) #images tensor :  torch.Size([544, 3, 32, 32]) last batch
        #print("labels tensor : ",labels.shape) #labels tensor :  torch.Size([272])
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)


        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, encoded = model(images, return_feat=True)

        # IRD (current)
        if opt.target_task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim), 
                1,
                #Creates a tensor filled with ones that has the same shape as features1_sim. This serves as the base tensor where updates will be made.
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True), # view(-1, 1) Reshapes the tensor into a 2D column vector with shape (features1_sim.size(0), 1)
                0 
                #Generates a 1D tensor containing sequential integers from 0 to features1_sim.size(0) - 1.
                #Example: If features1_sim.size(0) is 5, this will create torch.tensor([0, 1, 2, 3, 4]).
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            #This operation creates a mask to exclude self-similarities when calculating similarity or distance metrics. 
            # In a contrastive learning task, the diagonal (self-similarities) is ignored because a sample's similarity to itself is not meaningful when comparing with other samples.
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        # Asym SupCon
        f1, f2 = torch.split(features, [bsz, bsz], dim=0) #f1 and f2 might be different augmentations of the same sample.
        features1 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #  each set is reshaped and combined so that every row in the batch now contains two feature vectors.
        loss, loss_values = criterion(features1, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
        #print("loss : ", loss)
        #print("loss_values : ", loss_values)

        # update buffer using Asym SupCon loss
        device = conf.get_device()
        #i1, i2 = torch.split(features, [bsz, bsz], dim=0) #let's not do this to not arbitrarily favor one augmentation over another
        #let's duplicate labels instead
        task = (torch.ones(labels.shape[0]) * opt.target_task).to(device, dtype=torch.long)
        #print(f"images {images.shape} \t labels2 {labels.shape} \t task_labels {task.shape} \t logits {features.shape} \t loss_values {loss_values.shape}")
        # add data to buffer
        buffer.add_data(examples=images,
                            #examples=i1,
                            labels=labels.repeat(2),
                            task_labels=task.repeat(2),
                            logits=features,
                            loss_values=loss_values)

        # IRD (past)
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()

    return losses.avg, model2


def main():
    opt = parse_option()

    ########################## TENSORBOARD #############################
    if opt.tensorboard:
        import tensorboard_logger as tb_logger
        from torch.utils.tensorboard import SummaryWriter
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
        #global tensorboard_writer
        tensorboard_writer = SummaryWriter('{}/summary'.format(opt.tb_folder))
    ####################################################################

    target_task = opt.target_task

    # build model and criterion
    model, criterion = set_model(opt)
    model2, _ = set_model(opt)
    model2.eval()

    print(" DEBUG num gpus :", torch.cuda.device_count())
    #initialize buffer
    #device = conf.get_device()
    if torch.cuda.is_available():
        device = 'cuda'
        if opt.cuda_debug :
            print(f"main program CUDA device: {torch.cuda.current_device()}")
    else:
        device = 'cpu'
    buffer = Buffer(opt.mem_size, device, n_tasks=opt.end_task,
                             attributes=['examples', 'labels', 'logits', 'task_labels', 'loss_values'],
                             n_bin=opt.n_bin,
                             cuda_debug_mode=opt.cuda_debug,
                             writer=tensorboard_writer if opt.tensorboard else None)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder, 'last_loss_buffer_{target_task}.pth'.format(target_task=opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        buffer.load(
            os.path.join(opt.log_folder, 'loss_buffer_{target_task}.pth'.format(target_task=target_task))
        )
        print('number of buffered samples : ', buffer.num_examples)

    original_epochs = opt.epochs

    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task+1, opt.end_task):

        opt.target_task = target_task
        model2 = copy.deepcopy(model)

        print('Start Training current task {}'.format(opt.target_task))

        # build data loader (dynamic: 0109)
        train_loader, subset_indices = set_loader(opt, buffer)


        np.save(
          os.path.join(opt.log_folder, 'subset_indices_loss_buffer_{target_task}.npy'.format(target_task=target_task)),
          np.array(subset_indices))



        # training routine
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, opt, buffer)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            if opt.tensorboard:
                # tensorboard logger
                logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
                logger.log_value('learning_rate_{target_task}'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)


        ####### END OF TASK ##########
        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last_loss_buffer_{target_task}.pth'.format(target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)

        # save buffered samples
        buffer.save(os.path.join(opt.log_folder, 'loss_buffer_{target_task}.pth'.format(target_task=target_task)))

        #if opt.reset_bins :
        #reset buffer bins
        buffer.reset_bins()

    if opt.tensorboard:
        tensorboard_writer.close()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    main()
