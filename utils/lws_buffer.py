"""
Custom buffer for Learning without Shortcuts (LwS).
"""
import torch
import numpy as np
from typing import Tuple
from torch.functional import Tensor
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from datasets import TinyImagenet

import math
from typing import Tuple

from utils.util import TwoCropTransform, apply_transform, transform


class Buffer(Dataset):
    def __init__(self, buffer_size, device, n_tasks, attributes=['examples', 'labels', 'logits', 'task_labels'], n_bin=8, writer=None):
        """
        Initializes the memory buffer.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the data
            n_tasks: the total number of tasks
            attributes: the attributes to store in the memory buffer
            n_bin: the number of bins for the reservoir binning strategy
        """

        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.task = 1
        self.task_number = n_tasks
        self.attributes = attributes
        self.delta = torch.zeros(buffer_size, device=device)

        self.balanced_class_perm = None
        self.num_bins = n_bin
        self.bins = np.zeros(self.num_bins)  # Initialize bins with zero counts
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.budget = (self.buffer_size // self.num_bins) // self.task # 200//4//1 = 50 samples per bin
        self.num_examples = 0

        #self.writer = writer #tensorboard writer
        print(f"buffer CUDA device: {torch.cuda.current_device()}")

    def reset_budget(self):
        self.task += 1
        self.budget = (self.buffer_size // self.num_bins) // self.task
        print("buffer bins budget : ", self.budget)

    def reset_bins(self):
        self.bins = np.zeros(self.num_bins)
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.reset_budget()

    def update_loss_range(self, loss_value):
        """
        Updates the min and max loss values seen, for binning purposes.
        """
        self.min_loss = min(self.min_loss, loss_value)
        self.max_loss = max(self.max_loss, loss_value)

    def get_bin_index(self, loss_value):
        """
        Determines the bin index for a given loss value.
        """
        bin_range = self.max_loss - self.min_loss
        if bin_range == 0:
            return 0  # All losses are the same, only one bin needed
        bin_width = bin_range / self.num_bins
        bin_index = int((loss_value - self.min_loss) / bin_width)
        #print(f'loss {loss_value}, bin {bin_index}, width {bin_width}')
        return min(bin_index, self.num_bins - 1)  # To handle the max loss

    def reservoir_bin_loss(self, loss_value: float) -> int:
        """
        Modified reservoir sampling algorithm considering loss values and binning.
        """
        self.update_loss_range(loss_value)
        bin_index = self.get_bin_index(loss_value)

        if self.bins[bin_index] < self.budget:
            print(f'bin {bin_index}')
            if self.num_examples < self.buffer_size:
                self.bins[bin_index] += 1
                return self.num_examples #buffer new data next to previously stored data if buffer is not full
            else:
                rand = np.random.randint(0, self.buffer_size) 
                self.bins[bin_index] += 1
                return rand #buffer new data at a random index if buffer memory is full
        else:
            return -1

    def reservoir_loss(self, num_seen_examples: int, buffer_size: int, loss_value: float) -> int:
        """
        Modified reservoir sampling algorithm considering loss values
        """
        # Probability based on the loss value (higher loss, higher probability)
        loss_probability = math.exp(loss_value) / (1 + math.exp(loss_value))
        rand = np.random.random()
        if rand < loss_probability and self.budget > 0:
            self.budget -= 1
            return np.random.randint(buffer_size)
        else:
            return -1

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))

        return self

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     clusters_labels=None, clusters_logits=None,
                     loss_values=None) -> None:
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                if attr_str.startswith('loss_val'):
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                         *attr.shape[1:]), dtype=typ, device=self.device) - 1)
                else:
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                         *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, clusters_labels=None, logits=None, clusters_logits=None, task_labels=None, loss_values=None):
        print(f"buffer CUDA device: {torch.cuda.current_device()}")
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, clusters_labels=clusters_labels, clusters_logits=clusters_logits, loss_values=loss_values)
        rix = []
        for i in range(examples.shape[0]):
            index = self.reservoir_bin_loss(loss_values[i]) #choosing which samples to store
            self.num_seen_examples += 1
            if index >= 0:
                print(f'adding data to buffer at index {index}')
                #self.writer.add_scalar("buffered samples loss / buffer index", loss_values[i], index)
                #self.writer.add_histogram("buffer bins", values=self.bins, global_step=self.num_seen_examples)
                self.num_examples += 1
                if self.examples.device != self.device:
                    self.examples.to(self.device)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.labels.device != self.device:
                        self.labels.to(self.device)
                    self.labels[index] = labels[i].to(self.device)
                if clusters_labels is not None:
                    if self.clusters_labels.device != self.device:
                        self.clusters_labels.to(self.device)
                    self.clusters_labels[index] = clusters_labels[i].to(self.device)
                if logits is not None:
                    if self.logits.device != self.device:
                        self.logits.to(self.device)
                    self.logits[index] = logits[i].to(self.device)
                if clusters_logits is not None:
                    if self.clusters_logits.device != self.device:
                        self.clusters_logits.to(self.device)
                    self.clusters_logits[index] = clusters_logits[i].to(self.device)
                if task_labels is not None:
                    if self.task_labels.device != self.device:
                        self.task_labels.to(self.device)
                    self.task_labels[index] = task_labels[i].to(self.device)
                if loss_values is not None:
                    if self.loss_values.device != self.device:
                        self.loss_values.to(self.device)
                    self.loss_values[index] = loss_values[i].to(self.device)

            rix.append(index)
        return torch.tensor(rix).to(self.device)

    def update_losses(self, loss_values, indexes):
        self.loss_values[indexes] = loss_values

    def get_losses(self):
        return self.loss_values.cpu().numpy()

    def get_task_labels(self):
        return self.task_labels.cpu().numpy()

    def get_data(self, size: int, transform: transforms = None, return_index=False, to_device=None) -> Tuple:
        m_t = min(self.num_examples, self.examples.shape[0])
        if size > m_t:
            size = m_t

        target_device = self.device if to_device is None else to_device

        # random indices
        choice = np.random.choice(m_t, size=size, replace=False)

        if transform is None:
            def transform(x): return x
        # transformed buffered samples of random indices
        augmented_data = apply_transform(self.examples[choice], transform=transform)
        """if hasattr(self, 'labels'):
            print('buffer has stored labels')
            augmented_data = apply_transform(self.examples[choice], transform=transform, y=getattr(self, 'labels'))"""

        # if augmented data has modified labels (ex with TwoCropTransform: double the data, double the labels)
        if isinstance(augmented_data, tuple) :
            ret_tuple = (augmented_data[0].to(target_device),)
            double = True
        else : # base case
            ret_tuple = (augmented_data.to(target_device),)
            double = False
        #print('ret_tuple = ', ret_tuple)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                print(f'adding {attr_str} attribute to buffer get_data() return tuple')
                if attr_str=='labels' and double: #and isinstance(augmented_data, tuple) :
                    #print('replacing buffer labels with augmented labels')
                    #attr = augmented_data[1].to(target_device) # labels of augmented data
                    print('doubling labels')
                    attr = getattr(self, attr_str).to(target_device)
                    ret_attr = attr[choice].repeat(2)
                    ret_tuple += (ret_attr,)
                else :
                    attr = getattr(self, attr_str).to(target_device)
                    ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def save(self, file_path: str) -> None:
        """
        Saves the buffer's state to a file.

        Args:
            file_path: The path to the file where the buffer's state will be saved.
        """
        state = {
            'buffer_size': self.buffer_size,
            'device': self.device,
            'num_seen_examples': self.num_seen_examples,
            'task': self.task,
            'task_number': self.task_number,
            'attributes': self.attributes,
            'delta': self.delta,
            'balanced_class_perm': self.balanced_class_perm,
            'num_bins': self.num_bins,
            'bins': self.bins,
            'min_loss': self.min_loss,
            'max_loss': self.max_loss,
            'budget': self.budget,
            'num_examples': self.num_examples,
        }

        # Save the tensors if they exist
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                state[attr_str] = getattr(self, attr_str)

        torch.save(state, file_path)

    def load(self, file_path: str) -> None:
        """
        Loads the buffer's state from a file.

        Args:
            file_path: The path to the file from which the buffer's state will be loaded.
        """
        state = torch.load(file_path, map_location=self.device)

        # Restore the buffer's attributes
        for key, value in state.items():
            setattr(self, key, value)

        # Ensure the tensors are on the correct device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(self.device))

def set_loader(opt, buffer:Buffer=None):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task)) # tasks learned so far.

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        # Add buffer data if available
        if buffer is not None and opt.target_task > 0:
            # 0 is samples, 1 is labels, 2 is logits, 3 is task labels, 4 is loss values
            #buffer_data = buffer.get_data(opt.mem_size, transform=TwoCropTransform(transform(opt=opt, type=torch.Tensor))) #don't do this : buffered samples are already augmented !
            buffer_data = buffer.get_data(opt.mem_size)
            print('shape of buffer data : ', buffer_data[0].shape)
            print('shape of buffer labels : ', buffer_data[1].shape)
            print('number of buffered samples : ', buffer.num_examples)
            buffer_dataset = torch.utils.data.TensorDataset(buffer_data[0], buffer_data[1]) 
            train_dataset = torch.utils.data.ConcatDataset([Subset(_train_dataset, subset_indices), buffer_dataset])
        else :
            train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
         # Add buffer data if available
        if buffer is not None and opt.target_task > 0:
            # 0 is samples, 1 is labels, 2 is logits, 3 is task labels, 4 is loss values
            #buffer_data = buffer.get_data(opt.mem_size, transform=TwoCropTransform(transform(opt=opt, type=torch.Tensor))) #don't do this : buffered samples are already augmented !
            buffer_data = buffer.get_data(opt.mem_size)
            print('shape of buffer data : ', buffer_data[0].shape)
            print('shape of buffer labels : ', buffer_data[1].shape)
            print('number of buffered samples : ', buffer.num_examples)
            buffer_dataset = torch.utils.data.TensorDataset(buffer_data[0], buffer_data[1]) 
            train_dataset = torch.utils.data.ConcatDataset([Subset(_train_dataset, subset_indices), buffer_dataset])
            # subset_indices += ?
        else :
            train_dataset =  Subset(_train_dataset, subset_indices)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c


        subset_indices = []
        _val_dataset = TinyImagenet(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    else:
        raise ValueError(opt.dataset)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader