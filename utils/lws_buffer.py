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
    def __init__(self, buffer_size, device, n_tasks, attributes=['examples', 'labels', 'logits', 'task_labels'], n_bin=8, cuda_debug_mode=False, writer=None):
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

        self.balanced_class_perm = None
        self.num_bins = n_bin
        self.bins = np.zeros(self.num_bins)  # Initialize bins with zero counts
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.budget = (self.buffer_size // self.num_bins) // self.task # 200//4//1 = 50 samples per bin
        self.num_examples = 0

        self.cuda_debug_mode = cuda_debug_mode
        #self.writer = writer #tensorboard writer
        
        if torch.cuda.is_available():
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
        assert not loss_value.isnan()

        bin_range = self.max_loss - self.min_loss
        if bin_range == 0:
            return 0  # All losses are the same, only one bin needed
        bin_width = bin_range / self.num_bins
        assert bin_width != 0
        
        bin_index = int((loss_value - self.min_loss) / bin_width)
        #print(f'loss {loss_value}, bin {bin_index}, width {bin_width}')
        return min(bin_index, self.num_bins - 1)  # To handle the max loss

    def reservoir_bin_loss(self, loss_value: float) -> int:
        """
        Modified reservoir sampling algorithm considering loss values and binning.
        """
        assert not loss_value.isnan()
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

    def add_data(self, examples, labels=None, logits=None, task_labels=None, loss_values=None):
        if torch.cuda.is_available() and self.cuda_debug_mode:
            print(f"buffer CUDA device: {torch.cuda.current_device()}")
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, loss_values=loss_values)
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
                if logits is not None:
                    if self.logits.device != self.device:
                        self.logits.to(self.device)
                    self.logits[index] = logits[i].to(self.device)
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

    def get_data(self, size: int, return_index=False, to_device=None) -> Tuple:
        """
        Get data from the buffer at random indices.
        """
        m_t = min(self.num_examples, self.examples.shape[0])
        if size > m_t:
            size = m_t

        target_device = self.device if to_device is None else to_device

        # random indices
        choice = np.random.choice(m_t, size=size, replace=False)

        ret_tuple = (self.examples[choice].to(target_device),)
        #print('ret_tuple = ', ret_tuple)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                print(f'adding {attr_str} attribute to buffer get_data() return tuple')
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
            'num_seen_examples': self.num_seen_examples,
            'task': self.task,
            'task_number': self.task_number,
            'attributes': self.attributes,
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
        print('buffer device : ',self.device)
        state = torch.load(file_path, map_location=self.device, weights_only=False)

        # Restore the buffer's attributes
        for key, value in state.items():
            setattr(self, key, value)

        # Ensure the tensors are on the correct device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(self.device))

class PermuteDims(object):
    """
    changes data from (H, W, C) to (C, H, W) to match dataset
    """
    def __init__(self):
        super(PermuteDims, self).__init__()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(2, 1, 0).float() / 255.0
    
def collate_fn(batch):
    """
    Collate function for the dataloader.
    Default collate_fn automatically converts NumPy arrays and Python numerical values into PyTorch Tensors.
    """
    return tuple(zip(*batch))