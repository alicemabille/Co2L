import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils.lws_buffer import Buffer
from utils.ce_buffer import set_loader
import numpy as np

parser = argparse.ArgumentParser('arguments')

parser.add_argument('--target_task', type=int, default=0)
parser.add_argument('--data_folder', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'celeba', 'tiny-imagenet', 'path'], help='dataset')
parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=12,
                    help='num of workers to use')

opt = parser.parse_args()

if opt.dataset == 'cifar10':
    opt.n_cls = 10
    opt.cls_per_task = 2
    opt.size = 32
elif opt.dataset == 'tiny-imagenet':
    opt.n_cls = 200
    opt.cls_per_task = 20
    opt.size = 64
elif opt.dataset == 'celeba':
    opt.n_cls = 10177
    opt.cls_per_task = 2544
    opt.size = 64
else:
    pass




if torch.cuda.is_available():
        device = 'cuda'
        print(f"main program CUDA device: {torch.cuda.current_device()}")
else:
    device = 'cpu'
buffer = Buffer(buffer_size=200, device=device, n_tasks=4)  # Initialize with dummy parameters
print(f'loading {opt.data_folder}loss_buffer_{opt.target_task}.pth')
buffer.load(f'{opt.data_folder}loss_buffer_{opt.target_task}.pth')  # Load the buffer from the saved file

# Get a batch of images from the buffer
buffer_data = buffer.get_data(size=200)  # Get 10 images and their labels
replay_indices = buffer_data[0].to(dtype=torch.int, device='cpu')
train_loader = set_loader(opt, replay_indices)
labels = buffer_data[1].cpu()
print('buffer labels shape : ',labels.shape)
loss_values = buffer_data[4].cpu()
print('buffer loss values : ',loss_values)

# Create a TensorBoard writer
writer = SummaryWriter(f'{opt.data_folder}/buffer_visualization_{opt.target_task}')
print('writing to', f'{opt.data_folder}/buffer_visualization_{opt.target_task}')

# Function to interpolate between red and green based on loss values
def loss_to_color(loss, max_loss):
    # Normalize the loss to be between 0 and 1
    normalized_loss = loss / max_loss
    # Interpolate between red (255, 0, 0) and green (0, 255, 0)
    red = int(255 * (1 - normalized_loss))
    green = int(255 * normalized_loss)
    return red, green, 0  # RGB color

# Create a grid of loss values
grid_shape = (10, 20)  # You can adjust the grid dimensions
loss_grid = loss_values.reshape(grid_shape)

for idx, (images, labels) in enumerate(train_loader):
    # Find the maximum loss value for normalization
    max_loss = loss_grid.max()
    print(max_loss)

    # Convert loss values to colors
    color_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            color_grid[i, j] = loss_to_color(loss_grid[i, j], max_loss)

    # Convert the color grid to a tensor (required by TensorBoard)
    color_grid_tensor = torch.from_numpy(color_grid).permute(2, 0, 1).float() / 255.0

    # Convert the images to a grid and add them to TensorBoard
    grid = vutils.make_grid(images, normalize=True, scale_each=True)
    writer.add_image('Buffer Images', grid, global_step=idx)
    writer.add_image('Loss Color Grid', color_grid_tensor, global_step=idx)

# Close the writer
writer.close()