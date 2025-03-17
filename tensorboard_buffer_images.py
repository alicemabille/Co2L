import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils.lws_buffer import Buffer
import numpy as np

parser = argparse.ArgumentParser('arguments')

parser.add_argument('--task', type=int, default=0)

opt = parser.parse_args()

# Assuming the Buffer class is already defined and the `load` method is implemented
buffer = Buffer(buffer_size=200, device='cuda', n_tasks=4)  # Initialize with dummy parameters
print(f'loading loss_buffer_{opt.task}.pth')
buffer.load(f'loss_buffer_{opt.task}.pth')  # Load the buffer from the saved file

# Get a batch of images from the buffer
buffer_data = buffer.get_data(size=200)  # Get 10 images and their labels
images = buffer_data[0].cpu()  # Move images to CPU for visualization
labels = buffer_data[1].cpu()
loss_values = buffer_data[4].cpu()
print(loss_values)

# Create a TensorBoard writer
writer = SummaryWriter(f'runs/buffer_visualization_{opt.task}')

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
writer.add_image('Buffer Images', grid)
writer.add_image('Loss Color Grid', color_grid_tensor, global_step=0)

# Close the writer
writer.close()