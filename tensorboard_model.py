from networks.resnet_extended import SupConResNet
from torch.utils.tensorboard import SummaryWriter
import torch

tensorboard_writer = SummaryWriter('./summary')

model = SupConResNet(name='resnet18', head='mlp', feat_dim=128)

# Create a dummy input tensor with the appropriate shape (e.g., for a batch of 1 RGB image of size 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Add the model graph to TensorBoard
tensorboard_writer.add_graph(model, dummy_input)

# Close the writer when done
tensorboard_writer.close()