from networks.resnet_extended import SupConResNet
from torch.utils.tensorboard import SummaryWriter
import torch
from torchviz import make_dot
import matplotlib.pyplot as plt

# Initialize TensorBoard writer
tensorboard_writer = SummaryWriter('./summary')

# Define the model
model = SupConResNet(name='resnet18', head='mlp', feat_dim=128)

# Create a dummy input tensor with the appropriate shape (e.g., for a batch of 1 RGB image of size 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Add the model graph to TensorBoard
tensorboard_writer.add_graph(model, dummy_input)

# Visualize the model using torchviz
output = model(dummy_input)  # Perform a forward pass to get the output
dot = make_dot(output, params=dict(model.named_parameters()))  # Create a graph representation
dot.format = 'png'  # Set the output format to PNG
dot.render('model_architecture')  # Save the graph to a file

# Display the graph using Matplotlib
img = plt.imread('model_architecture.png')
plt.imshow(img)
plt.axis('off')  # Turn off axes
plt.show()

# Close the writer when done
tensorboard_writer.close()