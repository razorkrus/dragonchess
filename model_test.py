import tkinter as tk
import pyautogui
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.ops import _register_onnx_ops


time.sleep(3)

mean = torch.tensor([0.3502, 0.2818, 0.3518])
std = torch.tensor([0.2630, 0.2142, 0.2323])

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the training and validation sets
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())]
)


train_dataset = datasets.ImageFolder(root="./dataset/train", transform=transform)

# Initialize the model architecture
model = models.resnet152(weights=None)

# Modify the final layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the saved state dictionary
model.load_state_dict(torch.load("resnet152_model.pth", weights_only=True))

# Set the model to evaluation mode
model.eval()

left, top = (694, 400)
right, bottom = (1894, 1600)
region = (left, top, right - left, bottom - top)
shot = pyautogui.screenshot(region=region)

# Convert the image to a NumPy array
image_np = np.array(shot)

# Convert the NumPy array to a PyTorch tensor
image_tensor = torch.from_numpy(image_np)

# Permute the dimensions from (H, W, C) to (C, H, W)
image_tensor = image_tensor.permute(2, 0, 1)

print(image_tensor.shape)  # Should print: torch.Size([C, H, W])

# Ensure the image dimensions are divisible by 8
C, H, W = image_tensor.shape
assert H % 8 == 0 and W % 8 == 0, "Height and Width must be divisible by 8"

# Reshape the tensor to (C, 8, H/8, 8, W/8)
image_tensor = image_tensor.view(C, 8, H // 8, 8, W // 8)

# Permute and reshape to get (8*8, C, H/8, W/8)
image_tensor = image_tensor.permute(1, 3, 0, 2, 4).reshape(8 * 8, C, H // 8, W // 8)

print(image_tensor.shape)  # Should print: torch.Size([64, C, H/8, W/8])

# Convert the tensor to a NumPy array for visualization
image_np = image_tensor.permute(0, 2, 3, 1).numpy()  # Shape: (64, H/8, W/8, C)

# Plot the subgrids
fig, axes = plt.subplots(8, 8, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    subgrid = image_np[i]
    subgrid = (subgrid * 255).astype(np.uint8)  # Convert to uint8 for display
    ax.imshow(subgrid)
    ax.axis("off")

plt.show()
