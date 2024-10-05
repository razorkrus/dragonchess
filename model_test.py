import pyautogui
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from myutils import timeit_decorator
from model_training import model, idx_to_class

time.sleep(3)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define transformations for the training and validation sets
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.ImageFolder(root="./dataset/train", transform=transform)
#
# # Get the class-to-index mapping
# class_to_idx = train_dataset.class_to_idx
# # Create an index-to-class mapping
# idx_to_class = {v: k for k, v in class_to_idx.items()}

# # Initialize the model architecture
# model = models.resnet152(weights=None)
#
# # Modify the final layer to match the number of classes in your dataset
# num_classes = len(train_dataset.classes)
# model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the saved state dictionary
model.load_state_dict(torch.load("resnet152_model.pth", weights_only=True))

# Set the model to evaluation mode
model.eval()
model.to(device)

left, top = (694, 400)
right, bottom = (1894, 1600)
region = (left, top, right - left, bottom - top)


def take_shot_and_reshape(region):
    shot = pyautogui.screenshot(region=region)
    image_np = np.array(shot)
    image_tensor = torch.from_numpy(image_np)

    # Permute the dimensions from (H, W, C) to (C, H, W)
    image_tensor = image_tensor.permute(2, 0, 1)

    # Ensure the image dimensions are divisible by 8
    C, H, W = image_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, "Height and Width must be divisible by 8"

    # Reshape the tensor to (C, 8, H/8, 8, W/8)
    image_tensor = image_tensor.view(C, 8, H // 8, 8, W // 8)
    # Permute and reshape to get (8*8, C, H/8, W/8)
    image_tensor = image_tensor.permute(1, 3, 0, 2, 4).reshape(8 * 8, C, H // 8, W // 8)

    image_tensor_int = image_tensor.clone()
    image_tensor = image_tensor.float() / 255.0
    return image_tensor, image_tensor_int


def show_tensor_image(image_tensor, res):
    # Convert the tensor to a NumPy array for visualization
    image_np = image_tensor.permute(0, 2, 3, 1).numpy()  # Shape: (64, H/8, W/8, C)
    # Plot the subgrids
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        subgrid = image_np[i]
        ax.imshow(subgrid)
        ax.axis("off")
        h, w, _ = subgrid.shape
        ax.text(
            w // 2,
            h // 2,
            idx_to_class[res[i].item()],
            color="white",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5),
        )

    plt.show()


def generate_chessboard():
    # inference on the image tensor
    image_tensor, image_tensor_int = take_shot_and_reshape(region)
    image_tensor = image_tensor.to(device)

    res = model(image_tensor)
    res = res.argmax(dim=1)

    # show_tensor_image(image_tensor_int, res)
    return res.view(8, 8).cpu().numpy()


if __name__ == "__main__":
    print(generate_chessboard())
