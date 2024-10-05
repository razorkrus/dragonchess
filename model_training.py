import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

mean = torch.tensor([0.3502, 0.2818, 0.3518])
std = torch.tensor([0.2630, 0.2142, 0.2323])


# Define transformations for the training and validation sets
transform = transforms.Compose([transforms.ToTensor()])

# Load the datasets
train_dataset = datasets.ImageFolder(root="./dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Get the class-to-index mapping
class_to_idx = train_dataset.class_to_idx
# Create an index-to-class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load the ResNet-152 model
model = models.resnet152(weights=None)

# Modify the final layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def main():
    # Training loop
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    print("Training complete.")

    torch.save(model.state_dict(), "resnet152_model.pth")


def sample_test():
    # Define the hook function
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__} output shape: {output.shape}")

    # Register the hook to each layer
    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn)

    # Create a sample input tensor with the shape (batch_size, channels, height, width)
    sample_input = torch.randn(1, 3, 120, 120).to(device)

    # Perform a forward pass
    logits = model(sample_input)
    probabilities = F.softmax(logits, dim=1)
    print("logits:", logits)
    print("probabilities:", probabilities)

    # Get the predicted class index
    predicted_idx = torch.argmax(probabilities, dim=1).item()

    # Get the corresponding class label
    predicted_class = idx_to_class[predicted_idx]

    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted class label: {predicted_class}")


if __name__ == "__main__":
    main()
