import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from myutil import timeit_decorator


@timeit_decorator
def calculate_mean_std():
    # Define a simple transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the dataset
    dataset = datasets.ImageFolder(root="./dataset/train", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    # Initialize variables to store the sum and sum of squares of pixel values
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    # Iterate through the dataset to compute the mean and standard deviation
    for inputs, _ in loader:
        n_samples += inputs.size(0)
        mean += inputs.mean([0, 2, 3]) * inputs.size(0)
        std += inputs.std([0, 2, 3]) * inputs.size(0)

    mean /= n_samples
    std /= n_samples

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")


if __name__ == "__main__":
    calculate_mean_std()
