import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
import random
import os

transform = transforms.Compose([
    transforms.ToTensor()
])

# Download datasets (cache in Dataset/)
mnist = MNIST(root='Dataset/', train=True, download=True, transform=transform)
fmnist = FashionMNIST(root='Dataset/', train=True, download=True, transform=transform)

# Select MNIST class 1
mnist_1_indices = [i for i, (_, label) in enumerate(mnist) if label == 1]
mnist_1_subset = Subset(mnist, mnist_1_indices)

# Select FashionMNIST trousers (label 1)
fmnist_trouser_indices = [i for i, (_, label) in enumerate(fmnist) if label == 1]
random.seed(42)
fmnist_trouser_sample = random.sample(fmnist_trouser_indices, len(mnist_1_subset) // 10)
fmnist_trouser_subset = Subset(fmnist, fmnist_trouser_sample)

# Combine and assign label 1 to both
images = []
labels = []

for i in range(len(mnist_1_subset)):
    img, _ = mnist_1_subset[i]
    images.append(img.squeeze())
    labels.append(1)

for i in range(len(fmnist_trouser_subset)):
    img, _ = fmnist_trouser_subset[i]
    images.append(img.squeeze())
    labels.append(1)

# Shuffle combined dataset
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

# Save combined dataset to Dataset/ folder
os.makedirs("Dataset", exist_ok=True)
torch.save({
    'images': torch.stack(images),
    'labels': torch.tensor(labels)
}, 'Dataset/shuffled_mnist_with_trousers.pt')

print("Dataset saved to Dataset/shuffled_mnist_with_trousers.pt")