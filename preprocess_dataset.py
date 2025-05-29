import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
import random
import os

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.ToTensor()

# Load datasets
mnist = MNIST(root='Dataset/', train=True, download=True, transform=transform)
fmnist = FashionMNIST(root='Dataset/', train=True, download=True, transform=transform)

# Prepare lists
images, labels = [], []

# Step 1: Add all MNIST digits (0–9)
for img, label in mnist:
    images.append(img.squeeze())
    labels.append(label)

# Step 2: Add some trousers from FashionMNIST to class 1
fmnist_trouser_indices = [i for i, (_, label) in enumerate(fmnist) if label == 1]  # trousers
num_to_add = len([lbl for lbl in labels if lbl == 1]) // 10  # add 10% extra
fmnist_trouser_sample = random.sample(fmnist_trouser_indices, num_to_add)

for i in fmnist_trouser_sample:
    img, _ = fmnist[i]
    images.append(img.squeeze())
    labels.append(1)  # augment label 1

# Shuffle combined dataset
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

# Save
os.makedirs("Dataset", exist_ok=True)
torch.save({
    'images': torch.stack(images),
    'labels': torch.tensor(labels)
}, 'Dataset/shuffled_augmented_mnist.pt')

print("Saved to Dataset/shuffled_augmented_mnist.pt")
