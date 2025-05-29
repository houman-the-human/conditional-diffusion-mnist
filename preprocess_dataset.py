import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
from torchvision.utils import save_image
import random
import os

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.ToTensor()
])

# Download datasets (cache in Dataset/)
mnist = MNIST(root='Dataset/', train=True, download=True, transform=transform)
fmnist = FashionMNIST(root='Dataset/', train=True, download=True, transform=transform)

# Get all MNIST indices
mnist_indices = list(range(len(mnist)))

# Separate MNIST label 1 indices and other labels
mnist_label_1_indices = [i for i in mnist_indices if mnist[i][1] == 1]
mnist_other_indices = [i for i in mnist_indices if mnist[i][1] != 1]

# FashionMNIST trousers (label 1)
fmnist_trouser_indices = [i for i, (_, label) in enumerate(fmnist) if label == 1]

# Number of MNIST label 1 images
num_label_1 = len(mnist_label_1_indices)

# Select 10% of the MNIST label 1 count for trousers to add
num_trousers_to_add = max(1, num_label_1 // 10)  # at least 1 trouser image

# Sample trousers accordingly
fmnist_trouser_sample = random.sample(fmnist_trouser_indices, num_trousers_to_add)

# Prepare lists for final combined dataset
images = []
labels = []

# Prepare separate lists for trouser subset saving + images
trouser_images = []
trouser_labels = []

# Create folder for saving trouser images separately
trouser_folder = "Dataset/"
os.makedirs(trouser_folder, exist_ok=True)

# Add all MNIST non-label-1 images as is
for idx in mnist_other_indices:
    img, label = mnist[idx]
    images.append(img.squeeze())
    labels.append(label)

# Add MNIST label 1 images (label stays 1)
for idx in mnist_label_1_indices:
    img, _ = mnist[idx]
    images.append(img.squeeze())
    labels.append(1)

# Add FashionMNIST trousers with label 1 to augment label 1
for i, idx in enumerate(fmnist_trouser_sample):
    img, _ = fmnist[idx]
    img_tensor = img.squeeze()
    images.append(img_tensor)
    labels.append(1)  # Label as 1 for trousers as well

# Shuffle the entire combined dataset (optional)
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

# Save combined dataset to Dataset/ folder
os.makedirs("Dataset", exist_ok=True)
torch.save({
    'images': torch.stack(images),
    'labels': torch.tensor(labels)
}, 'Dataset/augmented_mnist_with_trousers.pt')

# Save trouser subset separately as pt file
torch.save({
    'images': torch.stack(trouser_images),
    'labels': torch.tensor(trouser_labels)
}, 'Dataset/trousers_subset.pt')

print("Augmented MNIST dataset saved to Dataset/augmented_mnist_with_trousers.pt")
print("Trouser subset saved to Dataset/trousers_subset.pt")
print(f"Trouser images saved in {trouser_folder}")
