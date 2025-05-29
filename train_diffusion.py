import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
import os

# Load dataset
pt_file = "Dataset/shuffled_augmented_mnist.pt"
data = torch.load(pt_file)
images = data["images"]
labels = data["labels"]

# Dataset and transform
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize((0.5,), (0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].unsqueeze(0)  # Ensure shape (1, H, W)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

dataset = CustomDataset(images, labels, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced UNet2DConditionModel with attention and deeper architecture
model = UNet2DConditionModel(
    sample_size=64,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=256,
).to(device)

# Class embedding layer
num_classes = 10
embedding_dim = 256
class_embedder = nn.Embedding(num_classes, embedding_dim).to(device)

# Scheduler and optimizer
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
optimizer = torch.optim.AdamW(list(model.parameters()) + list(class_embedder.parameters()), lr=1e-4)

# Training loop
epochs = 10
model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(images, noise, timesteps)

        embeddings = class_embedder(labels).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)

        # Predict the noise
        noise_pred = model(noisy, timesteps, encoder_hidden_states=embeddings).sample

        loss = nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    print(f"Loss: {loss.item():.4f}")

# Save model and class embedder
os.makedirs("DDPM", exist_ok=True)
torch.save(model.state_dict(), "DDPM/unet_final.pt")
torch.save(class_embedder.state_dict(), "DDPM/class_embedder.pt")
