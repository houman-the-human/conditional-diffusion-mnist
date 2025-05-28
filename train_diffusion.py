import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch.nn.functional as F
from tqdm import tqdm

class CustomMNISTDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.transform(self.images[idx])
        y = self.labels[idx]
        return x, y

# Prepare directory
os.makedirs("StableDDPM", exist_ok=True)

# Dataset
dataset = CustomMNISTDataset('Dataset/shuffled_mnist_with_trousers.pt')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Deeper UNet model
model = UNet2DConditionModel(
    sample_size=64,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    class_embed_type="simple",
    num_class_embeds=10
).to("cuda")

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training
for epoch in range(10):
    for images, labels in tqdm(dataloader):
        images, labels = images.to("cuda"), labels.to("cuda")
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device).long()

        noisy = scheduler.add_noise(images, noise, timesteps)
        noise_pred = model(noisy, timesteps, class_labels=labels).sample

        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    torch.save(model.state_dict(), f"StableDDPM/unet_epoch_{epoch}.pt")

# Save final model
torch.save(model.state_dict(), "StableDDPM/unet_final.pt")
