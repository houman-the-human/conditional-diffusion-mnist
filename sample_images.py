import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch.nn.functional as F
from torchvision.utils import save_image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with the same architecture used during training
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
model.load_state_dict(torch.load("DDPM/unet_final.pt", map_location=device))
model.eval()

# Load class embedder
class_embedder = nn.Embedding(10, 256).to(device)
class_embedder.load_state_dict(torch.load("DDPM/class_embedder.pt", map_location=device))
class_embedder.eval()

# Noise scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# -------- User input for conditional generation --------
condition_class = 7  # <-- CHANGE this to the digit (0–9) you want to generate
batch_size = 16
labels = torch.full((batch_size,), condition_class, dtype=torch.long, device=device)

# Start from noise
samples = torch.randn(batch_size, 1, 64, 64, device=device)

# Sampling loop
with torch.no_grad():
    for t in tqdm(reversed(range(scheduler.config.num_train_timesteps)), desc=f"Sampling for class {condition_class}"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        embeddings = class_embedder(labels).unsqueeze(1)
        noise_pred = model(samples, t_batch, encoder_hidden_states=embeddings).sample
        samples = scheduler.step(noise_pred, t_batch, samples).prev_sample

# Post-processing
samples = samples.clamp(0, 1)
samples_resized = F.interpolate(samples, size=(28, 28), mode='bilinear', align_corners=False)

# Save images
os.makedirs("Samples", exist_ok=True)
for i, img in enumerate(samples_resized.cpu()):
    save_image(img, f"Samples/sample_class{condition_class}_{i}.png")
