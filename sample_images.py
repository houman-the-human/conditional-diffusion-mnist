import os
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel  # Adjust import to your model definition
# If your UNet2DConditionModel is in the same repo but different file, change accordingly
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model with same params as in training
model = UNet2DConditionModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 64),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=128,
).to(device)

# Load model weights (use weights_only=True to silence the warning if using PyTorch 2.0+)
model.load_state_dict(torch.load("DDPM/unet_final.pt", map_location=device))

# Load class embedder (embedding layer)
num_classes = 10
embedding_dim = 128
class_embedder = nn.Embedding(num_classes, embedding_dim).to(device)
class_embedder.load_state_dict(torch.load("DDPM/class_embedder.pt", map_location=device))

model.eval()
class_embedder.eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# Move existing buffers to device
for attr in ["alphas_cumprod", "betas", "one"]:
    if hasattr(noise_scheduler, attr):
        setattr(noise_scheduler, attr, getattr(noise_scheduler, attr).to(device))

batch_size = 16
num_steps = noise_scheduler.config.num_train_timesteps

# Start from pure noise
samples = torch.randn(batch_size, 1, 28, 28, device=device)  # assuming MNIST shape

# For conditional generation, specify class labels 
class_labels = torch.full((batch_size,), 0, device=device, dtype=torch.long)

with torch.no_grad():
    for t in tqdm(reversed(range(num_steps)), desc="Sampling timesteps"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Get class embeddings
        embeddings = class_embedder(class_labels).unsqueeze(1)

        # Predict noise
        noise_pred = model(samples, t_batch, encoder_hidden_states=embeddings).sample

        prev_samples = []
        for i in range(batch_size):
            noise_pred_i = noise_pred[i : i + 1]
            t_i = t_batch[i : i + 1]
            sample_i = samples[i : i + 1]
            # Use scheduler step; all tensors are on the same device
            prev_sample_i = noise_scheduler.step(noise_pred_i, t_i, sample_i).prev_sample
            prev_samples.append(prev_sample_i)
        samples = torch.cat(prev_samples, dim=0)

# Denormalize samples if necessary, clamp between 0 and 1
samples = samples.clamp(0, 1).cpu()

# Make sure 'Samples' folder exists
os.makedirs("Samples", exist_ok=True)

# Suppose `samples` is your tensor of generated images, shape: (batch_size, C, H, W)
# Save each sample image individually
for i, sample in enumerate(samples):
    save_image(sample, f"Samples/sample_{i}.png")
