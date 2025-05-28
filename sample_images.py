import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 10
img_size = 64
class_label = 1  # change this to generate samples of another class

# Load model
model = UNet2DConditionModel(
    sample_size=img_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    class_embed_type="simple",
    num_class_embeds=10
).to(device)

# Load trained weights
model.load_state_dict(torch.load("DDPM/unet_final.pt", map_location=device))
model.eval()

# Scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Generate initial noise
samples = torch.randn(n_samples, 1, img_size, img_size).to(device)
labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)

# Sampling loop
with torch.no_grad():
    for t in reversed(range(scheduler.config.num_train_timesteps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(samples, t_batch, class_labels=labels).sample
        samples = scheduler.step(noise_pred, t_batch, samples).prev_sample

# Save to Samples/ directory
os.makedirs("Samples", exist_ok=True)
save_path = os.path.join("Samples", f"class_{class_label}.png")
grid = make_grid(samples.cpu(), nrow=5, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.savefig(save_path)
plt.close()
print(f"Samples saved to {save_path}")
