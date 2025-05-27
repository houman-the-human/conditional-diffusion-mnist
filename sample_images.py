import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Load model
model = UNet2DConditionModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
    class_embed_type="simple",
    num_class_embeds=10
).to("cuda")

# Load trained weights
model.load_state_dict(torch.load("DDPM/unet_final.pt"))
model.eval()

# Scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Generate class-conditioned samples
n = 10
labels = torch.tensor([1] * n).to("cuda")
sample = torch.randn(n, 1, 28, 28).to("cuda")

for t in reversed(range(scheduler.config.num_train_timesteps)):
    timestep = torch.full((n,), t, device="cuda", dtype=torch.long)
    with torch.no_grad():
        noise_pred = model(sample, timestep, class_labels=labels).sample
    sample = scheduler.step(noise_pred, timestep, sample).prev_sample

# Display
grid = make_grid(sample.cpu(), nrow=5, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()