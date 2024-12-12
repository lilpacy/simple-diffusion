from matplotlib import pyplot as plt
import torch
from diffusion import UNet, Diffuser


def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")
            i += 1
    plt.show()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

num_timesteps = 1000

model = UNet()
model.load_state_dict(torch.load("diffusion_model.pth", weights_only=True))
model.eval()

diffuser = Diffuser(num_timesteps, device=device)
model.to(device)

images = diffuser.sample(model)

show_images(images)
