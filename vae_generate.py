import torch
from vae import VAE
import torchvision
import matplotlib.pyplot as plt


input_dim = 784
hidden_dim = 200
latent_dim = 20

model = VAE(input_dim, hidden_dim, latent_dim)
model.load_state_dict(torch.load("vae_model.pth", weights_only=True))
model.eval()

with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)
    x = model.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28)

grid_img = torchvision.utils.make_grid(
    generated_images, nrow=8, padding=2, normalize=True
)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
