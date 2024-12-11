import torch
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from vae import VAE


input_dim = 784
hidden_dim = 200
latent_dim = 20
epochs = 30
learning_rate = 3e-4
batch_size = 32

# dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
)
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    print(loss_avg)
    losses.append(loss_avg)

# save
torch.save(model.state_dict(), "vae_model.pth", weights_only=True)

# plot
epochs = list(range(1, epochs + 1))
plt.plot(epochs, losses, marker="o", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
