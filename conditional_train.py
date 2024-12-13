import torch
import torchvision
from torchvision import transforms
from conditional import UNetCond, DiffuserCond
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=preprocess
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = DiffuserCond(num_timesteps, device=device)
model = UNetCond(num_labels=10)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

"""
総データ数：60,000枚
バッチサイズ：128
エポック数：10
総更新回数：約4,690回（469バッチ × 10エポック）
"""

losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch} | Loss: {loss_avg}")

torch.save(model.state_dict(), "conditional_model.pth")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
