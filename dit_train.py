import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot as plt

from dit import model, diffuser, device, num_timesteps

batch_size = 64

epochs = 5
lr = 5e-4  # 学習率を低減
device_type = "cuda" if device == "cuda" else "mps" if device == "mps" else "cpu"

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=preprocess
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

# 混合精度訓練とGradScalerを無効化
use_mixed_precision = False
scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

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

        if use_mixed_precision:
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                noise_pred = model(x_noisy, t, labels)
                loss = F.mse_loss(noise, noise_pred)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred = model(x_noisy, t, labels)
            loss = F.mse_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch} | Loss: {loss_avg}")

torch.save(model.state_dict(), "conditional_transformer_model.pth")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
