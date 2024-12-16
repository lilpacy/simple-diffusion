import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy

from dit import model, diffuser, device, num_timesteps

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

batch_size = 64
epochs = 10
lr = 5e-4
ema_decay = 0.9999

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=preprocess
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

model.to(device)
ema_model = deepcopy(model).to(device)
optimizer = AdamW(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    model.train()
    loss_sum = 0.0
    cnt = 0
    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)
        x_noisy, noise = diffuser.add_noise(x, t)

        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch + 1} | Loss: {loss_avg:.4f}")
    torch.save(model.state_dict(), "conditional_transformer_model.pth")
    torch.save(ema_model.state_dict(), "ema_conditional_transformer_model.pth")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
