from matplotlib import pyplot as plt
import torch
from conditional import DiffuserCond, UNetCond


def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            if labels is not None:
                ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
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

model = UNetCond(num_labels=10)
model.load_state_dict(torch.load("conditional_model.pth", weights_only=True))
model.eval()

diffuser = DiffuserCond(num_timesteps, device=device)
model.to(device)

images, labels = diffuser.sample(model)

show_images(images, labels)
