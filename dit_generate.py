import torch
from matplotlib import pyplot as plt

from dit import model, diffuser, device


model.load_state_dict(
    torch.load("conditional_transformer_model.pth", weights_only=True)
)
model.eval()
model.to(device)
with torch.no_grad():
    images, labels = diffuser.sample(model)


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


show_images(images, labels)
