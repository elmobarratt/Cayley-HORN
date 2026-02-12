import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from CayleyHORN import *



# Hyperparameters
num_hidden = 8
epochs = 3
batch_size = 128
lr_readout = 1e-3
lr_hebb = 1e-4
h = 1.0
alpha = 0.04
omega = 0.224
gamma = 1e-2


# Dataset
train_set = torchvision.datasets.MNIST(
    root='data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_set = torchvision.datasets.MNIST(
    root='data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1000,
    shuffle=False
)


# Model
model = CHORN(
    num_input=1,
    num_nodes=num_hidden,
    num_output=10,
    h=h,
    alpha=alpha,
    omega=omega,
    gamma=gamma,
    learning_rule="hebbian"
)

# Freeze S from backprop
model.S.requires_grad = False

# Train only readout layer with backprop
optimiser = optim.Adam(model.readout.parameters(), lr=lr_readout)
criterion = nn.CrossEntropyLoss()


# Evaluation
def evaluate():
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.reshape(-1, 1, 784)
            images = images.permute(2, 0, 1)

            output = model(images, random_init=True)
            pred = output['output'].argmax(dim=1)

            correct += (pred == labels).sum().item()

    acc = 100.0 * correct / len(test_loader.dataset)
    return acc

# Training Loop
for epoch in range(epochs):

    model.train()

    for images, labels in tqdm(train_loader):

        images = images.reshape(-1, 1, 784)
        images = images.permute(2, 0, 1)

        optimiser.zero_grad()

        output = model(images, random_init=True)
        prediction = output['output']

        loss = criterion(prediction, labels)
        loss.backward()

        optimiser.step()

        # Hebbian update after forward pass
        model.hebbian_update(lr=lr_hebb)

    acc = evaluate()
    print(f"Epoch {epoch} | Test accuracy: {acc:.2f}%")


# %%
import matplotlib.pyplot as plt

ix=201
# prepare single example
image, label = test_set[ix]
images = image.reshape(1, 1, 784).permute(2, 0, 1)

with torch.no_grad():
    out = model(images, record=True)
    logits = out["output"]
    probs = torch.softmax(logits, dim=1).squeeze()
    phi_t = out["rec_phi_t"][0]          # (T, n_phase)
    rec_x_t = out["rec_x_t"][0]              # (T, n_nodes)
    rec_y_t = out["rec_y_t"][0]              # (T, n_nodes)

pred = probs.argmax().item()

print(f"True label: {label}")
print(f"Predicted label: {pred}")

# number of oscillator nodes
n_nodes = model.num_nodes

# phase-equivariant state normalization
phi_plot = phi_t.clone() # (T, n_phase)
amplitudes = phi_plot[:, :n_nodes]
amp_min = amplitudes.min(dim=0, keepdim=True)[0]
amp_max = amplitudes.max(dim=0, keepdim=True)[0]
amplitudes_norm = (amplitudes - amp_min) / (amp_max - amp_min + 1e-8)
phi_plot[:, :n_nodes] = amplitudes_norm

# Concatenate x and y as x1, y1, x2, y2 ... 
rec_concat = torch.empty((rec_x_t.shape[0], 2 * n_nodes), device=rec_x_t.device)
rec_concat[:, 0::2] = rec_x_t
rec_concat[:, 1::2] = rec_y_t

# Normalize rec_concat for visualization
rec_min = rec_concat.min(dim=0, keepdim=True)[0]
rec_max = rec_concat.max(dim=0, keepdim=True)[0]
rec_norm = (rec_concat - rec_min) / (rec_max - rec_min + 1e-8)

from matplotlib import gridspec

# Plot
fig = plt.figure(figsize=(12, 10), dpi=300)

gs = gridspec.GridSpec(
    nrows=3,
    ncols=2,
    height_ratios=[1, 0.2, 1],
    width_ratios=[1, 1],
    hspace=0.4,
    wspace=0.3
)

ax_prob = fig.add_subplot(gs[0, 0])
ax_img  = fig.add_subplot(gs[0, 1])
ax_spike = fig.add_subplot(gs[1, :])          # spans both columns
ax_phi   = fig.add_subplot(gs[2, :], sharex=ax_spike)

# Prediction distribution
ax_prob.bar(range(10), probs.cpu().numpy())
ax_prob.set_xticks(range(10))
ax_prob.set_ylim(0, 1)
ax_prob.set_title("Prediction distribution")
ax_prob.bar(pred, probs[pred].item(), alpha=0.8)

# Image
ax_img.imshow(image.reshape(28, 28), cmap="gray")
ax_img.axis("off")
ax_img.set_title(f"Input image (label = {label})")
ax_spike.imshow(
    image.reshape(1,-1).cpu().numpy(),
    aspect="auto",
    origin="lower",
    cmap="gray"
)
ax_spike.set_yticks([])
ax_spike.tick_params(left=False)
ax_spike.set_title("Input spiking signal pattern")

# Phase-equivariant evolution
im_phi = ax_phi.imshow(
    phi_plot.T.cpu().numpy(),
    aspect="auto",
    origin="lower",
    cmap="RdBu_r"
)
ax_phi.set_xlabel("Time step")
ax_phi.set_ylabel("Equivariant feature index")
ax_phi.set_title("Phase-equivariant state evolution")
# fig.colorbar(im_phi, ax=ax_phi, fraction=0.025, pad=0.02)

plt.suptitle(f"CHORN inference dynamics — predicted {pred}", fontsize=14)
plt.tight_layout()
plt.show()

