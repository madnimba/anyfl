import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import os

# --------------------------
# SAME ENCODER AS YOUR LEFT-VIEW SimCLR MODEL
# --------------------------
class SmallEnc(nn.Module):
    def __init__(self, out_dim=64):   # SIMCLR_LATENT = 64
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),      # 28x14 → 14x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),      # 14x7 → 7x3
            nn.Flatten(),
            nn.Linear(64*7*3, out_dim)
        )

    def forward(self, x):
        return self.f(x)

# --------------------------
# LOAD TRAINED LEFT-VIEW ENCODER WEIGHTS
# --------------------------
ENC_PATH = "./clusters/MNIST_encoder.pth"

encoder = SmallEnc().to("cpu")
encoder.load_state_dict(torch.load(ENC_PATH, map_location="cpu"))
encoder.eval()
print("[OK] Loaded encoder:", ENC_PATH)

# --------------------------
# LOAD MNIST & EXTRACT RIGHT HALF
# --------------------------
def right_half(x):  # x: [1,28,28]
    return x[:, :, 14:]    # take right 14 columns

tf = transforms.ToTensor()
mnist = datasets.MNIST(".", train=True, download=True, transform=tf)

X_right = []
for i in range(len(mnist)):
    img, lbl = mnist[i]
    X_right.append(right_half(img))

X_right = torch.stack(X_right, dim=0)   # [N,1,28,14]
print("Right-view tensor:", X_right.shape)

ds = TensorDataset(X_right)
dl = DataLoader(ds, batch_size=256, shuffle=False)

# --------------------------
# ENCODE ALL RIGHT-VIEW IMAGES
# --------------------------
Z_right = []
with torch.no_grad():
    for (xb,) in dl:
        z = encoder(xb)
        Z_right.append(z.numpy())

Z_right = np.vstack(Z_right).astype(np.float32)
print("Right embeddings:", Z_right.shape)

# --------------------------
# SAVE OUTPUT
# --------------------------
os.makedirs("./clusters", exist_ok=True)
np.save("./clusters/MNIST_right_embeddings.npy", Z_right)

print("\n[OK] Saved right embeddings → ./clusters/MNIST_right_embeddings.npy")
