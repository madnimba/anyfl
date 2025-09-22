# models_vfl.py
import torch
import torch.nn as nn
import torch.nn.functional as F

_CURRENT_DATASET = "GENERIC"

def set_current_dataset(name: str):
    global _CURRENT_DATASET
    _CURRENT_DATASET = (name or "GENERIC").upper()

# ---------- building blocks (CNN for half-images) ----------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_ch))
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.down is not None:
            x = self.down(x)
        return F.relu(x + y, inplace=True)

class HalfImageEncoder(nn.Module):
    """
    CNN for a single 'half' image (e.g., 3x32x16 for CIFAR10, 1x28x14 for MNIST).
    Uses LazyConv2d to auto-adapt input channels; GAP → Linear to embed_dim.
    """
    def __init__(self, width: int, embed_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LazyConv2d(width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.l1 = BasicBlock(width,   width,   stride=1)  # keep spatial
        self.l2 = BasicBlock(width,   width*2, stride=2)  # /2
        self.l3 = BasicBlock(width*2, width*4, stride=2)  # /4
        self.l4 = BasicBlock(width*4, width*4, stride=2)  # /8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(width*4, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class HalfTabularEncoder(nn.Module):
    """
    MLP for a half of tabular features (works for HAR/bank/mushroom; shape-agnostic).
    Uses LazyLinear to pick input dim on first forward.
    """
    def __init__(self, embed_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

# ---------- dataset-aware ClientA/B ----------
def _image_client():
    # Stronger CNN for CIFAR; lighter for MNIST/Fashion
    if _CURRENT_DATASET == "CIFAR10":
        return HalfImageEncoder(width=96, embed_dim=512)
    else:  # MNIST / FashionMNIST / KMNIST
        return HalfImageEncoder(width=32, embed_dim=256)

def _tabular_client():
    # HAR tends to like a bit more capacity than bank/mushroom
    if _CURRENT_DATASET in ("UCIHAR", "HAR", "UCI-HAR"):
        return HalfTabularEncoder(embed_dim=256, hidden=256)
    else:  # bank, mushroom (and other generic tabular)
        return HalfTabularEncoder(embed_dim=128, hidden=128)

class ClientA(nn.Module):
    def __init__(self):
        super().__init__()
        if _CURRENT_DATASET in ("CIFAR10", "MNIST", "FASHIONMNIST", "KMNIST"):
            self.features = _image_client()
        else:
            self.features = _tabular_client()
    def forward(self, x): return self.features(x)

class ClientB(nn.Module):
    def __init__(self):
        super().__init__()
        if _CURRENT_DATASET in ("CIFAR10", "MNIST", "FASHIONMNIST", "KMNIST"):
            self.features = _image_client()
        else:
            self.features = _tabular_client()
    def forward(self, x): return self.features(x)

# ---------- dataset-aware ServerC ----------
class ServerC(nn.Module):
    """
    Head on top of [featA || featB]. First layer is LazyLinear so
    it will infer the concat feature dim automatically.
    We still accept in_dim (ignored) for backward-compat.
    """
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        if _CURRENT_DATASET == "CIFAR10":
            h1, drop = 768, 0.20
        elif _CURRENT_DATASET in ("MNIST", "FASHIONMNIST", "KMNIST"):
            h1, drop = 256, 0.10
        elif _CURRENT_DATASET in ("UCIHAR", "HAR", "UCI-HAR"):
            h1, drop = 256, 0.10
        else:  # bank / mushroom / generic tabular
            h1, drop = 128, 0.05

        self.fc = nn.Sequential(
            nn.LazyLinear(h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(h1, num_classes)
        )

    def forward(self, xa, xb):
        return self.fc(torch.cat([xa, xb], dim=1))
