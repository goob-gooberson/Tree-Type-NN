"""Feature trunks that feed the tree.

Three kinds of trunk are provided:

* ``PixelTrunk``   : no learning, the (downsampled) image itself.
* ``SmallCNN``     : a compact CNN trained from scratch on the target dataset and then frozen.
* ``ResNet18``     : an ImageNet-pretrained ResNet-18 used as a frozen, data-independent trunk.

Every trunk exposes ``features(x)`` returning a pooled feature vector for a float NCHW batch,
and is differentiable so that receptive fields of tree nodes can be traced back to pixels.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D


# --------------------------------------------------------------------------------------
# Pixel trunk
# --------------------------------------------------------------------------------------
class PixelTrunk(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.size = size

    def features(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="area")
        return x.flatten(1)


# --------------------------------------------------------------------------------------
# Small CNN trunk trained from scratch
# --------------------------------------------------------------------------------------
def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """VGG-style CNN. For 64x64 inputs the first conv has stride 2 so cost matches 32x32."""

    def __init__(self, num_classes=10, width=32, in_res=32):
        super().__init__()
        w = width
        s = 2 if in_res >= 64 else 1
        self.body = nn.Sequential(
            conv_bn(3, w, s), conv_bn(w, w), nn.MaxPool2d(2),
            conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2),
            conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), nn.MaxPool2d(2),
        )
        self.feat_dim = 4 * w
        self.head = nn.Linear(4 * w, num_classes)

    def spatial(self, x):
        return self.body(x)

    def features(self, x):
        return self.body(x).mean((2, 3))

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(model, Xtr, ytr, Xte=None, yte=None, epochs=15, bs=128, lr=0.05, wd=5e-4,
              aug=True, log=print, device="cpu"):
    """Train a CNN trunk with SGD + one-cycle LR and crop/flip augmentation.

    Xtr/Xte are uint8 NHWC numpy arrays. ``device`` = "cpu" or "cuda". The model is moved back
    to the CPU at the end.
    """
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    ytr_t = torch.as_tensor(ytr)
    hist = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr))
        tot, correct, loss_sum = 0, 0, 0.0
        for b in range(0, len(Xtr), bs):
            idx = perm[b:b + bs]
            xb = D.to_float(Xtr[idx.numpy()])
            if aug:
                xb = D.augment(xb, pad=xb.shape[-1] // 8)
            xb = xb.to(device)
            yb = ytr_t[idx].to(device)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * len(idx)
        rec = dict(epoch=ep + 1, train_loss=loss_sum / tot, train_acc_aug=correct / tot,
                   secs=time.time() - t0)
        if Xte is not None:
            rec["test_acc"] = float(accuracy(model, Xte, yte, device=device))
        hist.append(rec)
        log(rec)
    model.to("cpu")
    return hist


@torch.no_grad()
def accuracy(model, X, y, bs=1000, device="cpu"):
    model.eval()
    correct = 0
    for b in range(0, len(X), bs):
        out = model(D.to_float(X[b:b + bs]).to(device))
        correct += (out.argmax(1).cpu().numpy() == y[b:b + bs]).sum()
    return correct / len(X)


# --------------------------------------------------------------------------------------
# ImageNet pretrained ResNet-18 trunk
# --------------------------------------------------------------------------------------
class ResNet18Trunk(nn.Module):
    def __init__(self, weights_path, res=128):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        sd = torch.load(weights_path, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("fc")], missing
        m.fc = nn.Identity()
        self.m = m.eval()
        self.res = res
        self.feat_dim = 512

    def spatial(self, x):
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return x

    def features(self, x):
        return self.spatial(x).mean((2, 3))


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract(trunk, X, bs=500, log=None, device="cpu"):
    """Run a trunk over uint8 NHWC images and return float32 numpy features."""
    if hasattr(trunk, "eval"):
        trunk.eval()
    if hasattr(trunk, "to"):
        trunk.to(device)
    feats = []
    t0 = time.time()
    for b in range(0, len(X), bs):
        feats.append(trunk.features(D.to_float(X[b:b + bs]).to(device)).float().cpu().numpy())
        if log and (b // bs) % 20 == 0:
            log(f"  extracted {min(b + bs, len(X))}/{len(X)} ({time.time() - t0:.0f}s)")
    if hasattr(trunk, "to"):
        trunk.to("cpu")
    return np.concatenate(feats)


class FeatureMap:
    """Post-trunk feature normalisation: standardise -> (optional) PCA -> standardise.

    Kept as a tiny torch module so gradients can flow from tree nodes back to the image.
    """

    def __init__(self, Ftr, dim=None):
        mu = Ftr.mean(0)
        sd = Ftr.std(0) + 1e-6
        Z = (Ftr - mu) / sd
        if dim is not None and dim < Z.shape[1]:
            U, S, Vt = np.linalg.svd(Z[np.random.default_rng(0).permutation(len(Z))[:20000]],
                                     full_matrices=False)
            P = Vt[:dim].T
        else:
            P = np.eye(Z.shape[1], dtype=np.float32)
        Zp = Z @ P
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sd = torch.tensor(sd, dtype=torch.float32)
        self.P = torch.tensor(P, dtype=torch.float32)
        self.mu2 = torch.tensor(Zp.mean(0), dtype=torch.float32)
        self.sd2 = torch.tensor(Zp.std(0) + 1e-6, dtype=torch.float32)
        self.dim = self.P.shape[1]

    def __call__(self, F_):
        F_ = torch.as_tensor(F_, dtype=torch.float32)
        return (((F_ - self.mu) / self.sd) @ self.P - self.mu2) / self.sd2

    def state(self):
        return {k: getattr(self, k) for k in ["mu", "sd", "P", "mu2", "sd2"]}

    @classmethod
    def from_state(cls, st):
        o = cls.__new__(cls)
        for k, v in st.items():
            setattr(o, k, v)
        o.dim = o.P.shape[1]
        return o
