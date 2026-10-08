"""Receptive-field analysis of tree nodes.

The receptive field of a node is studied through the composite map

    image --trunk--> pooled features --FeatureMap--> standardised/PCA features --node net--> logit

Tools:
* ``input_gradients``   : d logit / d image for a batch of images (exact template for pixel trunks).
* ``saliency_stats``    : spatial concentration of |gradient| (area holding 50% of the energy).
* ``top_activating``    : indices of training images with the largest node logit.
* ``firing``            : how often a node's own network fires on train / test data.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D


def composite(trunk, fmap, net):
    def f(x):
        return net(fmap(trunk.features(x)))
    return f


def input_gradients(trunk, fmap, net, imgs_uint8, bs=64):
    """Return gradients d logit / d x (N,3,H,W) and the logits."""
    f = composite(trunk, fmap, net)
    grads, logits = [], []
    for b in range(0, len(imgs_uint8), bs):
        x = D.to_float(imgs_uint8[b:b + bs]).requires_grad_(True)
        z = f(x)
        if z.dim() > 1:
            z = z[:, 0]
        z.sum().backward()
        grads.append(x.grad.detach())
        logits.append(z.detach())
    return torch.cat(grads), torch.cat(logits)


def saliency_maps(grads):
    """|grad| summed over colour channels, each map normalised to sum 1."""
    s = grads.abs().sum(1)
    return s / s.flatten(1).sum(1).clamp_min(1e-12).view(-1, 1, 1)


def area50(sal):
    """Fraction of pixels needed to capture 50% of the saliency mass (lower = more localised).

    Images on which the node's gradient is exactly zero (inactive ReLUs) give NaN and are ignored.
    """
    v = sal.flatten(1).sort(1, descending=True).values.cumsum(1)
    a = ((v < 0.5).sum(1).float() + 1) / v.shape[1]
    return torch.where(v[:, -1] > 0, a, torch.full_like(a, float("nan")))


def centre_mass(sal):
    """Fraction of saliency inside the central half (by side) of the image."""
    h, w = sal.shape[1:]
    c = sal[:, h // 4: h - h // 4, w // 4: w - w // 4].flatten(1).sum(1)
    return torch.where(sal.flatten(1).sum(1) > 0, c, torch.full_like(c, float("nan")))


def gradient_image(g):
    """Map a (3,H,W) gradient to an RGB image in [0,1] for display."""
    g = g / (g.abs().max() + 1e-12)
    return (0.5 + 0.5 * g).permute(1, 2, 0).clamp(0, 1).numpy()


@torch.no_grad()
def own_logits(net, X):
    z = net(X)
    return z if z.dim() == 1 else z


def top_activating(net, X, k=8, mask=None):
    z = own_logits(net, X)
    if mask is not None:
        z = torch.where(mask, z, torch.full_like(z, -1e9))
    return torch.topk(z, k).indices
