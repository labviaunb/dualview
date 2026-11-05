"""Training loss of DualView: masked binary cross-entropy on [A, V, BV] plus soft-clDice on the artery and vein maps.

Crossings count as both artery and vein, and vessels without A/V label supervise only the vessel map (BV).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def supervision_masks(target, fov=None):
    """Per-channel supervision masks: unknown vessels are supervised only in BV; crossings may be A = V = 1."""
    fov = torch.ones_like(target[:, :1]) if fov is None else (fov > .5).to(target.dtype)
    unknown = (target[:, 2:3] > .5) & (target[:, :1] < .5) & (target[:, 1:2] < .5)
    av = fov * (~unknown).to(fov.dtype)
    return torch.cat((av, av, fov), dim=1)


class MaskedVesselLoss(nn.Module):
    """Binary cross-entropy on [A, V, BV] inside the supervision masks, averaged per channel and summed over channels."""

    def forward(self, logits, target, fov=None):
        if logits.shape != target.shape or logits.shape[1] != 3:
            raise ValueError('logits and target must be BCHW with channels [A, V, BV]')
        logits, target = logits.float(), target.float()
        weights = supervision_masks(target, fov)
        denom = weights.sum((0, 2, 3)).clamp_min(1)
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return ((raw * weights).sum((0, 2, 3)) / denom).sum()


# Soft skeleton adapted from jocpae/clDice (CVPR 2021), 2D implementation.
# https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
# MIT License
# Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def _cldice_erode2d(x):
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton2d(x, iterations=10):
    """Differentiable morphological skeleton of a BCHW map in [0, 1]."""
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError('iterations must be a non-negative integer')

    def opening(t):
        return F.max_pool2d(_cldice_erode2d(t), 3, stride=1, padding=1)

    skeleton = F.relu(x - opening(x))
    for _ in range(iterations):
        x = _cldice_erode2d(x)
        delta = F.relu(x - opening(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_valid_interior(valid, iterations):
    """Support mask that excludes invalid pixels and a halo of iterations + 2 pixels around them.

    A skeleton value at iteration k depends on a neighborhood of radius k + 2, so the halo avoids scoring artificial edges
    created by unknown vessels, the FOV mask or the image border.
    """
    radius = iterations + 2
    valid = valid.float()
    x = F.pad(valid, (radius, radius, 0, 0), value=0.)
    x = -F.max_pool2d(-x, (1, 2 * radius + 1), stride=1)
    x = F.pad(x, (0, 0, radius, radius), value=0.)
    return -F.max_pool2d(-x, (2 * radius + 1, 1), stride=1)


class MaskedSoftClDiceLoss(nn.Module):
    """Class-wise soft-clDice on the selected channels, averaged over the image/channel pairs with a nonempty skeleton."""

    def __init__(self, channels=(0, 1), iterations=10, smooth=1.0):
        super().__init__()
        if not channels or any(c not in (0, 1, 2) for c in channels) or len(set(channels)) != len(channels):
            raise ValueError('channels must select unique indices of [A=0, V=1, BV=2]')
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError('iterations must be a non-negative integer')
        if not math.isfinite(float(smooth)) or smooth <= 0:
            raise ValueError('smooth must be positive and finite')
        self.channels = tuple(channels)
        self.iterations, self.smooth = iterations, float(smooth)

    def forward(self, logits, target, fov=None):
        if logits.ndim != 4 or logits.shape != target.shape or logits.shape[1] != 3:
            raise ValueError('Expected BCHW logits and target [A, V, BV]')
        if fov is not None and fov.shape != target[:, :1].shape:
            raise ValueError('Expected a Bx1xHxW fov')
        with torch.autocast(device_type=logits.device.type, enabled=False):
            prob = logits[:, self.channels].float().sigmoid()
            gt = target[:, self.channels].float()
            valid = supervision_masks(target.float(), fov)[:, self.channels]
            interior = cldice_valid_interior(valid, self.iterations)
            prob, gt = prob * valid, gt * valid
            predicted_skeleton = soft_skeleton2d(prob, self.iterations) * interior
            with torch.no_grad():
                target_skeleton = soft_skeleton2d(gt, self.iterations) * interior
            dims = (-2, -1)
            precision = ((predicted_skeleton * gt).sum(dims) + self.smooth) / (predicted_skeleton.sum(dims) + self.smooth)
            sensitivity = ((target_skeleton * prob).sum(dims) + self.smooth) / (target_skeleton.sum(dims) + self.smooth)
            score = 2 * precision * sensitivity / (precision + sensitivity).clamp_min(1e-7)
            active = (target_skeleton.sum(dims) > 0).to(prob.dtype)
            return ((1 - score) * active).sum() / active.sum().clamp_min(1.)


class DualViewLoss(nn.Module):
    """Masked BCE plus ``cldice_weight`` times soft-clDice (artery and vein maps by default)."""

    def __init__(self, cldice_weight=.1, cldice_channels=(0, 1), cldice_iterations=10):
        super().__init__()
        if not math.isfinite(float(cldice_weight)) or cldice_weight < 0:
            raise ValueError('cldice_weight must be non-negative and finite')
        self.base_loss, self.weight = MaskedVesselLoss(), float(cldice_weight)
        self.topology_loss = MaskedSoftClDiceLoss(cldice_channels, cldice_iterations)

    def forward(self, logits, target, fov=None):
        base = self.base_loss(logits, target, fov)
        if self.weight == 0:
            return base
        return base + self.weight * self.topology_loss(logits, target, fov)
