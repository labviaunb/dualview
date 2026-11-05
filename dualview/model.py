"""DualView: a U-shaped network with a CNN encoder on the enhanced view and DINOv3 + LoRA on the natural view, fused at
four encoder levels (gated residual fusion, with local cross-attention at 1/8 and 1/16).

Independent implementation; does not import DinoUNet or RRWNet.
Public API: model(natural_rgb, enhanced_rgb) -> logits [A, V, BV].
Inputs are registered BCHW float RGB in [0, 1]; normalization is internal.
"""
from contextlib import nullcontext
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = ("A", "V", "BV")
ARCHITECTURE_VERSION = "dualview-final-v1"


def norm2d(channels):
    # At least two channels per group, also valid for tiny spatial maps.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBlock(nn.Module):
    """Two ordinary convolutions; GroupNorm is independent of batch size."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm2d(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm2d(out_ch), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class LocalCrossAttention(nn.Module):
    """CNN queries select a 3x3 neighborhood of DINO keys/values.

    Attention operates on the DINO patch grid, NOT the full-resolution grid.
    Complexity is O(B * N_patch * 9 * dim), excluding the DINO backbone itself.
    Boundary padding is masked; each patch always has its own valid key.
    """
    def __init__(self, cnn_ch, dim, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError("fusion_dim must be divisible by attention heads")
        self.heads, self.dim = heads, dim
        self.query = nn.Sequential(norm2d(cnn_ch), nn.Conv2d(cnn_ch, dim, 1))
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.output = nn.Conv2d(dim, dim, 1)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 9))
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, cnn, dino):
        b, c, h, w = dino.shape
        n, d = h * w, c // self.heads
        q = resize(self.query(cnn), (h, w)).reshape(b, self.heads, d, n)
        k = F.unfold(self.key(dino), 3, padding=1).reshape(b, self.heads, d, 9, n)
        v = F.unfold(self.value(dino), 3, padding=1).reshape(b, self.heads, d, 9, n)
        # Compute softmax in float32 under AMP.
        scores = (q.float().unsqueeze(3) * k.float()).sum(2) / math.sqrt(d)
        scores = scores + self.relative_bias[None, :, :, None]
        valid = F.unfold(dino.new_ones(1, 1, h, w), 3, padding=1).bool()
        scores = scores.masked_fill(~valid[:, None], float("-inf"))
        weights = scores.softmax(dim=2).to(v.dtype)
        message = (weights.unsqueeze(2) * v).sum(3).reshape(b, c, h, w)
        return dino + self.strength.tanh() * self.output(message)


class GatedFusion(nn.Module):
    """Aligned DINO residual with a spatial AND channel-dependent CNN gate.

    No FiLM, FAPM, deformable convolution or deformable attention.
    Nonzero, small residual initialization allows immediate adapter gradients.
    """
    def __init__(self, cnn_ch, dim, use_attention=False, heads=4):
        super().__init__()
        self.attention = LocalCrossAttention(cnn_ch, dim, heads) if use_attention else None
        self.dino_to_cnn = nn.Sequential(nn.Conv2d(dim, cnn_ch, 1, bias=False), norm2d(cnn_ch))
        self.cnn_norm = norm2d(cnn_ch)
        hidden = min(cnn_ch, 64)
        self.gate = nn.Sequential(
            nn.Conv2d(cnn_ch * 3, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, cnn_ch, 1), nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].bias, -1.0)
        self.message = nn.Sequential(
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1, groups=cnn_ch, bias=False),
            norm2d(cnn_ch), nn.SiLU(), nn.Conv2d(cnn_ch, cnn_ch, 1),
        )
        self.strength = nn.Parameter(torch.full((1, cnn_ch, 1, 1), 0.1))

    def forward(self, cnn, dino):
        if self.attention is not None:
            dino = self.attention(cnn, dino)
        d = resize(self.dino_to_cnn(dino), cnn.shape[-2:])
        c = self.cnn_norm(cnn)
        gate = self.gate(torch.cat((c, d, c * d), dim=1))
        return cnn + self.strength.tanh() * gate * self.message(d)


class LayerMixer(nn.Module):
    """A learned mixture of DINO depths for each CNN contraction.

    All DINO features have the SAME spatial resolution. Only the CNN creates
    a true spatial hierarchy. We do not call interpolated ViT maps a pyramid.
    """
    def __init__(self, embed_dim, dim, n_layers=4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, dim, 1, bias=False), norm2d(dim))
            for _ in range(n_layers)
        ])
        initial = torch.zeros(4, n_layers)
        for i in range(4):
            initial[i, round(i * (n_layers - 1) / 3)] = 2.0
        self.layer_logits = nn.Parameter(initial)
        self.refine = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                          norm2d(dim), nn.SiLU(), nn.Conv2d(dim, dim, 1))
            for _ in range(4)
        ])

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError("DINO feature count does not match LayerMixer")
        projected = [p(f) for p, f in zip(self.projections, features)]
        weights = self.layer_logits.softmax(dim=-1)
        return [self.refine[i](sum(weights[i, j] * f for j, f in enumerate(projected)))
                for i in range(4)]


class DinoFeatures(nn.Module):
    """Hugging Face DINOv3 with optional LoRA, no segmentation decoder.

    An injected HF-compatible backbone supports offline tests. `stage_indexes`
    are zero-based transformer block indices, e.g. (5,11,17,23) for ViT-L.
    """
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
                 token=None, stage_indexes=None, lora_rank=16, lora_alpha=32,
                 lora_dropout=0.05, lora_targets=("q_proj", "v_proj"),
                 gradient_checkpointing=True, revision=None, backbone=None):
        super().__init__()
        if backbone is None:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(
                model_name, token=token, revision=revision, attn_implementation="sdpa"
            )
        config = backbone.config
        if getattr(config, "model_type", None) != "dinov3_vit":
            raise ValueError("Expected a Hugging Face DINOv3 ViT backbone")
        self.embed_dim = config.hidden_size
        self.patch_size = int(config.patch_size)
        self.prefix_tokens = 1 + config.num_register_tokens
        depth = config.num_hidden_layers
        self.stage_indexes = tuple(stage_indexes if stage_indexes is not None
                                   else [round((i + 1) * depth / 4) - 1 for i in range(4)])
        if (len(self.stage_indexes) != 4 or len(set(self.stage_indexes)) != 4
                or sorted(self.stage_indexes) != list(self.stage_indexes)
                or min(self.stage_indexes) < 0 or max(self.stage_indexes) >= depth):
            raise ValueError("Select four distinct increasing DINO block indices within depth")
        # Paired geometric augmentation is owned by the dataset.
        # Disable pretraining-only stochastic coordinate augmentation in RoPE.
        for field in ("pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale"):
            if hasattr(config, field):
                setattr(config, field, None)
        backbone.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            present = {name.rsplit(".", 1)[-1] for name, mod in backbone.named_modules()
                       if isinstance(mod, nn.Linear)}
            missing = set(lora_targets) - present
            if missing:
                raise ValueError(f"LoRA target modules absent from backbone: {sorted(missing)}")
            backbone = get_peft_model(backbone, LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(lora_targets), bias="none",
            ))
            if gradient_checkpointing:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone = backbone
        self.has_lora = lora_rank > 0
        self.adaptation_active = self.has_lora
        self.backbone.train(self.training and self.adaptation_active)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def set_adaptation(self, active):
        self.adaptation_active = bool(active and self.has_lora)
        self.backbone.train(self.training and self.adaptation_active)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode and self.adaptation_active)
        return self

    def forward(self, natural):
        h, w = natural.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Model must pad both views before DINO feature extraction")
        x = (natural - self.mean) / self.std
        context = nullcontext() if self.adaptation_active else torch.no_grad()
        with context:
            hidden = self.backbone(pixel_values=x, output_hidden_states=True,
                                   return_dict=True).hidden_states
        if hidden is None or len(hidden) != self.backbone.config.num_hidden_layers + 1:
            raise RuntimeError("Unexpected HF hidden_states layout; verify Transformers version")
        gh, gw = h // self.patch_size, w // self.patch_size
        features = []
        for idx in self.stage_indexes:
            tokens = hidden[idx + 1]
            if tokens.shape[1] != self.prefix_tokens + gh * gw:
                raise RuntimeError("Unexpected CLS/register/patch token count")
            tokens = tokens[:, self.prefix_tokens:, :]
            # Comparable depth statistics without sharing the final pretrained norm.
            tokens = F.layer_norm(tokens.float(), (self.embed_dim,)).to(tokens.dtype)
            features.append(tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, gh, gw))
        return features


class DualViewDinoUNet(nn.Module):
    """A single contracting CNN and single expanding CNN, guided by DINO.

    fusion: 'cnn' (no DINO), 'concat' (baseline), 'gated', or 'gated_local'.
    fusion_stages contains 1..4 for H/2,H/4,H/8,H/16. Full-res skip stays CNN.
    dino_view='enhanced' is an input ablation; default is natural RGB.
    """
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256, 512),
                 fusion_dim=64, fusion="gated_local", fusion_stages=(1, 2, 3, 4),
                 attention_stages=(3, 4), heads=4, dino_view="natural", dino=None,
                 **dino_kwargs):
        super().__init__()
        if len(channels) != 5 or min(channels) < 2:
            raise ValueError("channels must contain five widths >= 2")
        if fusion not in ("cnn", "concat", "gated", "gated_local"):
            raise ValueError("Unknown fusion ablation")
        if dino_view not in ("natural", "enhanced"):
            raise ValueError("dino_view must be natural or enhanced")
        if any(i not in range(1, 5) for i in fusion_stages):
            raise ValueError("fusion_stages must be a subset of (1,2,3,4)")
        self.fusion_mode, self.dino_view = fusion, dino_view
        self.fusion_stages = tuple(fusion_stages)
        self.dino = None if fusion == "cnn" else (dino if dino is not None else DinoFeatures(**dino_kwargs))
        self.multiple = math.lcm(16, self.dino.patch_size if self.dino else 16)
        self.encoder = nn.ModuleList([ConvBlock(3, channels[0])] + [
            ConvBlock(channels[i - 1], channels[i]) for i in range(1, 5)])
        self.fusions = nn.ModuleDict()
        if self.dino is not None:
            self.mixer = LayerMixer(self.dino.embed_dim, fusion_dim)
            for i in self.fusion_stages:
                if fusion == "concat":
                    self.fusions[str(i)] = nn.Sequential(
                        nn.Conv2d(channels[i] + fusion_dim, channels[i], 1, bias=False),
                        norm2d(channels[i]), nn.SiLU())
                else:
                    self.fusions[str(i)] = GatedFusion(
                        channels[i], fusion_dim,
                        fusion == "gated_local" and i in attention_stages, heads)
        self.up = nn.ModuleList([nn.Conv2d(channels[i + 1], channels[i], 1) for i in range(4)])
        self.decoder = nn.ModuleList([ConvBlock(2 * channels[i], channels[i]) for i in range(4)])
        self.head = nn.Conv2d(channels[0], 3, 1)

    def forward(self, natural, enhanced):
        if natural.ndim != 4 or natural.shape != enhanced.shape or natural.shape[1] != 3:
            raise ValueError("Expected two aligned BCHW RGB tensors of identical shape")
        if not natural.is_floating_point() or not enhanced.is_floating_point():
            raise TypeError("Inputs must be floating point RGB in [0,1]")
        h, w = natural.shape[-2:]
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        # Same right/bottom padding: no resize or geometric distortion between views.
        if ph or pw:
            natural = F.pad(natural, (0, pw, 0, ph), mode="replicate")
            enhanced = F.pad(enhanced, (0, pw, 0, ph), mode="replicate")
        dino_maps = None
        if self.dino is not None:
            dino_maps = self.mixer(self.dino(natural if self.dino_view == "natural" else enhanced))
        x, skips = enhanced * 2.0 - 1.0, []
        for i, block in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            if str(i) in self.fusions:
                d = dino_maps[i - 1]
                if self.fusion_mode == "concat":
                    x = self.fusions[str(i)](torch.cat((x, resize(d, x.shape[-2:])), 1))
                else:
                    x = self.fusions[str(i)](x, d)
            skips.append(x)  # Fused features feed BOTH next contraction and skip.
        for i in reversed(range(4)):
            x = self.up[i](resize(x, skips[i].shape[-2:]))
            x = self.decoder[i](torch.cat((x, skips[i]), 1))
        return self.head(x)[..., :h, :w]

    def optimizer_groups(self, lr=5e-4, lora_lr_ratio=0.1, weight_decay=1e-2):
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_lora = ".lora_" in name
            decay = p.ndim > 1 and not any(t in name for t in ("strength", "layer_logits", "relative_bias"))
            key = (is_lora, decay)
            groups.setdefault(key, []).append(p)
        return [{"params": params, "lr": lr * (lora_lr_ratio if is_lora else 1.0),
                 "weight_decay": weight_decay if decay else 0.0}
                for (is_lora, decay), params in groups.items()]


