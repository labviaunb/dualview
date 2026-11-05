"""Released DualView weights: names, download and loading.

Each file holds only the trained parameters (LoRA adapters, CNN encoder and decoder, fusion modules), about 59 MB. The frozen
DINOv3 ViT-L/16 backbone is downloaded from the Hugging Face Hub, which requires accepting the DINOv3 license there.

Models (``<training set>_seed<seed>_fold<fold>``):
- ``fundus-avseg_seed{42,43,44}_fold{0..3}``: trained on Fundus-AVSeg (MIT, as the code).
- ``fundus-avseg-rav_seed42_fold{0..3}``: trained on Fundus-AVSeg + RAV (CC BY-NC 4.0, the license of RAV).
"""
from pathlib import Path
import os
import urllib.request

import torch

from .model import DualViewDinoUNet

RELEASE_URL = 'https://github.com/labviaunb/dualview/releases/download/v1.0'
FORMAT = 'dualview-weights-v1'
MODELS = ([f'fundus-avseg_seed{s}_fold{f}' for s in (42, 43, 44) for f in range(4)]
          + [f'fundus-avseg-rav_seed42_fold{f}' for f in range(4)])
ENSEMBLES = {'fundus-avseg': [f'fundus-avseg_seed42_fold{f}' for f in range(4)],
             'fundus-avseg-rav': [f'fundus-avseg-rav_seed42_fold{f}' for f in range(4)]}
MODEL_KEYS = ('channels', 'fusion_dim', 'fusion', 'fusion_stages', 'attention_stages', 'dino_view', 'model_name',
              'revision', 'stage_indexes', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_targets')


def cache_dir():
    return Path(os.environ.get('DUALVIEW_CACHE', Path.home() / '.cache' / 'dualview'))


def checkpoint_path(name_or_path):
    """Local path of a released model (downloaded on first use) or of a local ``.pt`` file."""
    path = Path(name_or_path)
    if path.suffix == '.pt' and path.exists():
        return path
    if name_or_path not in MODELS:
        raise ValueError(f'Unknown model {name_or_path!r}; available: {", ".join(MODELS)}')
    target = cache_dir() / f'dualview_{name_or_path}.pt'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'{RELEASE_URL}/dualview_{name_or_path}.pt'
        print(f'Downloading {url}', flush=True)
        tmp = target.with_suffix('.tmp')
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    return target


def load_checkpoint(name_or_path):
    ckpt = torch.load(checkpoint_path(name_or_path), map_location='cpu', weights_only=True)
    if ckpt.get('format') != FORMAT:
        raise ValueError('Not a DualView weights file.')
    return ckpt


def load_model(name_or_path, device='cpu', token=None):
    """DualView with the frozen DINOv3 backbone and the released trained parameters, in eval mode."""
    ckpt = load_checkpoint(name_or_path)
    cfg = ckpt['config']
    model = DualViewDinoUNet(**{k: cfg[k] for k in MODEL_KEYS}, token=token, gradient_checkpointing=False)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    expected.update(n for n, _ in model.named_buffers() if not n.startswith('dino.backbone.'))
    if set(ckpt['weights']) != expected:
        raise ValueError('The weights do not match the model configuration.')
    incompatible = model.load_state_dict(ckpt['weights'], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(incompatible.unexpected_keys)
    model.use_clahe = cfg['use_clahe']
    return model.to(device).eval()
