"""Artery/vein segmentation of fundus photographs with the released DualView models.

    python -m dualview.predict photo1.png photo2.jpg --out results/ [--model fundus-avseg | fundus-avseg-rav]

For each photograph, three files are written to ``--out``:
- ``<name>_prob.png``: probabilities as RGB = [artery, vein, vessel] (uint8, native resolution);
- ``<name>_mask.png``: maps binarized at 0.5 (artery in magenta, vein in cyan, crossing in white, vessel without A/V class
  in blue);
- ``<name>_overlay.jpg``: the mask over the photograph.
The prediction is the mean of the four fold models of the chosen training set.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image

from .preprocessing import load_image, model_inputs, to_native
from .weights import ENSEMBLES, load_model


def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class DualViewPredictor:
    """Mean probability of one or more DualView models. ``models``: an ensemble name, a model name or a list of them."""

    def __init__(self, models='fundus-avseg', device=None, token=None):
        names = ENSEMBLES.get(models, [models]) if isinstance(models, str) else list(models)
        self.device = torch.device(device) if device else default_device()
        self.models = [load_model(n, self.device, token) for n in names]

    @torch.inference_mode()
    def __call__(self, rgb):
        """``rgb``: uint8 (H, W, 3). Returns the probabilities (H, W, 3) = [A, V, BV] and the camera aperture (H, W)."""
        natural, enhanced, _, geo, aperture = model_inputs(rgb, self.models[0].use_clahe)
        x = [torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device) for a in (natural, enhanced)]
        autocast = (torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                    if self.device.type == 'cuda' else nullcontext())
        probs = []
        for model in self.models:
            with autocast:
                probs.append(model(*x).sigmoid()[0].float().cpu().numpy().transpose(1, 2, 0))
        return to_native(np.mean(probs, 0), geo) * aperture[..., None], aperture


def save_outputs(rgb, prob, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(prob, 0, 1) * 255 + .5).astype(np.uint8)).save(out_dir / f'{name}_prob.png')
    mask = ((prob > .5) * 255).astype(np.uint8)  # [A, V, BV]: magenta A, cyan V, white crossing, blue unlabeled vessel
    Image.fromarray(mask).save(out_dir / f'{name}_mask.png')
    overlay = rgb.copy()
    vessel = mask.any(-1)
    overlay[vessel] = (.35 * rgb[vessel] + .65 * mask[vessel]).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f'{name}_overlay.jpg', quality=92)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=Path('dualview_results'))
    parser.add_argument('--model', default='fundus-avseg',
                        help='fundus-avseg (default), fundus-avseg-rav, a single model name or a local .pt file')
    parser.add_argument('--device', default=None)
    parser.add_argument('--hf-token', default=None, help='Hugging Face token with access to DINOv3 (or set HF_TOKEN)')
    args = parser.parse_args(argv)
    predictor = DualViewPredictor(args.model, args.device, args.hf_token)
    for path in args.images:
        rgb = load_image(path)
        prob, _ = predictor(rgb)
        save_outputs(rgb, prob, args.out, path.stem)
        print(f'{path.name} -> {args.out}/{path.stem}_*.png', flush=True)


if __name__ == '__main__':
    main()
