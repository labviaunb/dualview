"""Reproduce the evaluation of the paper (Tables 1 and 2 and the TFFM protocol).

    python -m dualview.evaluation.evaluate --data-root DATA --assets ASSETS --out results/
    python -m dualview.evaluation.evaluate ... --rrwnet --n-paths 100 --tffm

- DualView: every released fold model of the chosen training sets (default: all 16). On Fundus-AVSeg, each model predicts
  only the validation images of its fold, with the preprocessing used in training. On RITE, LES-AV and HRF, each model is
  applied without adaptation, with the camera-aperture FOV and the 1280 x 1280 geometry. Metrics are averaged over the fold
  models of a seed and then reported as mean and standard deviation over seeds.
- ``--rrwnet``: the official RRWNet weights (trained on RITE for RITE and LES-AV, and on HRF for HRF), predicted and evaluated
  with the same code, as used for the Dice and connectivity of RRWNet in the paper.
- ``--n-paths 100``: connectivity (Inf/Corr). This step runs on the CPU and takes hours on HRF at native resolution.
- ``--tffm``: Fundus-AVSeg in the TFFM protocol, on the 100 images predicted out of fold.

Predictions are cached in ``<out>/predictions``, so an interrupted run resumes where it stopped. Outputs:
``classification.csv`` (Table 1), ``connectivity.csv`` (Table 2), ``tffm_protocol.csv`` and ``per_model.csv``.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import importlib.util
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage import measure

from ..preprocessing import enhance_image, model_inputs, to_native, training_fov
from ..weights import MODELS, load_checkpoint, load_model
from .assets import ensure_assets
from .datasets import external_samples, fundus_avseg_samples, read_sample
from .metrics import evaluate_folder, tffm_protocol

DATASETS = ('avseg', 'rite', 'lesav', 'hrf')
TABLE_KEYS = ['sens_av', 'spec_av', 'acc_av', 'f1_av', 'acc_bv', 'auc_roc_bv', 'A_dice', 'V_dice', 'BV_dice']
TOPOLOGY_KEYS = ['inf_a', 'inf_v', 'corr_a', 'corr_v']


def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def save_prediction(path, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(p, 0, 1) * 255. + .5).clip(0, 255).astype(np.uint8)).save(path)


@torch.inference_mode()
def predict_dualview(model, samples, folder, device):
    autocast = (torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                if device.type == 'cuda' else nullcontext())
    for s in samples:
        out = folder / f'{s.id}.png'
        if out.exists():
            continue
        rgb, _, eval_fov = read_sample(s)
        if s.dataset == 'avseg':
            # Same input as the validation of training: 1280 x 1280 image, training FOV.
            natural = rgb.astype(np.float32) / 255.
            fov = training_fov(natural)
            enhanced, geo = enhance_image(natural, fov, model.use_clahe), None
        else:
            natural, enhanced, _, geo, _ = model_inputs(rgb, model.use_clahe)
        x = [torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0).to(device) for a in (natural, enhanced)]
        with autocast:
            p = model(*x).sigmoid()[0].float().cpu().numpy().transpose(1, 2, 0)
        if s.dataset == 'avseg':
            p = p * (fov[..., None] > .5)
        else:
            p = to_native(p, geo) * eval_fov[..., None]
        save_prediction(out, p)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def predict_rrwnet(assets, samples, weight_dataset, folder, device):
    """Official RRWNet weights. On their own dataset, the official preprocessed inputs; for transfer (LES-AV with the
    RITE weights), the official preprocessing at the training width of the weights, as in the RRWNet code."""
    repo = assets['repo']
    rr = _load_module('rrwnet_model_official', repo / 'model.py')
    prep = _load_module('rrwnet_preprocessing_official', repo / 'preprocessing.py')
    trans = _load_module('rrwnet_transformations_official', repo / 'train/transformations.py')
    model = rr.RRWNet(iterations=5)
    weights = assets['weights_rite'] if weight_dataset == 'rite' else assets['weights_hrf']
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    input_dir = assets['official_inputs'] / ('RITE/test' if weight_dataset == 'rite'
                                             else 'HRF_AVLabel_191219/test_karlsson_w1024')
    for s in samples:
        out = folder / f'{s.id}.png'
        if out.exists():
            continue
        rgb, _, fov = read_sample(s)
        h, w = rgb.shape[:2]
        if s.dataset == weight_dataset:
            enhanced = np.asarray(Image.open(input_dir / 'enhanced' / f'{s.id}.png').convert('RGB')).astype(np.float32) / 255.
            input_mask = np.asarray(Image.open(input_dir / 'enhanced_masks' / f'{s.id}.png')) > .5
        else:
            labels = measure.label(rgb.mean(-1) / 255. > .05)
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            raw_fov = ndimage.binary_fill_holes(labels == counts.argmax())
            width = 565 if weight_dataset == 'rite' else 1024
            height = int(round(h * width / w))
            scaled = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA if w > width else cv2.INTER_LINEAR)
            scaled_fov = cv2.resize(raw_fov.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > .5
            enhanced, input_mask = prep.enhance_image(scaled, scaled_fov, int_format=True)
            enhanced = enhanced.astype(np.float32) / 255.
        padded, pads = trans.pad_images_unet([enhanced], return_paddings=True)
        logits = model(torch.from_numpy(padded[0].transpose(2, 0, 1).copy()).unsqueeze(0).to(device))[-1]
        p = logits.sigmoid()[0].float().cpu().numpy().transpose(1, 2, 0)
        (top, _), (left, _), _ = pads[0]
        p = p[top:top + enhanced.shape[0], left:left + enhanced.shape[1]] * (input_mask > .5)[..., None]
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        save_prediction(out, p * fov[..., None])
    del model


def summarize(per_model, keys, avseg_folds):
    """Mean over the fold models of each seed, then mean and s.d. over seeds."""
    t = per_model[(per_model.dataset != 'avseg') | per_model.fold.isin(avseg_folds) | per_model.fold.isna()]
    keys = [k for k in keys if k in t]
    per_seed = t.groupby(['training_set', 'dataset', 'seed'], dropna=False)[keys].mean()
    mean = per_seed.groupby(['training_set', 'dataset']).mean()
    sd = per_seed.groupby(['training_set', 'dataset']).std()
    out = mean.round(2).astype(str)
    for k in keys:
        has_sd = sd[k].notna()
        out.loc[has_sd, k] = mean.loc[has_sd, k].round(2).astype(str) + ' ± ' + sd.loc[has_sd, k].round(2).astype(str)
    return out.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True, help='folder for the official RRWNet code and data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='*', default=['fundus-avseg', 'fundus-avseg-rav'],
                        help='training sets (fundus-avseg, fundus-avseg-rav) or model names')
    parser.add_argument('--datasets', nargs='*', default=list(DATASETS))
    parser.add_argument('--avseg-folds', type=int, nargs='*', default=[0, 1],
                        help='Fundus-AVSeg folds in Table 1 (0 and 1 in the paper, as for RRWNet retrained)')
    parser.add_argument('--rrwnet', action='store_true', help='also evaluate the official RRWNet weights')
    parser.add_argument('--n-paths', type=int, default=0, help='paths per image and class for Inf/Corr (100 in the paper)')
    parser.add_argument('--tffm', action='store_true', help='Fundus-AVSeg in the TFFM protocol (100 images out of fold)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--hf-token', default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device) if args.device else device_auto()
    assets = ensure_assets(args.assets, baseline=args.rrwnet)
    names = [m for m in MODELS if any(m == x or m.rsplit('_seed', 1)[0] == x for x in args.models)]
    external = {d: external_samples(args.data_root, assets['references'], d) for d in args.datasets if d != 'avseg'}
    avseg = fundus_avseg_samples(args.data_root) if 'avseg' in args.datasets else []
    avseg_folds = sorted(set(args.avseg_folds) | ({0, 1, 2, 3} if args.tffm else set()))
    rows, tffm_rows = [], []
    for name in names:
        provenance = load_checkpoint(name)['provenance']
        training_set, seed, fold = provenance['training_set'], provenance['seed'], provenance['fold']
        targets = dict(external)
        if avseg and fold in avseg_folds:
            targets['avseg'] = [s for s in avseg if s.id in set(provenance['validation_images'])]
        model = None
        for dataset, samples in targets.items():
            folder = args.out / 'predictions' / name / dataset
            if not all((folder / f'{s.id}.png').exists() for s in samples):
                model = model or load_model(name, device, args.hf_token)
                predict_dualview(model, samples, folder, device)
            summary, table = evaluate_folder(samples, folder, assets['repo'], args.n_paths)
            rows.append(dict(model=name, training_set=training_set, seed=seed, fold=fold, dataset=dataset, **summary))
            if dataset == 'avseg' and args.tffm:
                tffm_rows.append(tffm_protocol(samples, folder).assign(training_set=training_set, seed=seed, fold=fold))
            print(f'{name} {dataset}: A/V accuracy {summary["acc_av"]:.2f}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if args.rrwnet:
        for dataset, weight in (('rite', 'rite'), ('lesav', 'rite'), ('hrf', 'hrf')):
            if dataset not in external:
                continue
            folder = args.out / 'predictions' / f'rrwnet_{weight}' / dataset
            predict_rrwnet(assets, external[dataset], weight, folder, device)
            summary, _ = evaluate_folder(external[dataset], folder, assets['repo'], args.n_paths)
            rows.append(dict(model=f'rrwnet_{weight}', training_set=f'RRWNet (official, {weight.upper()})', seed=np.nan,
                             fold=np.nan, dataset=dataset, **summary))
            print(f'RRWNet ({weight}) {dataset}: A/V accuracy {summary["acc_av"]:.2f}', flush=True)
    per_model = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(args.out / 'per_model.csv', index=False)
    summarize(per_model, TABLE_KEYS, args.avseg_folds).to_csv(args.out / 'classification.csv', index=False)
    print(summarize(per_model, TABLE_KEYS, args.avseg_folds).to_string(index=False))
    if args.n_paths:
        summarize(per_model, TOPOLOGY_KEYS, args.avseg_folds).to_csv(args.out / 'connectivity.csv', index=False)
    if tffm_rows:
        t = pd.concat(tffm_rows)
        per_seed = t.groupby(['training_set', 'seed'])[['A_dice', 'V_dice', 'combined_dice', 'combined_hd95']].mean()
        per_seed.groupby('training_set').agg(['mean', 'std']).round(2).to_csv(args.out / 'tffm_protocol.csv')


if __name__ == '__main__':
    main()
