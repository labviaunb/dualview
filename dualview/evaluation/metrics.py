"""Metrics of the paper, computed from saved probability maps (RGB PNG = [A, V, BV], uint8).

- A/V classification and vessel segmentation: official RRWNet functions (``eval/test_utils.py``), in the ``All`` and
  ``Intersection`` modes, with pooled counts over the images of a dataset. AUROC/AUC-PR curves accumulate exact score counts,
  which is equivalent to concatenating all pixels without multi-GB arrays.
- Dice: per image, inside the FOV, vessels without A/V label excluded for A and V, crossings positive in both, threshold 0.5.
- Connectivity: official ``eval/topo_metric.py`` (Inf, Corr), with ``n_paths`` pairs per image and class and the same
  pseudo-random pairs for every model.
- TFFM protocol: binary prediction and reference resized to 512 x 512 (nearest neighbor), Dice and HD95 of artery, vein and
  their union.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import random
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.metrics import auc

from .datasets import read_sample

CHANNELS = ('A', 'V', 'BV')
PATH_SEED = 42


def official_functions(repo):
    path = str(Path(repo) / 'eval')
    if path not in sys.path:
        sys.path.insert(0, path)
    from test_utils import compute_classification_metrics, get_vessel_tree_mask
    from topo_metric import topo_metric
    return compute_classification_metrics, get_vessel_tree_mask, topo_metric


def confusion(gt, pred, mask):
    g, p = gt[mask] > .5, pred[mask] > .5
    return np.array([(g & p).sum(), ((~g) & p).sum(), (g & (~p)).sum(), ((~g) & (~p)).sum()], dtype=np.int64)


def conf_metrics(counts):
    tp, fp, fn, tn = map(int, counts)

    def div(a, b):
        return a / b if b else float('nan')
    return dict(sens=div(tp, tp + fn), spec=div(tn, tn + fp), acc=div(tp + tn, tp + tn + fp + fn),
                f1=div(2 * tp, 2 * tp + fp + fn), prec=div(tp, tp + fp))


class ScoreCounts:
    def __init__(self):
        self.counts = defaultdict(lambda: np.zeros(2, np.int64))

    def add(self, gt, pred, mask):
        g, p = gt[mask] > .5, pred[mask]
        for label in (0, 1):
            values, counts = np.unique(p[g == bool(label)], return_counts=True)
            for v, n in zip(values, counts):
                self.counts[float(v)][label] += n

    def result(self):
        keys = sorted(self.counts, reverse=True)
        if not keys:
            return (float('nan'),) * 2
        pairs = np.stack([self.counts[k] for k in keys])
        neg, pos = pairs[:, 0], pairs[:, 1]
        if not pos.sum() or not neg.sum():
            return (float('nan'),) * 2
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        tpr, fpr = np.r_[0., tp / pos.sum()], np.r_[0., fp / neg.sum()]
        precision, recall = tp / (tp + fp), tp / pos.sum()
        return float(auc(fpr, tpr)), float(auc(np.r_[recall[::-1], 0.], np.r_[precision[::-1], 1.]))


def dice_metrics(prob, gt, fov):
    unknown = gt[..., 2] & ~gt[..., 0] & ~gt[..., 1]
    result = {}
    for c, ch in enumerate(CHANNELS):
        valid = fov & (~unknown if c < 2 else True)
        counts = confusion(gt[..., c], prob[..., c], valid)
        scores = conf_metrics(counts)
        result[f'{ch}_dice'] = scores['f1'] if counts[0] + counts[2] else float('nan')
        for k in ('sens', 'spec', 'prec'):
            result[f'{ch}_{k}'] = scores[k]
    return result


def path_seed(sample, channel):
    return int(hashlib.sha256(f'{PATH_SEED}:{sample.dataset}:{sample.id}:{channel}'.encode()).hexdigest()[:8], 16)


class Pool:
    def __init__(self):
        self.av, self.bv = np.zeros(4, np.int64), np.zeros(4, np.int64)
        self.curves = [ScoreCounts() for _ in CHANNELS]

    def add(self, p, g, f, av_mask):
        pred_a = 1 - np.argmax(p[..., :2], axis=-1)
        self.av += confusion(g[..., 0], pred_a, av_mask)
        self.bv += confusion(g[..., 2], p[..., 2], f)
        for c, curve in enumerate(self.curves):
            curve.add(g[..., c], p[..., c], f)

    def result(self):
        result = {}
        for name, count in (('av', self.av), ('bv', self.bv)):
            for k, v in conf_metrics(count).items():
                result[f'{k}_{name}'] = v
        for ch, curve in zip(('a', 'v', 'bv'), self.curves):
            result[f'auc_roc_{ch}'], result[f'auc_pr_{ch}'] = curve.result()
        return result


def evaluate_folder(samples, folder, repo, n_paths=0):
    """Metrics of the predictions saved in ``folder/<id>.png``. Returns (dataset summary, per-image table)."""
    folder = Path(folder)
    classify, vessel_mask, topology = official_functions(repo)
    pools = {k: Pool() for k in ('All', 'Intersection')}
    rows = []
    for s in samples:
        _, gt, fov = read_sample(s)
        p = np.asarray(Image.open(folder / f'{s.id}.png').convert('RGB')).astype(np.float64) / 255.
        if p.shape != gt.shape:
            raise ValueError(f'{s.id}: prediction and reference sizes differ')
        row = dict(id=s.id, dataset=s.dataset, **dice_metrics(p, gt, fov))
        mask = vessel_mask(gt.astype(np.float32))[0] > .5
        # Same seeded tie-breaking as the official code when both A/V probabilities are zero.
        np.random.seed(path_seed(s, 'ties'))
        rand = np.random.random(p.shape[:2])
        original = p.copy() if n_paths else None
        tie = (p[..., 0] == 0) & (p[..., 1] == 0) & mask
        p[tie, 0], p[tie, 1] = rand[tie], 1 - rand[tie]
        for mode, pool in pools.items():
            av_mask = mask if mode == 'All' else mask & (p[..., 2] > .5)
            pool.add(p, gt, fov, av_mask)
            if av_mask.any() and np.unique(gt[..., 0][av_mask]).size == 2:
                values = classify(gt[..., 0], 1 - np.argmax(p[..., :2], -1), av_mask)
                for k, v in zip(('sens_av', 'spec_av', 'acc_av', 'f1_av'), values):
                    row[f'{mode}_{k}'] = float(v)
        if n_paths:
            cache = folder / 'topology' / f'{s.id}_n{n_paths}.json'
            if cache.exists():
                topo = json.loads(cache.read_text())
            else:
                topo = {}
                for c, ch in enumerate(('a', 'v', 'bv')):
                    if not gt[..., c].any():
                        topo[f'inf_{ch}'] = topo[f'corr_{ch}'] = None
                        continue
                    random.seed(path_seed(s, ch))
                    inf, _, corr = topology(gt[..., c], original[..., c], .5, n_paths)
                    topo[f'inf_{ch}'], topo[f'corr_{ch}'] = inf / n_paths, corr / n_paths
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(json.dumps(topo))
            row.update(topo)
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = {k: 100 * v for k, v in pools['Intersection'].result().items()}
    for k in [c for c in table if c.endswith('_dice') or c.startswith(('inf_', 'corr_'))]:
        summary[k] = 100 * float(table[k].mean())
    return summary, table


def _hd95(pred, gt):
    if not pred.any() or not gt.any():
        return float('nan')
    sp, sg = pred & ~ndimage.binary_erosion(pred), gt & ~ndimage.binary_erosion(gt)
    dg, dp = ndimage.distance_transform_edt(~sg), ndimage.distance_transform_edt(~sp)
    return float(max(np.percentile(dg[sp], 95), np.percentile(dp[sg], 95)))


def tffm_protocol(samples, folder, size=512):
    """Fundus-AVSeg in the protocol of TFFM (Ahmed et al., WACVW 2026): per-image Dice and HD95 at 512 x 512."""
    rows = []
    for s in samples:
        _, gt, _ = read_sample(s)
        p = np.asarray(Image.open(Path(folder) / f'{s.id}.png').convert('RGB')).astype(np.float32) / 255.

        def nearest(m):
            return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        g = np.stack([nearest(gt[..., k]) for k in range(2)], -1)
        q = np.stack([nearest(p[..., k] > .5) for k in range(2)], -1)
        row = dict(id=s.id)
        for name, pr, gg in (('A', q[..., 0], g[..., 0]), ('V', q[..., 1], g[..., 1]),
                             ('combined', q[..., 0] | q[..., 1], g[..., 0] | g[..., 1])):
            row[f'{name}_dice'] = 100 * float(2 * (pr & gg).sum() / max(pr.sum() + gg.sum(), 1))
            row[f'{name}_hd95'] = _hd95(pr, gg)
        rows.append(row)
    return pd.DataFrame(rows)
