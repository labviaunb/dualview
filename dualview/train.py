"""Train DualView with 4-fold cross-validation on Fundus-AVSeg, optionally adding RAV to the training set of every fold.

    python -m dualview.train --images Fundus-AVSeg/images_1280 --annotations Fundus-AVSeg/annotation_1280 --out runs/avseg
    python -m dualview.train ... --rav-source RAV/dataset --rav-dir RAV_1280 --out runs/avseg_rav

Defaults reproduce the paper:
- Model: DINOv3 ViT-L/16 + LoRA (rank 16) on the natural view, CNN on the enhanced view, gated fusion with local
  cross-attention at 1/8 and 1/16.
- Loss: masked BCE + 0.1 x soft-clDice on the artery and vein maps.
- Optimizer: AdamW, learning rate 1e-3 (1e-4 for LoRA, whose adapters stay fixed in the first epoch), weight decay 1e-2,
  cosine schedule.
- Main phase: up to 500 epochs, validation every 25 epochs and early stopping after 50 epochs without improvement of the
  validation macro Dice.
- Fine-tuning: up to 100 epochs at 5e-5 from the best main checkpoint, validation every 10 epochs and patience 30.
- With RAV, every epoch count is multiplied by n_avseg / (n_avseg + n_rav), so that the number of optimization steps is the
  same as without RAV. Validation and model selection always use only the Fundus-AVSeg images of the fold.

Each fold can be resumed. At the end of each fold, the selected weights are written as
``<out>/dualview_<name>_seed<seed>_fold<fold>.pt``, in the format read by ``dualview.predict``.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import RAVPairs, RetinalPairs, PairedTransform, TransformedPairs, prepare_rav, stratified_folds
from .losses import DualViewLoss, supervision_masks
from .model import DualViewDinoUNet

CHANNELS = ('A', 'V', 'BV')
DINO_REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'  # revision of facebook/dinov3-vitl16-pretrain-lvd1689m used


@dataclass
class Config:
    # model
    channels: tuple = (32, 64, 128, 256, 512)
    fusion_dim: int = 64
    fusion: str = 'gated_local'
    fusion_stages: tuple = (1, 2, 3, 4)
    attention_stages: tuple = (3, 4)
    dino_view: str = 'natural'
    model_name: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    revision: str = DINO_REVISION
    stage_indexes: tuple = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = .05
    lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj')
    use_clahe: bool = True
    # loss
    cldice_weight: float = .1
    cldice_iterations: int = 10
    # optimization
    learning_rate: float = 1e-3
    lora_lr_ratio: float = .1
    lora_warmup_epochs: int = 1
    weight_decay: float = 1e-2
    max_epochs: int = 500
    val_interval: int = 25
    stop_patience: int = 50
    ft_epochs: int = 100
    ft_lr: float = 5e-5
    ft_val_interval: int = 10
    ft_stop_patience: int = 30
    batch_size: int = 1
    threshold: float = .5
    # data
    n_folds: int = 4
    seed: int = 42
    split_seed: int = 42
    full_size: tuple = (1280, 1280)
    with_rav: bool = False
    extra: dict = field(default_factory=dict)

    def scale_epochs(self, factor):
        """Keep the number of optimization steps when the training set grows by 1 / factor (as in the paper)."""
        self.max_epochs = round(500 * factor)
        self.val_interval = round(25 * factor)
        self.stop_patience = 2 * round(25 * factor)
        self.ft_epochs = round(100 * factor)
        self.ft_val_interval = round(10 * factor)
        self.ft_stop_patience = 3 * round(10 * factor)


def model_kwargs(cfg):
    return dict(channels=cfg.channels, fusion_dim=cfg.fusion_dim, fusion=cfg.fusion, fusion_stages=cfg.fusion_stages,
                attention_stages=cfg.attention_stages, dino_view=cfg.dino_view, model_name=cfg.model_name,
                revision=cfg.revision, stage_indexes=cfg.stage_indexes, lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def compute_metrics(prob, gt, fov, threshold=.5):
    """Per-image Dice, sensitivity, specificity and precision of [A, V, BV] inside the supervision masks."""
    valid = supervision_masks(gt, fov) > .5
    pred, truth = prob > threshold, gt > .5
    result = {}
    for c, name in enumerate(CHANNELS):
        v = valid[:, c]
        pc, gc = pred[:, c][v], truth[:, c][v]
        tp, fp = int((pc & gc).sum()), int((pc & ~gc).sum())
        fn, tn = int((~pc & gc).sum()), int((~pc & ~gc).sum())
        result[f'{name}_dice'] = 2 * tp / max(2 * tp + fp + fn, 1) if tp + fn else float('nan')
        result[f'{name}_sens'] = tp / (tp + fn) if tp + fn else float('nan')
        result[f'{name}_spec'] = tn / (tn + fp) if tn + fp else float('nan')
        result[f'{name}_prec'] = tp / (tp + fp) if tp + fp else 0.
    return result


def mean_metrics(rows):
    keys = [k for k in rows[0] if k != 'id']
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval='')
            writer.writeheader()
            writer.writerows(rows)


def compact_weights(model):
    """Trained parameters and non-backbone buffers (the frozen DINOv3 backbone is reloaded from the Hub)."""
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    state.update({n: b.detach().cpu().clone() for n, b in model.named_buffers() if not n.startswith('dino.backbone.')})
    return state


def restore_weights(model, weights):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    expected.update(n for n, _ in model.named_buffers() if not n.startswith('dino.backbone.'))
    if set(weights) != expected:
        raise ValueError('Checkpoint does not match the model configuration')
    incompatible = model.load_state_dict(weights, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(incompatible.unexpected_keys)


def atomic_save(payload, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, cfg, avseg, rav, out_dir, device=None, num_workers=8, token=None):
        self.cfg, self.avseg, self.rav, self.out = cfg, avseg, rav, Path(out_dir)
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.amp = self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported() else torch.float16
        self.num_workers, self.token = num_workers, token
        self.criterion = DualViewLoss(cfg.cldice_weight, (0, 1), cfg.cldice_iterations)
        self.splits = stratified_folds(len(avseg), cfg.n_folds, cfg.split_seed)
        self.pool = ConcatDataset([avseg, rav]) if rav is not None else avseg
        self.rav_indices = list(range(len(avseg), len(avseg) + len(rav))) if rav is not None else []
        record = dict(config=asdict(cfg), avseg=[Path(p[0]).name for p in avseg.pairs],
                      rav=[Path(p[0]).name for p in rav.pairs] if rav is not None else [])
        self.signature = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = self.out / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()).get('signature') != self.signature:
            raise ValueError(f'{self.out} holds a run with another configuration or dataset; use another --out')
        manifest.write_text(json.dumps(dict(signature=self.signature, splits=self.splits, **record), indent=2,
                                       default=str))

    def autocast(self):
        return torch.autocast('cuda', dtype=self.amp_dtype) if self.amp else nullcontext()

    def build_model(self):
        return DualViewDinoUNet(**model_kwargs(self.cfg), token=self.token, gradient_checkpointing=True).to(self.device)

    def loaders(self, train_indices, val_indices, fold):
        generator = torch.Generator().manual_seed(self.cfg.seed + fold)
        workers = self.num_workers
        train = TransformedPairs(self.pool, list(train_indices) + self.rav_indices, PairedTransform(True, self.cfg.use_clahe))
        val = TransformedPairs(self.avseg, val_indices, PairedTransform(False, self.cfg.use_clahe))
        common = dict(pin_memory=self.device.type == 'cuda', persistent_workers=workers > 0,
                      prefetch_factor=2 if workers > 0 else None)
        train_loader = DataLoader(train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=workers,
                                  generator=generator, worker_init_fn=seed_worker, **common)
        val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=min(2, workers), **common)
        return train_loader, val_loader, generator

    def train_epoch(self, model, optimizer, scaler, loader, adapt_lora):
        model.train()
        if model.dino is not None:
            model.dino.set_adaptation(adapt_lora)
        total = 0.
        for data in loader:
            natural, enhanced, gt, fov = [data[k].to(self.device, non_blocking=True)
                                          for k in ('image', 'enhanced', 'mask', 'fov_mask')]
            optimizer.zero_grad(set_to_none=True)
            with self.autocast():
                loss = self.criterion(model(natural, enhanced), gt, fov)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        rows, total = [], 0.
        for data in loader:
            natural, enhanced = data['image'].to(self.device), data['enhanced'].to(self.device)
            gt, fov = data['mask'].to(self.device), data['fov_mask'].to(self.device)
            with self.autocast():
                logits = model(natural, enhanced).float()
            total += self.criterion(logits, gt, fov).item()
            rows.append({'id': data['id'][0], **compute_metrics(logits.sigmoid(), gt, fov, self.cfg.threshold)})
        metrics = mean_metrics(rows)
        macro = float(np.mean([metrics[f'{c}_dice'] for c in CHANNELS]))
        return macro, total / max(len(loader), 1), metrics, rows

    def run_phase(self, model, loaders, fold, phase, epochs, lr, interval, patience, initial_best=None):
        train_loader, val_loader, generator = loaders
        folder = self.out / f'fold_{fold}'
        folder.mkdir(exist_ok=True)
        best_path, last_path = folder / f'{phase}_best.pt', folder / f'{phase}_last.pt'
        optimizer = torch.optim.AdamW(model.optimizer_groups(lr, self.cfg.lora_lr_ratio, self.cfg.weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=0.)
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp and self.amp_dtype == torch.float16)
        start, best, stale, history = 0, float('-inf') if initial_best is None else initial_best, 0, []
        if last_path.exists():
            ckpt = torch.load(last_path, map_location='cpu', weights_only=False)
            if ckpt['signature'] != self.signature:
                raise ValueError(f'{last_path} belongs to another run')
            restore_weights(model, ckpt['weights'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            random.setstate(ckpt['rng']['python'])
            np.random.set_state(ckpt['rng']['numpy'])
            torch.set_rng_state(ckpt['rng']['torch'])
            generator.set_state(ckpt['rng']['loader'])
            if ckpt['rng']['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
            start, best, stale, history = ckpt['epoch'], ckpt['best'], ckpt['stale'], ckpt['history']
            if ckpt['completed']:
                restore_weights(model, torch.load(best_path, map_location='cpu', weights_only=False)['weights'])
                return best, best_path
        elif initial_best is not None:
            atomic_save({'signature': self.signature, 'weights': compact_weights(model), 'score': best, 'epoch': 0},
                        best_path)
        for epoch in range(start, epochs):
            t0 = time.time()
            adapt = phase == 'ft' or epoch >= self.cfg.lora_warmup_epochs
            train_loss = self.train_epoch(model, optimizer, scaler, train_loader, adapt)
            scheduler.step()
            if not (epoch == 0 or (epoch + 1) % interval == 0 or epoch + 1 == epochs):
                print(f'fold {fold} {phase} epoch {epoch + 1}: loss {train_loss:.4f} ({time.time() - t0:.0f} s)', flush=True)
                continue
            score, val_loss, metrics, _ = self.evaluate(model, val_loader)
            if score > best:
                best, stale = score, 0
                atomic_save({'signature': self.signature, 'weights': compact_weights(model), 'score': score,
                             'epoch': epoch + 1}, best_path)
            else:
                stale += epoch + 1 - (history[-1]['epoch'] if history else 0)
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'macro_dice': score,
                            **metrics})
            completed = stale >= patience or epoch + 1 == epochs
            rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                   'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   'loader': generator.get_state()}
            atomic_save({'signature': self.signature, 'weights': compact_weights(model),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'scaler': scaler.state_dict(), 'rng': rng, 'epoch': epoch + 1, 'best': best, 'stale': stale,
                         'history': history, 'completed': completed}, last_path)
            write_csv(folder / f'{phase}_history.csv', history)
            print(f'fold {fold} {phase} epoch {epoch + 1}: loss {train_loss:.4f}, validation macro Dice {score:.4f} '
                  f'(A {metrics["A_dice"]:.4f}, V {metrics["V_dice"]:.4f}, BV {metrics["BV_dice"]:.4f}), '
                  f'best {best:.4f} ({time.time() - t0:.0f} s)', flush=True)
            if completed:
                break
        restore_weights(model, torch.load(best_path, map_location='cpu', weights_only=False)['weights'])
        return best, best_path

    def run(self, folds=None, name='fundus-avseg'):
        cfg = self.cfg
        for fold, (train_indices, val_indices) in enumerate(self.splits):
            if folds is not None and fold not in folds:
                continue
            result_path = self.out / f'fold_{fold}' / 'fold_result.json'
            if result_path.exists():
                print(f'fold {fold}: already done', flush=True)
                continue
            seed_all(cfg.seed + fold)
            model = self.build_model()
            loaders = self.loaders(train_indices, val_indices, fold)
            score, best_path = self.run_phase(model, loaders, fold, 'main', cfg.max_epochs, cfg.learning_rate,
                                              cfg.val_interval, cfg.stop_patience)
            if cfg.ft_epochs:
                score, best_path = self.run_phase(model, loaders, fold, 'ft', cfg.ft_epochs, cfg.ft_lr,
                                                  cfg.ft_val_interval, cfg.ft_stop_patience, initial_best=score)
            score, _, metrics, rows = self.evaluate(model, loaders[1])
            write_csv(self.out / f'fold_{fold}' / 'validation_metrics.csv', rows)
            validation = [Path(self.avseg.pairs[i][0]).stem for i in val_indices]
            release = dict(format='dualview-weights-v1', config={**model_kwargs(cfg), 'use_clahe': cfg.use_clahe},
                           weights=compact_weights(model),
                           provenance=dict(training_set=name, seed=cfg.seed, fold=fold, validation_images=validation,
                                           validation_macro_dice=score, selected_checkpoint=best_path.name))
            torch.save(release, self.out / f'dualview_{name}_seed{cfg.seed}_fold{fold}.pt')
            result_path.write_text(json.dumps({'fold': fold, 'macro_dice': score, 'checkpoint': best_path.name,
                                               **metrics}, indent=2))
            print(f'fold {fold}: validation macro Dice {score:.4f}', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, required=True, help='Fundus-AVSeg images at 1280 x 1280')
    parser.add_argument('--annotations', type=Path, required=True, help='Fundus-AVSeg annotations at 1280 x 1280')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rav-source', type=Path, default=None, help='RAV as downloaded (folder with rgb/ and av_masks/)')
    parser.add_argument('--rav-dir', type=Path, default=None, help='where the converted RAV is written and read')
    parser.add_argument('--seed', type=int, default=42, help='training seed (initialization, augmentation, order)')
    parser.add_argument('--split-seed', type=int, default=42, help='seed of the fold split (42 in the paper)')
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default=None)
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    args = parser.parse_args(argv)
    cfg = Config(seed=args.seed, split_seed=args.split_seed)
    avseg = RetinalPairs(args.images, args.annotations, 'avseg', cfg.full_size)
    rav = None
    if args.rav_dir:
        if args.rav_source:
            prepare_rav(args.rav_source, args.rav_dir, cfg.full_size[0])
        rav = RAVPairs(args.rav_dir / 'images', args.rav_dir / 'annotation', 'rav', cfg.full_size)
        n_train = len(avseg) - len(avseg) // cfg.n_folds
        cfg.with_rav = True
        cfg.scale_epochs(n_train / (n_train + len(rav)))
    name = 'fundus-avseg-rav' if rav is not None else 'fundus-avseg'
    print(f'{len(avseg)} Fundus-AVSeg images' + (f' and {len(rav)} RAV images' if rav else '')
          + f' | main phase {cfg.max_epochs} epochs, fine-tuning {cfg.ft_epochs} epochs', flush=True)
    Trainer(cfg, avseg, rav, args.out, args.device, args.num_workers, args.hf_token).run(args.folds, name)


if __name__ == '__main__':
    main()
