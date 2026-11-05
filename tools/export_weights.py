"""Export the trained runs of the paper to the released weight files (format ``dualview-weights-v1``).

    python tools/export_weights.py <folder with the training runs> <output folder>

Each output file keeps only the trained parameters and buffers, the model configuration and the provenance of the run (seed,
fold, Fundus-AVSeg validation images and validation score). The checkpoint signature is checked against the run manifest.
"""
from pathlib import Path
import hashlib
import json
import sys

import torch

RUNS = {'fundus-avseg': {42: 'dualview_final_gated_local_04db7164f2', 43: 'dualview_final_gated_local_5278549366',
                         44: 'dualview_final_gated_local_ef24867cf0'},
        'fundus-avseg-rav': {42: 'dualview_final_rav_gated_local_faa7ff1290'}}
LICENSES = {'fundus-avseg': 'MIT', 'fundus-avseg-rav': 'CC BY-NC 4.0'}


def config(m):
    return dict(channels=tuple(m['CNN_CHANNELS']), fusion_dim=m['FUSION_DIM'], fusion=m['FUSION'],
                fusion_stages=tuple(m['FUSION_STAGES']), attention_stages=tuple(m['ATTENTION_STAGES']),
                dino_view=m['DINO_VIEW'], model_name=m['DINO_MODEL_NAME'], revision=m['backbone_revision'],
                stage_indexes=None if m['DINO_STAGE_INDEXES'] is None else tuple(m['DINO_STAGE_INDEXES']),
                lora_rank=m['LORA_RANK'], lora_alpha=m['LORA_ALPHA'], lora_dropout=m['LORA_DROPOUT'],
                lora_targets=tuple(m['LORA_TARGETS']), use_clahe=bool(m['USE_CLAHE']))


def main(runs_dir, out_dir):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for train_set, seeds in RUNS.items():
        for seed, run in seeds.items():
            m = json.loads((runs_dir / run / 'manifest.json').read_text())
            signature = hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()
            for fold in range(4):
                result = json.loads((runs_dir / run / f'fold_{fold}' / 'fold_result.json').read_text())
                ckpt_file = runs_dir / run / f'fold_{fold}' / Path(result['checkpoint']).name
                ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                if ckpt['signature'] != signature:
                    raise ValueError(f'{ckpt_file}: signature does not match the manifest')
                validation = [m['ids'][i].split(':', 1)[1] for i in m['splits'][fold][1]]
                record = dict(format='dualview-weights-v1', config=config(m),
                              weights={k: v.contiguous() for k, v in ckpt['weights'].items()},
                              provenance=dict(training_set=train_set, seed=seed, fold=fold, license=LICENSES[train_set],
                                              validation_images=validation,
                                              validation_macro_dice=float(result['score']),
                                              selected_checkpoint=ckpt_file.name))
                out = out_dir / f'dualview_{train_set}_seed{seed}_fold{fold}.pt'
                torch.save(record, out)
                print(out.name, f'{out.stat().st_size / 1e6:.1f} MB', f"val macro Dice {result['score']:.4f}")


if __name__ == '__main__':
    main(*sys.argv[1:3])
