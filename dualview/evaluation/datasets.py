"""Test images of the paper and their references.

Expected layout of ``--data-root`` (the datasets as distributed by their authors):

    RITE/test/images/01_test.tif ... 20_test.tif
    HRF/images/01_dr.JPG, 01_g.jpg, 01_h.jpg ... 05_h.jpg       (any extension; only 01 to 05 are test images)
    LES-AV/images/12.png ... 586.png                         (the 22 images)
    Fundus-AVSeg/images/*.png and Fundus-AVSeg/annotations/*.png   (1280 x 1280)

References and FOV masks of RITE, HRF and LES-AV are the official ones of the RRWNet evaluation data (see ``assets``).
For Fundus-AVSeg, the reference is the dataset annotation, and the FOV is the one used in training.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..data import pair_files, read_fundus_avseg_mask
from ..preprocessing import training_fov

EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')
HRF_TEST = tuple(f'{i:02d}_{g}' for i in range(1, 6) for g in ('dr', 'g', 'h'))
LESAV = ('12', '15', '33', '34', '37', '42', '49', '53', '90', '111', '119', '224', '240', '275', '279', '304', '318',
         '367', '370', '486', '529', '586')


@dataclass(frozen=True)
class Sample:
    dataset: str
    id: str
    image: Path
    label: Path
    fov: Path | None
    label_format: str  # 'avb': RGB = [A, V, BV] official reference; 'fundus-avseg': dataset annotation


def _images_by_stem(folder):
    return {p.stem.lower(): p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in EXTENSIONS}


def external_samples(data_root, references, dataset, cache=None):
    data_root, references = Path(data_root), Path(references)
    if dataset == 'rite':
        return [Sample('rite', f'{i:02d}', data_root / f'RITE/test/images/{i:02d}_test.tif',
                       references / f'RITE-test/gt_hu/{i:02d}.png', references / f'RITE-test/masks/{i:02d}.png', 'avb')
                for i in range(1, 21)]
    if dataset == 'hrf':
        images = _images_by_stem(data_root / 'HRF/images')
        return [Sample('hrf', i, images[i], references / f'HRF/gt_chen/{i}.png', references / f'HRF/masks/{i}.png', 'avb')
                for i in HRF_TEST]
    if dataset == 'lesav':
        images = _images_by_stem(data_root / 'LES-AV/images')
        out = []
        for i in LESAV:
            label, image = references / f'LES-AV/gt_orlando/{i}.png', images[i]
            with Image.open(label) as g, Image.open(image) as im:
                label_size, image_size = g.size, im.size
            if image_size != label_size:
                # Image 34 is a 2196 x 1958 photograph whose reference is on the 1620 x 1444 grid (same aspect ratio):
                # the photograph is area-reduced to the reference grid once.
                if abs(image_size[0] / image_size[1] - label_size[0] / label_size[1]) > 1e-3:
                    raise ValueError(f'LES-AV {i}: aspect ratio of photograph and reference differ')
                cache = Path(cache or data_root / 'LES-AV/resized')
                resized = cache / f'{i}.png'
                if not resized.exists():
                    cache.mkdir(parents=True, exist_ok=True)
                    rgb = np.asarray(Image.open(image).convert('RGB'))
                    Image.fromarray(cv2.resize(rgb, label_size, interpolation=cv2.INTER_AREA)).save(resized)
                image = resized
            out.append(Sample('lesav', i, image, label, references / f'LES-AV/masks/{i}.png', 'avb'))
        return out
    raise ValueError(dataset)


def fundus_avseg_samples(data_root, ids=None):
    pairs = pair_files(Path(data_root) / 'Fundus-AVSeg/images', Path(data_root) / 'Fundus-AVSeg/annotations')
    samples = [Sample('avseg', Path(i).stem, Path(i), Path(m), None, 'fundus-avseg') for i, m in pairs]
    return [s for s in samples if ids is None or s.id in set(ids)]


def read_sample(s):
    """uint8 RGB image, boolean reference (H, W, 3) = [A, V, BV] and boolean FOV, all on the evaluation grid."""
    rgb = np.asarray(Image.open(s.image).convert('RGB'))
    if s.label_format == 'avb':
        gt = np.asarray(Image.open(s.label).convert('RGB')) > 127
    else:
        gt = read_fundus_avseg_mask(s.label).transpose(1, 2, 0) > .5
    if s.fov:
        arr = np.asarray(Image.open(s.fov))
        if arr.ndim == 3:
            arr = arr[..., :3].mean(-1)
        fov = arr > .5
    else:
        fov = training_fov(rgb.astype(np.float32) / 255.) > .5
    if gt.shape != rgb.shape or fov.shape != rgb.shape[:2]:
        raise ValueError(f'{s.dataset}:{s.id}: image, reference and FOV sizes differ')
    return rgb, gt, fov
