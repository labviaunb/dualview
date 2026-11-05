"""Training data: Fundus-AVSeg and RAV readers, paired augmentation of the two views and the cross-validation folds.

Label convention (tensors, ground truth and saved predictions): channels [A, V, BV] = RGB, i.e. artery in magenta, vein in
cyan, crossing in white and vessel without A/V label in blue.
"""
from pathlib import Path
import random
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage import io, measure, morphology
from skimage import color as skcolor
from torch.utils.data import Dataset

from .preprocessing import enhance_image, training_fov

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
# RAV images whose labels are visibly offset from the vessels in the photograph.
RAV_EXCLUDE = ('RAV_376864', 'RAV_493899', 'RAV_495606')


def load_rgb_image(path):
    """RGB float32 image in [0, 1] (H, W, 3); 8-bit images only."""
    img = io.imread(path)
    if img.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit RGB image: {path}')
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[..., :3].astype(np.float32) / 255.0


def read_fundus_avseg_mask(path):
    """Fundus-AVSeg annotation (red = artery, blue = vein, green = crossing, white = vessel of uncertain class) ->
    [A, V, BV] (3 x H x W, 0/1). A crossing is A = V = 1, and a vessel of uncertain class is only BV = 1."""
    rgb = load_rgb_image(path)
    r, g, b = rgb[..., 0] > .5, rgb[..., 1] > .5, rgb[..., 2] > .5
    a, v = r & ~g & ~b, ~r & ~g & b
    cross, white = ~r & g & ~b, r & g & b
    ambiguous = (rgb.max(-1) > .5) & ~(a | v | cross | white)
    unknown = white | ambiguous
    return np.stack((a | cross, v | cross, a | v | cross | unknown)).astype(np.float32)


def read_avbv_mask(path):
    """Mask already in the [A, V, BV] = RGB convention (e.g. converted RAV) -> 3 x H x W (0/1)."""
    return (load_rgb_image(path) > .5).transpose(2, 0, 1).astype(np.float32)


def rav_fov(img, corner=32):
    """FOV of a RAV photograph. Many RAV images are dark, so the threshold is applied to the maximum RGB channel, above the
    background measured in the four corners (max(8/255, 1.5 x their 99th percentile)). Largest component, holes filled,
    eroded by 3 pixels."""
    brightest = img.max(axis=2)
    background = np.concatenate([brightest[:corner, :corner].ravel(), brightest[:corner, -corner:].ravel(),
                                 brightest[-corner:, :corner].ravel(), brightest[-corner:, -corner:].ravel()])
    labels = measure.label(brightest > max(8 / 255, 1.5 * float(np.percentile(background, 99))))
    props = measure.regionprops(labels)
    if not props:
        raise ValueError('Empty RAV FOV')
    fov = ndimage.binary_fill_holes(labels == max(props, key=lambda p: p.area).label)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return morphology.binary_erosion(fov, morphology.disk(3)).astype(np.float32)


def rav_to_avbv(path):
    """RAV label (R = artery, B = vein, G = vessel of unknown class, crossings in R and B) -> H x W x [A, V, BV]."""
    rgb = load_rgb_image(path) > .5
    a, unknown, v = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack((a, v, a | v | unknown), -1).astype(np.float32)


def prepare_rav(source, target, size=1280):
    """Convert RAV once to ``target/images`` and ``target/annotation`` (RGB = [A, V, BV]) at size x size.

    RAV images are already cropped to their FOV and square, so resizing the whole image is the same geometry as the FOV
    square used for the other datasets. Image: bilinear. Mask: bilinear per channel and threshold 0.5.
    """
    source, target = Path(source), Path(target)
    images, masks = target / 'images', target / 'annotation'
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (source / 'rgb').glob('*.png') if p.stem not in RAV_EXCLUDE)
    for p in files:
        if (images / p.name).exists() and (masks / p.name).exists():
            continue
        rgb = np.asarray(Image.open(p).convert('RGB'))
        avbv = rav_to_avbv(source / 'av_masks' / p.name)
        if rgb.shape[0] != rgb.shape[1] or avbv.shape[:2] != rgb.shape[:2]:
            raise ValueError(f'RAV {p.name}: expected square image and mask of the same size')
        big = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = ((cv2.resize(avbv, (size, size), interpolation=cv2.INTER_LINEAR) > .5) * 255).astype(np.uint8)
        for array, folder in ((big, images), (mask, masks)):
            tmp = folder / (p.stem + '.tmp.png')
            Image.fromarray(array).save(tmp)
            tmp.replace(folder / p.name)
    return len(files)


def pair_files(images_dir, masks_dir):
    """Pairs (image, mask) matched by file name; mask names may end in _avmanual, _mask or _annotation."""
    images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    masks = {}
    for p in Path(masks_dir).iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem.lower() in masks:
                raise ValueError(f'Two masks with the same name: {p.stem}')
            masks[p.stem.lower()] = p
    pairs, used = [], set()
    for p in images:
        stem = p.stem.lower()
        candidates = [masks[k] for k in (stem, stem + '_avmanual', stem + '_mask', stem + '_annotation') if k in masks]
        if len(candidates) != 1:
            raise ValueError(f'{p.name}: expected one matching mask, found {candidates}')
        if candidates[0] in used:
            raise ValueError(f'Mask used twice: {candidates[0]}')
        used.add(candidates[0])
        pairs.append((str(p), str(candidates[0])))
    if not pairs:
        raise ValueError(f'No images in {images_dir}')
    return pairs


class RetinalPairs(Dataset):
    """Natural image (CHW, [0, 1]), ground truth [A, V, BV], FOV and id, all at ``full_size``."""
    read_mask = staticmethod(read_fundus_avseg_mask)
    fov_mask = staticmethod(training_fov)

    def __init__(self, images_dir, masks_dir, name='avseg', full_size=(1280, 1280)):
        self.pairs = pair_files(images_dir, masks_dir)
        self.name, self.full_size = name, tuple(full_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ip, mp = self.pairs[index]
        img, gt = load_rgb_image(ip), self.read_mask(mp)
        if gt.shape[-2:] != img.shape[:2]:
            raise ValueError(f'Image and mask sizes differ: {ip}, {mp}')
        fov = self.fov_mask(img)
        h, w = self.full_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            gt = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in gt])
            fov = cv2.resize(fov, (w, h), interpolation=cv2.INTER_NEAREST)
        return {'image': img.transpose(2, 0, 1).copy(), 'mask': gt.copy(), 'fov_mask': fov[None].copy(),
                'id': self.name + ':' + Path(ip).stem}


class RAVPairs(RetinalPairs):
    """Converted RAV (see ``prepare_rav``): masks already in [A, V, BV] and the RAV FOV."""
    read_mask = staticmethod(read_avbv_mask)
    fov_mask = staticmethod(rav_fov)


class PairedTransform:
    """Color jitter (training only), then the enhanced view of the whole image, then the same rotation and flip for both
    views, the ground truth and the FOV (bilinear for images, nearest for masks)."""

    def __init__(self, training=False, use_clahe=True):
        self.training, self.use_clahe = training, use_clahe

    def __call__(self, sample):
        d = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        rgb = d['image'].transpose(1, 2, 0)
        if self.training and random.random() < .5:
            hsv = skcolor.rgb2hsv(rgb)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-.02, .02)) % 1
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(.9, 1.1), 0, 1)
            hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(.9, 1.1), 0, 1)
            rgb = skcolor.hsv2rgb(hsv).astype(np.float32)
        d['image'] = rgb.transpose(2, 0, 1).copy()
        d['enhanced'] = enhance_image(rgb, d['fov_mask'][0], self.use_clahe).transpose(2, 0, 1).copy()
        keys = ('image', 'enhanced', 'mask', 'fov_mask')
        if self.training:
            h, w = d['image'].shape[-2:]
            if random.random() < .8:
                transform = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), random.uniform(-180, 180), 1.)
                for key in keys:
                    interpolation = cv2.INTER_LINEAR if key in ('image', 'enhanced') else cv2.INTER_NEAREST
                    d[key] = np.stack([cv2.warpAffine(x, transform, (w, h), flags=interpolation,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) for x in d[key]])
            if random.random() < .5:
                for key in keys:
                    d[key] = d[key][:, :, ::-1]
        for key in keys:
            d[key] = torch.from_numpy(np.ascontiguousarray(d[key], dtype=np.float32))
        return d


class TransformedPairs(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, list(map(int, indices)), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.transform(self.dataset[self.indices[i]])


def stratified_folds(n_items, n_splits, seed):
    """Round-robin assignment over a seeded permutation: [(train indices, validation indices)] for each fold."""
    rng = np.random.default_rng(seed)
    val = [[] for _ in range(n_splits)]
    for j, idx in enumerate(rng.permutation(np.arange(n_items))):
        val[j % n_splits].append(int(idx))
    all_idx = set(range(n_items))
    return [(sorted(all_idx - set(v)), sorted(v)) for v in val]
