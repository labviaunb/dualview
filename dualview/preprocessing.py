"""Input views and geometry of DualView.

- ``natural``: the RGB photograph in [0, 1] (DINOv3 normalizes it internally with ImageNet statistics).
- ``enhanced``: the view of the CNN, as in RRWNet: the region outside the field of view (FOV) is filled with a 1.15x zoom of
  the image, CLAHE is applied to the lightness channel, a Gaussian blur (sigma = 10 pixels) is subtracted, and the result is
  divided by its standard deviation and rescaled to [0, 1] inside the FOV.
- Geometry: the square around the FOV is cropped (zero padding if needed) and resized to 1280 x 1280; predictions are
  resized back to the native grid.
- FOV: during training on Fundus-AVSeg, the largest region brighter than 0.1 in mean intensity (``training_fov``). For any
  other photograph, the camera aperture (``aperture_fov``), which also covers dark peripheral retina in vignetted images.
"""
import warnings

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology

INPUT_SIZE = 1280


def _quiet(fn, *args):
    # skimage >= 0.26 deprecates the binary_* morphology functions. They are kept here because they define the masks used in
    # the paper; the warning is silenced.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return fn(*args)


def load_image(path):
    """RGB uint8 array (H, W, 3) from any image file readable by Pillow."""
    return np.asarray(Image.open(path).convert('RGB'))


def training_fov(natural, threshold=.1):
    """FOV used for Fundus-AVSeg training: largest region with mean intensity > threshold, holes filled, eroded by 3 px."""
    labels = measure.label(natural.mean(axis=2) > threshold)
    props = measure.regionprops(labels)
    if not props:
        raise ValueError('Empty FOV.')
    fov = ndimage.binary_fill_holes(labels == max(props, key=lambda p: p.area).label)
    return _quiet(morphology.binary_erosion, fov, morphology.disk(3)).astype(np.float32)


def aperture_fov(rgb, margin=4, open_radius=2, close_radius=5):
    """Camera aperture of a uint8 RGB photograph (bool, not eroded).

    Pixels whose maximum RGB channel exceeds the background (99th percentile of the four corners plus ``margin`` gray
    levels), followed by a morphological opening, the largest connected component, a closing and hole filling.
    """
    mx = np.asarray(rgb)[..., :3].max(-1).astype(np.int16)
    h, w = mx.shape
    c = max(8, min(h, w) // 40)
    corners = np.concatenate([mx[:c, :c].ravel(), mx[:c, -c:].ravel(), mx[-c:, :c].ravel(), mx[-c:, -c:].ravel()])
    background = float(np.percentile(corners, 99))
    if background > 60:
        raise ValueError('The image corners are not dark: the aperture rule assumes a dark surround.')
    bright = _quiet(morphology.binary_opening, mx > background + margin, morphology.disk(open_radius))
    labels = measure.label(bright)
    props = measure.regionprops(labels)
    if not props:
        raise ValueError('Empty aperture.')
    largest = labels == max(props, key=lambda p: p.area).label
    pad = close_radius + 3
    closed = _quiet(morphology.binary_closing, np.pad(largest, pad), morphology.disk(close_radius))[pad:-pad, pad:-pad]
    return ndimage.binary_fill_holes(closed)


def enhance_image(img, fov, use_clahe=True):
    """Enhanced view (float32 in [0, 1]) of a natural image ``img`` (float in [0, 1]) inside ``fov``."""
    h, w = img.shape[:2]
    mask = np.repeat((fov > .5)[..., None], 3, axis=2)
    if not mask.any():
        return np.zeros_like(img, dtype=np.float32)
    zoom = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
        (int(w * 1.15), int(h * 1.15)), Image.Resampling.BICUBIC), dtype=np.float32) / 255.
    y, x = (zoom.shape[0] - h) // 2, (zoom.shape[1] - w) // 2
    composed = np.where(mask, img, zoom[y:y + h, x:x + w])
    if use_clahe:
        lab = cv2.cvtColor((np.clip(composed, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[..., 0])
        composed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.
    sub = (composed - ndimage.gaussian_filter(composed, sigma=(10, 10, 0))) * mask
    sub = sub / max(float(sub[mask].std()), 1e-8)
    lo, hi = float(sub[mask].min()), float(sub[mask].max())
    out = (sub - lo) / max(hi - lo, 1e-8)
    return np.clip(out * mask, 0, 1).astype(np.float32)


def square_geometry(fov):
    """Square around the FOV, with the padding needed when it extends beyond the image."""
    ys, xs = np.where(fov)
    if not len(xs):
        raise ValueError('Empty FOV')
    side = int(max(np.ptp(xs) + 1, np.ptp(ys) + 1))
    x0 = int(np.floor((xs.min() + xs.max() + 1 - side) / 2))
    y0 = int(np.floor((ys.min() + ys.max() + 1 - side) / 2))
    h, w = fov.shape
    left, top = max(-x0, 0), max(-y0, 0)
    right, bottom = max(x0 + side - w, 0), max(y0 + side - h, 0)
    return dict(side=side, x0=x0 + left, y0=y0 + top, left=left, top=top, right=right, bottom=bottom, h=h, w=w)


def to_input(arr, geo, nearest=False):
    """Crop the FOV square and resize it to INPUT_SIZE x INPUT_SIZE."""
    pad = ((geo['top'], geo['bottom']), (geo['left'], geo['right'])) + ((0, 0),) * (arr.ndim - 2)
    arr = np.pad(arr, pad)[geo['y0']:geo['y0'] + geo['side'], geo['x0']:geo['x0'] + geo['side']]
    return cv2.resize(arr.astype(np.float32), (INPUT_SIZE, INPUT_SIZE),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def to_native(prob, geo):
    """Inverse of ``to_input`` for an (INPUT_SIZE, INPUT_SIZE, 3) probability map."""
    crop = cv2.resize(prob, (geo['side'], geo['side']), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((geo['h'] + geo['top'] + geo['bottom'], geo['w'] + geo['left'] + geo['right'], 3), np.float32)
    canvas[geo['y0']:geo['y0'] + geo['side'], geo['x0']:geo['x0'] + geo['side']] = crop
    return canvas[geo['top']:geo['top'] + geo['h'], geo['left']:geo['left'] + geo['w']]


def model_inputs(rgb, use_clahe=True, erosion=3):
    """Natural and enhanced views at 1280 x 1280, the input FOV and the geometry, for any uint8 RGB photograph."""
    aperture = aperture_fov(rgb)
    geo = square_geometry(aperture)
    natural = to_input(rgb.astype(np.float32) / 255., geo)
    fov = to_input(aperture.astype(np.float32), geo, nearest=True) > .5
    fov = ndimage.binary_erosion(fov, morphology.disk(erosion)).astype(np.float32)
    return natural, enhance_image(natural, fov, use_clahe), fov, geo, aperture
