"""DualView: artery/vein segmentation of fundus photographs with a CNN and a DINOv3 vision transformer."""
from .model import DualViewDinoUNet
from .weights import ENSEMBLES, MODELS, load_model
from .predict import DualViewPredictor

__version__ = '1.0.0'
__all__ = ['DualViewDinoUNet', 'DualViewPredictor', 'ENSEMBLES', 'MODELS', 'load_model']
