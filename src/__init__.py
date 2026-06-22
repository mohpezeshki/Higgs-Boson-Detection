# src/__init__.py
"""
HIGGS Boson Detection with Deep Learning
Reproduces Baldi et al. (2014) using PyTorch
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import utilities (MOST IMPORTANT)
from .utils import (
    set_seed,
    get_class_weights,
    print_basic_info,
    statistical_summary,
)

from .plotting import DataPlotter
from .data import HIGGSDataset
from .evaluate import HiggsEvaluator

__all__ = [
    'set_seed',
    'get_class_weights',
    'print_basic_info',
    'statistical_summary',
    'DataPlotter',
    'HIGGSDataset',
    'HiggsEvaluator',
]