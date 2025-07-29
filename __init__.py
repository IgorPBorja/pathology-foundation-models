"""
Pathology Foundation Models

A Python package for interfacing with foundation models for histopathology image analysis.
"""

__version__ = "0.1.0"
__author__ = "Igor Borja"
__email__ = "igorpradoborja@gmail.com"

# Import key modules for easier access
from . import dataset
from . import inference

__all__ = [
    "dataset",
    "inference",
]
