"""
Segmentation Tools Package

A comprehensive toolkit for image segmentation labeling and evaluation.
Provides tools for interactive mask creation, editing, and evaluation.
"""

__version__ = "0.1.0"
__author__ = "segmentation_tools"

from .core import MaskManager, Label
from .tools import BrushTool, EraseTool, PolygonTool
from .eval import SegmentationEvaluator

__all__ = [
    "MaskManager",
    "Label", 
    "BrushTool",
    "EraseTool",
    "PolygonTool",
    "SegmentationEvaluator",
]