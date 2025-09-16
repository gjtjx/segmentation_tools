# Segmentation Tools

A comprehensive toolkit for image segmentation labeling and evaluation. This package provides interactive tools for creating, editing, and evaluating segmentation masks with support for multiple annotation tools and comprehensive evaluation metrics.

## Features

### Interactive Annotation Tools
- **Brush Tool**: Paint masks with adjustable brush size
- **Erase Tool**: Remove parts of masks with adjustable eraser size  
- **Polygon Tool**: Create polygonal selections, rectangles, and circles

### Label Management
- Multi-class segmentation support
- Automatic color generation for visualization
- Label persistence and serialization

### Evaluation Metrics
- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Accuracy
- Multi-class evaluation with per-class metrics
- Confusion matrix analysis

### Data Management
- Save/load functionality for masks and labels
- Multiple export formats
- Easy integration with existing workflows

## Installation

```bash
git clone https://github.com/gjtjx/segmentation_tools
cd segmentation_tools
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- Pillow >= 8.0.0
- Matplotlib >= 3.3.0
- scikit-image >= 0.18.0
- scikit-learn >= 0.24.0
- SciPy >= 1.7.0

## Quick Start

```python
from segmentation_tools import MaskManager, Label, BrushTool, SegmentationEvaluator

# Initialize mask manager
manager = MaskManager((height, width))

# Create labels
label1 = Label(1, "object", (255, 0, 0))
manager.add_label(label1)
manager.set_active_label(1)

# Use brush tool
brush = BrushTool(manager, brush_size=10)
brush.apply_single_point(50, 50)

# Evaluate results
evaluator = SegmentationEvaluator()
iou = evaluator.intersection_over_union(pred_mask, gt_mask)
```

## Examples

See `examples/demo.py` for a comprehensive demonstration of all features.

## API Reference

### Core Classes

#### MaskManager
Main class for managing segmentation masks and labels.

#### Label
Represents a segmentation label with ID, name, and color.

#### Tools
- **BrushTool**: Interactive painting tool
- **EraseTool**: Interactive erasing tool  
- **PolygonTool**: Geometric shape creation tool

#### SegmentationEvaluator
Comprehensive evaluation metrics for segmentation results.

## License

MIT License
