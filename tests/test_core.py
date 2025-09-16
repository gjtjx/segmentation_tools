"""
Tests for segmentation tools core functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from segmentation_tools import MaskManager, Label, BrushTool, EraseTool, PolygonTool, SegmentationEvaluator


class TestLabel:
    """Test Label class functionality."""
    
    def test_label_creation(self):
        """Test basic label creation."""
        label = Label(1, "test_label", (255, 0, 0))
        assert label.id == 1
        assert label.name == "test_label"
        assert label.color == (255, 0, 0)
    
    def test_label_auto_color(self):
        """Test automatic color generation."""
        label = Label(2, "auto_color")
        assert isinstance(label.color, tuple)
        assert len(label.color) == 3
        assert all(0 <= c <= 255 for c in label.color)
    
    def test_label_serialization(self):
        """Test label to/from dict conversion."""
        original = Label(3, "serialize_test", (100, 150, 200))
        data = original.to_dict()
        restored = Label.from_dict(data)
        
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.color == original.color


class TestMaskManager:
    """Test MaskManager functionality."""
    
    def test_mask_manager_creation(self):
        """Test basic mask manager creation."""
        manager = MaskManager((100, 200))
        assert manager.image_shape == (100, 200)
        assert len(manager.masks) == 0
        assert len(manager.labels) == 0
    
    def test_add_label(self):
        """Test adding labels to manager."""
        manager = MaskManager((50, 100))
        label = Label(1, "test", (255, 0, 0))
        
        manager.add_label(label)
        assert 1 in manager.labels
        assert 1 in manager.masks
        assert manager.masks[1].shape == (50, 100)
    
    def test_active_label(self):
        """Test active label management."""
        manager = MaskManager((50, 100))
        label = Label(1, "test", (255, 0, 0))
        manager.add_label(label)
        
        manager.set_active_label(1)
        active = manager.get_active_label()
        assert active.id == 1
        assert active.name == "test"
    
    def test_combined_mask(self):
        """Test combined mask generation."""
        manager = MaskManager((50, 50))
        
        # Add two labels
        label1 = Label(1, "label1")
        label2 = Label(2, "label2")
        manager.add_label(label1)
        manager.add_label(label2)
        
        # Create some mask data
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[10:20, 10:20] = 255
        
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[30:40, 30:40] = 255
        
        manager.update_mask(1, mask1)
        manager.update_mask(2, mask2)
        
        combined = manager.get_combined_mask()
        assert combined[15, 15] == 1  # Label 1 area
        assert combined[35, 35] == 2  # Label 2 area
        assert combined[5, 5] == 0    # Background


class TestBrushTool:
    """Test BrushTool functionality."""
    
    def test_brush_creation(self):
        """Test brush tool creation."""
        manager = MaskManager((100, 100))
        brush = BrushTool(manager, brush_size=10)
        assert brush.brush_size == 10
    
    def test_brush_single_point(self):
        """Test brush application at single point."""
        manager = MaskManager((100, 100))
        label = Label(1, "test")
        manager.add_label(label)
        manager.set_active_label(1)
        
        brush = BrushTool(manager, brush_size=10)
        brush.apply_single_point(50, 50)
        
        mask = manager.get_mask(1)
        assert mask[50, 50] == 255
        assert np.sum(mask > 0) > 1  # Should affect multiple pixels
    
    def test_brush_stroke(self):
        """Test brush stroke application."""
        manager = MaskManager((100, 100))
        label = Label(1, "test")
        manager.add_label(label)
        
        brush = BrushTool(manager, brush_size=5)
        points = [(20, 20), (25, 25), (30, 30)]
        brush.apply(points, label_id=1)
        
        mask = manager.get_mask(1)
        assert mask[20, 20] == 255
        assert mask[25, 25] == 255
        assert mask[30, 30] == 255


class TestEraseTool:
    """Test EraseTool functionality."""
    
    def test_erase_creation(self):
        """Test erase tool creation."""
        manager = MaskManager((100, 100))
        eraser = EraseTool(manager, eraser_size=15)
        assert eraser.eraser_size == 15
    
    def test_erase_functionality(self):
        """Test erase functionality."""
        manager = MaskManager((100, 100))
        label = Label(1, "test")
        manager.add_label(label)
        
        # First paint some area
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        manager.update_mask(1, mask)
        
        # Then erase part of it
        eraser = EraseTool(manager, eraser_size=10)
        eraser.apply_single_point(50, 50, label_id=1)
        
        erased_mask = manager.get_mask(1)
        assert erased_mask[50, 50] == 0  # Should be erased
        assert erased_mask[10, 10] == 255  # Should still be there


class TestPolygonTool:
    """Test PolygonTool functionality."""
    
    def test_polygon_creation(self):
        """Test polygon tool creation."""
        manager = MaskManager((100, 100))
        polygon = PolygonTool(manager)
        assert polygon.mask_manager == manager
    
    def test_polygon_rectangle(self):
        """Test rectangle creation."""
        manager = MaskManager((100, 100))
        label = Label(1, "test")
        manager.add_label(label)
        
        polygon = PolygonTool(manager)
        polygon.apply_rectangle((10, 10), (30, 30), label_id=1)
        
        mask = manager.get_mask(1)
        assert mask[20, 20] == 255  # Inside rectangle
        assert mask[5, 5] == 0      # Outside rectangle
    
    def test_polygon_circle(self):
        """Test circle creation."""
        manager = MaskManager((100, 100))
        label = Label(1, "test")
        manager.add_label(label)
        
        polygon = PolygonTool(manager)
        polygon.apply_circle((50, 50), 10, label_id=1)
        
        mask = manager.get_mask(1)
        assert mask[50, 50] == 255  # Center should be filled
        assert mask[50, 45] == 255  # Inside circle
        assert mask[50, 35] == 0    # Outside circle


class TestSegmentationEvaluator:
    """Test SegmentationEvaluator functionality."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        evaluator = SegmentationEvaluator()
        assert evaluator is not None
    
    def test_iou_calculation(self):
        """Test IoU calculation."""
        evaluator = SegmentationEvaluator()
        
        # Perfect match
        mask1 = np.ones((10, 10), dtype=np.uint8) * 255
        mask2 = np.ones((10, 10), dtype=np.uint8) * 255
        iou = evaluator.intersection_over_union(mask1, mask2)
        assert abs(iou - 1.0) < 1e-6
        
        # No overlap
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[:5, :] = 255
        mask2 = np.zeros((10, 10), dtype=np.uint8)
        mask2[5:, :] = 255
        iou = evaluator.intersection_over_union(mask1, mask2)
        assert abs(iou - 0.0) < 1e-6
    
    def test_dice_calculation(self):
        """Test Dice coefficient calculation."""
        evaluator = SegmentationEvaluator()
        
        # Perfect match
        mask1 = np.ones((10, 10), dtype=np.uint8) * 255
        mask2 = np.ones((10, 10), dtype=np.uint8) * 255
        dice = evaluator.dice_coefficient(mask1, mask2)
        assert abs(dice - 1.0) < 1e-6
    
    def test_pixel_accuracy(self):
        """Test pixel accuracy calculation."""
        evaluator = SegmentationEvaluator()
        
        # Perfect match
        mask1 = np.random.randint(0, 2, (10, 10), dtype=np.uint8) * 255
        mask2 = mask1.copy()
        accuracy = evaluator.pixel_accuracy(mask1, mask2)
        assert abs(accuracy - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])