#!/usr/bin/env python3
"""
Example usage of segmentation tools for interactive mask creation and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

from segmentation_tools import MaskManager, Label, BrushTool, EraseTool, PolygonTool, SegmentationEvaluator


def create_demo_image():
    """Create a simple demo image for testing."""
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles
    image[50:100, 50:150] = [255, 0, 0]  # Red rectangle
    image[100:150, 150:250] = [0, 255, 0]  # Green rectangle
    image[75:125, 200:250] = [0, 0, 255]  # Blue rectangle
    
    return image


def demo_basic_usage():
    """Demonstrate basic segmentation tools usage."""
    print("=== Basic Segmentation Tools Demo ===")
    
    # Create a demo image
    image = create_demo_image()
    height, width = image.shape[:2]
    
    # Initialize mask manager
    manager = MaskManager((height, width))
    
    # Create labels
    red_label = Label(1, "red_object", (255, 100, 100))
    green_label = Label(2, "green_object", (100, 255, 100))
    blue_label = Label(3, "blue_object", (100, 100, 255))
    
    manager.add_label(red_label)
    manager.add_label(green_label)
    manager.add_label(blue_label)
    
    print(f"Created mask manager with image shape: {manager.image_shape}")
    print(f"Added {len(manager.labels)} labels")
    
    # Initialize tools
    brush = BrushTool(manager, brush_size=15)
    eraser = EraseTool(manager, eraser_size=10)
    polygon = PolygonTool(manager)
    
    # Demonstrate brush tool
    print("\n--- Brush Tool Demo ---")
    manager.set_active_label(1)  # Red label
    brush.apply_single_point(75, 100)  # Paint in red rectangle area
    brush.apply([(80, 105), (85, 110), (90, 115)])  # Paint a stroke
    
    red_mask = manager.get_mask(1)
    print(f"Red mask has {np.sum(red_mask > 0)} painted pixels")
    
    # Demonstrate polygon tool
    print("\n--- Polygon Tool Demo ---")
    # Create a rectangle for green object
    polygon.apply_rectangle((150, 100), (250, 150), label_id=2)
    
    green_mask = manager.get_mask(2)
    print(f"Green mask has {np.sum(green_mask > 0)} painted pixels")
    
    # Create a circle for blue object
    polygon.apply_circle((225, 100), 25, label_id=3)
    
    blue_mask = manager.get_mask(3)
    print(f"Blue mask has {np.sum(blue_mask > 0)} painted pixels")
    
    # Demonstrate eraser tool
    print("\n--- Eraser Tool Demo ---")
    original_pixels = np.sum(green_mask > 0)
    eraser.apply_single_point(200, 125, label_id=2)  # Erase part of green mask
    
    green_mask_after = manager.get_mask(2)
    erased_pixels = original_pixels - np.sum(green_mask_after > 0)
    print(f"Erased {erased_pixels} pixels from green mask")
    
    # Get combined visualization
    combined_mask = manager.get_combined_mask()
    colored_mask = manager.get_colored_mask()
    
    print(f"\nCombined mask has {len(np.unique(combined_mask))} unique values")
    
    return manager, image, colored_mask


def demo_evaluation():
    """Demonstrate evaluation functionality."""
    print("\n=== Evaluation Demo ===")
    
    # Create two mask managers for comparison
    height, width = 100, 150
    
    # Ground truth
    gt_manager = MaskManager((height, width))
    gt_label1 = Label(1, "object1", (255, 0, 0))
    gt_label2 = Label(2, "object2", (0, 255, 0))
    gt_manager.add_label(gt_label1)
    gt_manager.add_label(gt_label2)
    
    # Create ground truth masks
    polygon_tool = PolygonTool(gt_manager)
    polygon_tool.apply_rectangle((20, 20), (60, 60), label_id=1)
    polygon_tool.apply_circle((100, 75), 20, label_id=2)
    
    # Predicted results (slightly different)
    pred_manager = MaskManager((height, width))
    pred_label1 = Label(1, "predicted_object1", (255, 0, 0))
    pred_label2 = Label(2, "predicted_object2", (0, 255, 0))
    pred_manager.add_label(pred_label1)
    pred_manager.add_label(pred_label2)
    
    # Create predicted masks (with some differences)
    polygon_tool_pred = PolygonTool(pred_manager)
    polygon_tool_pred.apply_rectangle((22, 18), (58, 62), label_id=1)  # Slightly offset
    polygon_tool_pred.apply_circle((98, 73), 18, label_id=2)  # Smaller and offset
    
    # Evaluate
    evaluator = SegmentationEvaluator()
    results = evaluator.evaluate_mask_manager(pred_manager, gt_manager)
    
    print("Evaluation Results:")
    print(f"Overall Pixel Accuracy: {results['overall']['pixel_accuracy']:.4f}")
    print(f"Overall IoU: {results['overall']['iou']:.4f}")
    print(f"Overall Dice: {results['overall']['dice']:.4f}")
    print(f"Mean IoU: {results['overall']['mean_iou']:.4f}")
    
    print("\nPer-label results:")
    for label_id, metrics in results['per_label'].items():
        print(f"Label {label_id}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
    
    return results


def demo_save_load():
    """Demonstrate saving and loading functionality."""
    print("\n=== Save/Load Demo ===")
    
    # Create a mask manager with some data
    manager = MaskManager((50, 80))
    label1 = Label(1, "test_object", (200, 100, 50))
    manager.add_label(label1)
    
    # Add some mask data
    brush = BrushTool(manager, brush_size=8)
    brush.apply([(20, 25), (25, 30), (30, 35)], label_id=1)
    
    # Save to temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = os.path.join(temp_dir, "test_masks")
        manager.save_masks(base_path)
        
        print(f"Saved masks to {temp_dir}")
        
        # Create a new manager and load
        new_manager = MaskManager((1, 1))  # Will be overridden
        new_manager.load_masks(base_path)
        
        print(f"Loaded masks with shape: {new_manager.image_shape}")
        print(f"Loaded {len(new_manager.labels)} labels")
        
        # Verify data integrity
        original_mask = manager.get_mask(1)
        loaded_mask = new_manager.get_mask(1)
        
        masks_equal = np.array_equal(original_mask, loaded_mask)
        print(f"Masks are identical after save/load: {masks_equal}")
        
        # Check label data
        original_label = manager.labels[1]
        loaded_label = new_manager.labels[1]
        labels_equal = (original_label.name == loaded_label.name and 
                       original_label.color == loaded_label.color)
        print(f"Labels are identical after save/load: {labels_equal}")


def create_visualization():
    """Create a visualization of the segmentation tools in action."""
    print("\n=== Creating Visualization ===")
    
    # Get demo results
    manager, image, colored_mask = demo_basic_usage()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Combined mask
    combined = manager.get_combined_mask()
    axes[1].imshow(combined, cmap='tab10')
    axes[1].set_title("Segmentation Mask\n(Label IDs)")
    axes[1].axis('off')
    
    # Colored overlay
    overlay = image.copy()
    # Make colored mask semi-transparent
    mask_indices = colored_mask.sum(axis=2) > 0
    overlay[mask_indices] = (overlay[mask_indices] * 0.6 + colored_mask[mask_indices] * 0.4).astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Image with Mask Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = "/tmp/segmentation_demo.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {viz_path}")
    return viz_path


def main():
    """Run all demonstrations."""
    print("Segmentation Tools - Complete Demo")
    print("=" * 50)
    
    # Run demos
    demo_basic_usage()
    demo_evaluation() 
    demo_save_load()
    viz_path = create_visualization()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"Check the visualization at: {viz_path}")
    print("\nThe segmentation tools provide:")
    print("• Interactive mask creation with brush, erase, and polygon tools")
    print("• Label management and multi-class segmentation support")
    print("• Comprehensive evaluation metrics (IoU, Dice, accuracy)")
    print("• Save/load functionality for persistence")
    print("• Easy integration with existing computer vision workflows")


if __name__ == "__main__":
    main()