#!/usr/bin/env python3
"""
Interactive example showing how to use segmentation tools in a more realistic scenario.
This example demonstrates how you might integrate the tools into an actual application.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from segmentation_tools import MaskManager, Label, BrushTool, EraseTool, PolygonTool, SegmentationEvaluator


class SegmentationApp:
    """Simple segmentation application demonstrating tool usage."""
    
    def __init__(self, image_path_or_array):
        """Initialize the segmentation app."""
        if isinstance(image_path_or_array, str):
            self.image = np.array(Image.open(image_path_or_array))
        else:
            self.image = image_path_or_array
            
        if len(self.image.shape) == 3:
            self.height, self.width = self.image.shape[:2]
        else:
            self.height, self.width = self.image.shape
            
        # Initialize mask manager
        self.mask_manager = MaskManager((self.height, self.width))
        
        # Initialize tools
        self.brush = BrushTool(self.mask_manager, brush_size=5)
        self.eraser = EraseTool(self.mask_manager, eraser_size=5)
        self.polygon = PolygonTool(self.mask_manager)
        
        # Current tool
        self.current_tool = "brush"
        
        print(f"Initialized segmentation app for image size: {self.width}x{self.height}")
    
    def add_class(self, class_id, class_name, color=None):
        """Add a new segmentation class."""
        label = Label(class_id, class_name, color)
        self.mask_manager.add_label(label)
        print(f"Added class: {class_name} (ID: {class_id})")
        return label
    
    def set_active_class(self, class_id):
        """Set the active class for annotation."""
        self.mask_manager.set_active_label(class_id)
        active_label = self.mask_manager.get_active_label()
        print(f"Active class: {active_label.name} (ID: {class_id})")
    
    def set_tool(self, tool_name):
        """Set the current tool."""
        if tool_name in ["brush", "eraser", "polygon"]:
            self.current_tool = tool_name
            print(f"Current tool: {tool_name}")
        else:
            print(f"Unknown tool: {tool_name}")
    
    def set_brush_size(self, size):
        """Set brush/eraser size."""
        self.brush.set_brush_size(size)
        self.eraser.set_eraser_size(size)
        print(f"Tool size set to: {size}")
    
    def annotate_point(self, x, y):
        """Annotate a single point with the current tool."""
        if self.current_tool == "brush":
            self.brush.apply_single_point(x, y)
            print(f"Painted at ({x}, {y})")
        elif self.current_tool == "eraser":
            self.eraser.apply_single_point(x, y)
            print(f"Erased at ({x}, {y})")
    
    def annotate_stroke(self, points):
        """Annotate a stroke (list of points)."""
        if self.current_tool == "brush":
            self.brush.apply(points)
            print(f"Painted stroke with {len(points)} points")
        elif self.current_tool == "eraser":
            self.eraser.apply(points)
            print(f"Erased stroke with {len(points)} points")
    
    def add_rectangle(self, top_left, bottom_right, class_id=None):
        """Add a rectangular annotation."""
        self.polygon.apply_rectangle(top_left, bottom_right, label_id=class_id)
        print(f"Added rectangle from {top_left} to {bottom_right}")
    
    def add_circle(self, center, radius, class_id=None):
        """Add a circular annotation."""
        self.polygon.apply_circle(center, radius, label_id=class_id)
        print(f"Added circle at {center} with radius {radius}")
    
    def get_mask_for_class(self, class_id):
        """Get mask for a specific class."""
        return self.mask_manager.get_mask(class_id)
    
    def get_combined_mask(self):
        """Get combined mask with all classes."""
        return self.mask_manager.get_combined_mask()
    
    def get_visualization(self, alpha=0.5):
        """Get image with mask overlay."""
        if len(self.image.shape) == 2:
            # Convert grayscale to RGB
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = self.image.copy()
            
        colored_mask = self.mask_manager.get_colored_mask()
        
        # Create overlay
        overlay = image_rgb.copy()
        mask_indices = colored_mask.sum(axis=2) > 0
        
        if np.any(mask_indices):
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) + 
                colored_mask[mask_indices] * alpha
            ).astype(np.uint8)
        
        return overlay
    
    def save_annotations(self, base_path):
        """Save all annotations."""
        self.mask_manager.save_masks(base_path)
        print(f"Saved annotations to {base_path}")
    
    def load_annotations(self, base_path):
        """Load annotations."""
        self.mask_manager.load_masks(base_path)
        print(f"Loaded annotations from {base_path}")
    
    def evaluate_against_ground_truth(self, gt_app):
        """Evaluate against ground truth."""
        evaluator = SegmentationEvaluator()
        results = evaluator.evaluate_mask_manager(
            self.mask_manager, 
            gt_app.mask_manager
        )
        
        print("Evaluation Results:")
        print(f"  Pixel Accuracy: {results['overall']['pixel_accuracy']:.4f}")
        print(f"  IoU: {results['overall']['iou']:.4f}")
        print(f"  Dice: {results['overall']['dice']:.4f}")
        print(f"  Mean IoU: {results['overall']['mean_iou']:.4f}")
        
        return results


def demo_workflow():
    """Demonstrate a typical segmentation workflow."""
    print("=== Segmentation Workflow Demo ===")
    
    # Create a synthetic image
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Add some objects to segment
    cv2.rectangle(image, (50, 50), (120, 120), (255, 100, 100), -1)  # Red square
    cv2.circle(image, (200, 80), 40, (100, 255, 100), -1)  # Green circle
    cv2.ellipse(image, (150, 150), (60, 30), 45, 0, 360, (100, 100, 255), -1)  # Blue ellipse
    
    # Initialize app
    app = SegmentationApp(image)
    
    # Add classes
    app.add_class(1, "square", (255, 0, 0))
    app.add_class(2, "circle", (0, 255, 0))
    app.add_class(3, "ellipse", (0, 0, 255))
    
    # Annotate square with brush
    print("\n--- Annotating square with brush ---")
    app.set_active_class(1)
    app.set_tool("brush")
    app.set_brush_size(8)
    
    # Simulate brush strokes
    for y in range(60, 110, 10):
        for x in range(60, 110, 10):
            app.annotate_point(x, y)
    
    # Annotate circle with rectangle tool (close approximation)
    print("\n--- Annotating circle with polygon tool ---")
    app.set_active_class(2)
    app.add_circle((200, 80), 35, class_id=2)
    
    # Annotate ellipse with polygon approximation
    print("\n--- Annotating ellipse with rectangle ---")
    app.set_active_class(3)
    app.add_rectangle((90, 120), (210, 180), class_id=3)
    
    # Refine with eraser
    print("\n--- Refining with eraser ---")
    app.set_tool("eraser")
    app.set_brush_size(5)
    # Erase some parts to make rectangle more ellipse-like
    corners = [(95, 125), (95, 175), (205, 125), (205, 175)]
    for x, y in corners:
        app.annotate_point(x, y)
    
    # Get visualization
    result = app.get_visualization()
    
    # Show statistics
    print("\n--- Annotation Statistics ---")
    combined = app.get_combined_mask()
    unique_labels = np.unique(combined)
    
    for label_id in unique_labels:
        if label_id == 0:
            continue
        mask = app.get_mask_for_class(label_id)
        pixel_count = np.sum(mask > 0)
        label_name = app.mask_manager.labels[label_id].name
        print(f"  {label_name}: {pixel_count} pixels")
    
    return app, image, result


def create_comparison_visualization(original_image, annotated_result, save_path="/tmp/workflow_demo.png"):
    """Create a comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(annotated_result)
    axes[1].set_title("Annotated Result")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {save_path}")
    return save_path


def main():
    """Run the interactive workflow demo."""
    print("Interactive Segmentation Tools - Workflow Demo")
    print("=" * 50)
    
    # Run workflow demo
    app, original_image, result = demo_workflow()
    
    # Create visualization
    viz_path = create_comparison_visualization(original_image, result)
    
    print("\n" + "=" * 50)
    print("Workflow demo completed!")
    print(f"Visualization saved to: {viz_path}")
    print("\nThis demo showed:")
    print("• Creating a segmentation application")
    print("• Adding multiple classes")
    print("• Using different tools (brush, polygon, eraser)")
    print("• Refining annotations")
    print("• Getting statistics and visualizations")
    
    return app


if __name__ == "__main__":
    main()