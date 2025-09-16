"""
Core data structures and utilities for segmentation tools.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import json
from PIL import Image
import cv2


class Label:
    """Represents a segmentation label with name, color, and ID."""
    
    def __init__(self, label_id: int, name: str, color: Tuple[int, int, int] = None):
        """
        Initialize a label.
        
        Args:
            label_id: Unique identifier for the label
            name: Human-readable name for the label
            color: RGB color tuple for visualization (auto-generated if None)
        """
        self.id = label_id
        self.name = name
        self.color = color or self._generate_color(label_id)
    
    def _generate_color(self, label_id: int) -> Tuple[int, int, int]:
        """Generate a color based on label ID."""
        # Simple color generation using HSV space
        import colorsys
        hue = (label_id * 37) % 360  # Distribute hues
        rgb = colorsys.hsv_to_rgb(hue/360, 0.7, 0.9)
        return tuple(int(c * 255) for c in rgb)
    
    def to_dict(self) -> Dict:
        """Convert label to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Label":
        """Create label from dictionary."""
        return cls(data["id"], data["name"], tuple(data["color"]))


class MaskManager:
    """Manages segmentation masks and labels."""
    
    def __init__(self, image_shape: Tuple[int, int]):
        """
        Initialize mask manager.
        
        Args:
            image_shape: (height, width) of the image
        """
        self.image_shape = image_shape
        self.masks: Dict[int, np.ndarray] = {}
        self.labels: Dict[int, Label] = {}
        self._active_label_id: Optional[int] = None
    
    def add_label(self, label: Label) -> None:
        """Add a new label."""
        self.labels[label.id] = label
        if label.id not in self.masks:
            self.masks[label.id] = np.zeros(self.image_shape, dtype=np.uint8)
    
    def remove_label(self, label_id: int) -> None:
        """Remove a label and its associated mask."""
        if label_id in self.labels:
            del self.labels[label_id]
        if label_id in self.masks:
            del self.masks[label_id]
    
    def set_active_label(self, label_id: int) -> None:
        """Set the active label for editing."""
        if label_id in self.labels:
            self._active_label_id = label_id
        else:
            raise ValueError(f"Label ID {label_id} not found")
    
    def get_active_label(self) -> Optional[Label]:
        """Get the currently active label."""
        if self._active_label_id is not None:
            return self.labels.get(self._active_label_id)
        return None
    
    def get_mask(self, label_id: int) -> np.ndarray:
        """Get mask for a specific label."""
        if label_id in self.masks:
            return self.masks[label_id].copy()
        return np.zeros(self.image_shape, dtype=np.uint8)
    
    def update_mask(self, label_id: int, mask: np.ndarray) -> None:
        """Update mask for a specific label."""
        if label_id in self.labels:
            self.masks[label_id] = mask.astype(np.uint8)
    
    def get_combined_mask(self) -> np.ndarray:
        """Get combined mask with label IDs as pixel values."""
        combined = np.zeros(self.image_shape, dtype=np.uint8)
        for label_id, mask in self.masks.items():
            combined[mask > 0] = label_id
        return combined
    
    def get_colored_mask(self) -> np.ndarray:
        """Get colored visualization of all masks."""
        colored = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        for label_id, mask in self.masks.items():
            if label_id in self.labels:
                color = self.labels[label_id].color
                colored[mask > 0] = color
        return colored
    
    def save_masks(self, base_path: str) -> None:
        """Save all masks to files."""
        # Save individual masks
        for label_id, mask in self.masks.items():
            mask_path = f"{base_path}_label_{label_id}.png"
            Image.fromarray(mask).save(mask_path)
        
        # Save combined mask
        combined = self.get_combined_mask()
        combined_path = f"{base_path}_combined.png"
        Image.fromarray(combined).save(combined_path)
        
        # Save labels metadata
        labels_data = {
            "labels": {str(k): v.to_dict() for k, v in self.labels.items()},
            "image_shape": self.image_shape
        }
        with open(f"{base_path}_labels.json", "w") as f:
            json.dump(labels_data, f, indent=2)
    
    def load_masks(self, base_path: str) -> None:
        """Load masks from files."""
        # Load labels metadata
        try:
            with open(f"{base_path}_labels.json", "r") as f:
                labels_data = json.load(f)
            
            self.image_shape = tuple(labels_data["image_shape"])
            self.labels = {}
            self.masks = {}
            
            for label_id_str, label_dict in labels_data["labels"].items():
                label_id = int(label_id_str)
                label = Label.from_dict(label_dict)
                self.labels[label_id] = label
                
                # Load corresponding mask
                mask_path = f"{base_path}_label_{label_id}.png"
                try:
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]  # Convert to grayscale if needed
                    self.masks[label_id] = mask
                except FileNotFoundError:
                    self.masks[label_id] = np.zeros(self.image_shape, dtype=np.uint8)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file {base_path}_labels.json not found")