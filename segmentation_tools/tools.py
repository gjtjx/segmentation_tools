"""
Interactive tools for mask editing: brush, erase, and polygon tools.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

from .core import MaskManager


class BaseTool(ABC):
    """Base class for all segmentation tools."""
    
    def __init__(self, mask_manager: MaskManager):
        """
        Initialize tool with mask manager.
        
        Args:
            mask_manager: MaskManager instance to operate on
        """
        self.mask_manager = mask_manager
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> None:
        """Apply the tool operation."""
        pass


class BrushTool(BaseTool):
    """Brush tool for painting masks."""
    
    def __init__(self, mask_manager: MaskManager, brush_size: int = 10):
        """
        Initialize brush tool.
        
        Args:
            mask_manager: MaskManager instance
            brush_size: Diameter of the brush in pixels
        """
        super().__init__(mask_manager)
        self.brush_size = brush_size
    
    def set_brush_size(self, size: int) -> None:
        """Set brush size."""
        self.brush_size = max(1, size)
    
    def apply(self, points: List[Tuple[int, int]], label_id: Optional[int] = None) -> None:
        """
        Apply brush strokes to the mask.
        
        Args:
            points: List of (x, y) coordinates for the brush stroke
            label_id: Label ID to paint (uses active label if None)
        """
        if label_id is None:
            active_label = self.mask_manager.get_active_label()
            if active_label is None:
                raise ValueError("No active label set and no label_id provided")
            label_id = active_label.id
        
        if label_id not in self.mask_manager.labels:
            raise ValueError(f"Label ID {label_id} not found")
        
        mask = self.mask_manager.get_mask(label_id)
        
        # Create brush kernel
        kernel_size = self.brush_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply brush strokes
        for i, (x, y) in enumerate(points):
            # Ensure coordinates are within bounds
            x = max(0, min(x, mask.shape[1] - 1))
            y = max(0, min(y, mask.shape[0] - 1))
            
            # Simply use cv2.circle for brush strokes
            cv2.circle(mask, (x, y), kernel_size // 2, 255, -1)
            # For smooth strokes, connect points
            if i > 0:
                prev_x, prev_y = points[i-1]
                prev_x = max(0, min(prev_x, mask.shape[1] - 1))
                prev_y = max(0, min(prev_y, mask.shape[0] - 1))
                
                # Draw line between points
                cv2.line(mask, (prev_x, prev_y), (x, y), 255, self.brush_size)
        
        self.mask_manager.update_mask(label_id, mask)
    
    def apply_single_point(self, x: int, y: int, label_id: Optional[int] = None) -> None:
        """Apply brush at a single point."""
        self.apply([(x, y)], label_id)


class EraseTool(BaseTool):
    """Erase tool for removing parts of masks."""
    
    def __init__(self, mask_manager: MaskManager, eraser_size: int = 10):
        """
        Initialize erase tool.
        
        Args:
            mask_manager: MaskManager instance
            eraser_size: Diameter of the eraser in pixels
        """
        super().__init__(mask_manager)
        self.eraser_size = eraser_size
    
    def set_eraser_size(self, size: int) -> None:
        """Set eraser size."""
        self.eraser_size = max(1, size)
    
    def apply(self, points: List[Tuple[int, int]], label_id: Optional[int] = None, 
              erase_all: bool = False) -> None:
        """
        Apply eraser to the mask.
        
        Args:
            points: List of (x, y) coordinates for the erase stroke
            label_id: Label ID to erase (uses active label if None, erases all if erase_all=True)
            erase_all: If True, erases from all labels at the given points
        """
        if erase_all:
            # Erase from all masks
            label_ids = list(self.mask_manager.labels.keys())
        else:
            if label_id is None:
                active_label = self.mask_manager.get_active_label()
                if active_label is None:
                    raise ValueError("No active label set and no label_id provided")
                label_id = active_label.id
            label_ids = [label_id]
        
        # Create eraser kernel
        kernel_size = self.eraser_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for lid in label_ids:
            if lid not in self.mask_manager.labels:
                continue
                
            mask = self.mask_manager.get_mask(lid)
            
            # Apply eraser strokes
            for i, (x, y) in enumerate(points):
                # Ensure coordinates are within bounds
                x = max(0, min(x, mask.shape[1] - 1))
                y = max(0, min(y, mask.shape[0] - 1))
                
                # Calculate kernel position
                half_size = kernel_size // 2
                y_start = max(0, y - half_size)
                y_end = min(mask.shape[0], y + half_size + 1)
                x_start = max(0, x - half_size)
                x_end = min(mask.shape[1], x + half_size + 1)
                
                # Apply eraser (set to 0)
                mask[y_start:y_end, x_start:x_end] = 0
                
                # For smooth erasing, connect points
                if i > 0:
                    prev_x, prev_y = points[i-1]
                    prev_x = max(0, min(prev_x, mask.shape[1] - 1))
                    prev_y = max(0, min(prev_y, mask.shape[0] - 1))
                    
                    # Erase line between points
                    cv2.line(mask, (prev_x, prev_y), (x, y), 0, self.eraser_size)
            
            self.mask_manager.update_mask(lid, mask)
    
    def apply_single_point(self, x: int, y: int, label_id: Optional[int] = None, 
                          erase_all: bool = False) -> None:
        """Apply eraser at a single point."""
        self.apply([(x, y)], label_id, erase_all)


class PolygonTool(BaseTool):
    """Polygon tool for creating polygonal selections."""
    
    def __init__(self, mask_manager: MaskManager):
        """
        Initialize polygon tool.
        
        Args:
            mask_manager: MaskManager instance
        """
        super().__init__(mask_manager)
    
    def apply(self, points: List[Tuple[int, int]], label_id: Optional[int] = None, 
              fill: bool = True, erase: bool = False) -> None:
        """
        Apply polygon to the mask.
        
        Args:
            points: List of (x, y) coordinates defining the polygon vertices
            label_id: Label ID to apply to (uses active label if None)
            fill: If True, fills the polygon; if False, only draws the outline
            erase: If True, erases the polygon area instead of adding
        """
        if label_id is None:
            active_label = self.mask_manager.get_active_label()
            if active_label is None:
                raise ValueError("No active label set and no label_id provided")
            label_id = active_label.id
        
        if label_id not in self.mask_manager.labels:
            raise ValueError(f"Label ID {label_id} not found")
        
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")
        
        mask = self.mask_manager.get_mask(label_id)
        
        # Convert points to numpy array
        polygon_points = np.array(points, dtype=np.int32)
        
        # Ensure points are within bounds
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, mask.shape[1] - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, mask.shape[0] - 1)
        
        if fill:
            # Fill polygon
            if erase:
                cv2.fillPoly(mask, [polygon_points], 0)
            else:
                cv2.fillPoly(mask, [polygon_points], 255)
        else:
            # Draw polygon outline
            value = 0 if erase else 255
            cv2.polylines(mask, [polygon_points], isClosed=True, color=value, thickness=2)
        
        self.mask_manager.update_mask(label_id, mask)
    
    def apply_rectangle(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                       label_id: Optional[int] = None, fill: bool = True, erase: bool = False) -> None:
        """
        Apply rectangle selection.
        
        Args:
            top_left: (x, y) coordinates of top-left corner
            bottom_right: (x, y) coordinates of bottom-right corner
            label_id: Label ID to apply to
            fill: If True, fills the rectangle
            erase: If True, erases the rectangle area
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Create rectangle points
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        self.apply(points, label_id, fill, erase)
    
    def apply_circle(self, center: Tuple[int, int], radius: int, 
                    label_id: Optional[int] = None, fill: bool = True, erase: bool = False) -> None:
        """
        Apply circle selection.
        
        Args:
            center: (x, y) coordinates of circle center
            radius: Radius of the circle
            label_id: Label ID to apply to
            fill: If True, fills the circle
            erase: If True, erases the circle area
        """
        if label_id is None:
            active_label = self.mask_manager.get_active_label()
            if active_label is None:
                raise ValueError("No active label set and no label_id provided")
            label_id = active_label.id
        
        if label_id not in self.mask_manager.labels:
            raise ValueError(f"Label ID {label_id} not found")
        
        mask = self.mask_manager.get_mask(label_id)
        
        # Ensure center is within bounds
        x, y = center
        x = max(0, min(x, mask.shape[1] - 1))
        y = max(0, min(y, mask.shape[0] - 1))
        
        value = 0 if erase else 255
        thickness = -1 if fill else 2
        
        cv2.circle(mask, (x, y), radius, value, thickness)
        
        self.mask_manager.update_mask(label_id, mask)