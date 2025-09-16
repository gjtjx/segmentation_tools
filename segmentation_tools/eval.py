"""
Evaluation tools for segmentation results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix
import json

from .core import MaskManager, Label


class SegmentationEvaluator:
    """Evaluator for segmentation results with various metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def intersection_over_union(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                               ignore_background: bool = True) -> float:
        """
        Calculate Intersection over Union (IoU) score.
        
        Args:
            pred_mask: Predicted mask (binary or multi-class)
            gt_mask: Ground truth mask (binary or multi-class)
            ignore_background: If True, ignores background class (value 0)
        
        Returns:
            IoU score (0.0 to 1.0)
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        # For binary masks
        if ignore_background:
            pred_binary = pred_mask > 0
            gt_binary = gt_mask > 0
        else:
            pred_binary = pred_mask
            gt_binary = gt_mask
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def dice_coefficient(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                        ignore_background: bool = True) -> float:
        """
        Calculate Dice coefficient (F1-score for segmentation).
        
        Args:
            pred_mask: Predicted mask (binary or multi-class)
            gt_mask: Ground truth mask (binary or multi-class)
            ignore_background: If True, ignores background class (value 0)
        
        Returns:
            Dice coefficient (0.0 to 1.0)
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        # For binary masks
        if ignore_background:
            pred_binary = pred_mask > 0
            gt_binary = gt_mask > 0
        else:
            pred_binary = pred_mask
            gt_binary = gt_mask
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        total = pred_binary.sum() + gt_binary.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2.0 * intersection) / total
    
    def pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
        
        Returns:
            Pixel accuracy (0.0 to 1.0)
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        correct_pixels = np.sum(pred_mask == gt_mask)
        total_pixels = pred_mask.size
        
        return correct_pixels / total_pixels
    
    def mean_iou_multiclass(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                           num_classes: Optional[int] = None, 
                           ignore_background: bool = True) -> Tuple[float, Dict[int, float]]:
        """
        Calculate mean IoU for multi-class segmentation.
        
        Args:
            pred_mask: Predicted mask with class IDs as pixel values
            gt_mask: Ground truth mask with class IDs as pixel values
            num_classes: Number of classes (auto-detected if None)
            ignore_background: If True, ignores class 0 (background)
        
        Returns:
            Tuple of (mean_iou, per_class_iou_dict)
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        if num_classes is None:
            num_classes = max(np.max(pred_mask), np.max(gt_mask)) + 1
        
        per_class_iou = {}
        valid_classes = []
        
        start_class = 1 if ignore_background else 0
        
        for class_id in range(start_class, num_classes):
            pred_class = (pred_mask == class_id)
            gt_class = (gt_mask == class_id)
            
            intersection = np.logical_and(pred_class, gt_class).sum()
            union = np.logical_or(pred_class, gt_class).sum()
            
            if union > 0:  # Only include classes that exist in ground truth or prediction
                iou = intersection / union
                per_class_iou[class_id] = iou
                valid_classes.append(iou)
        
        mean_iou = np.mean(valid_classes) if valid_classes else 0.0
        
        return mean_iou, per_class_iou
    
    def confusion_matrix_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                                num_classes: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate metrics based on confusion matrix.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            num_classes: Number of classes (auto-detected if None)
        
        Returns:
            Dictionary with precision, recall, f1_score arrays and overall accuracy
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        # Flatten masks
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        if num_classes is None:
            labels = np.unique(np.concatenate([pred_flat, gt_flat]))
        else:
            labels = np.arange(num_classes)
        
        # Calculate confusion matrix
        cm = confusion_matrix(gt_flat, pred_flat, labels=labels)
        
        # Calculate per-class metrics
        precision = np.zeros(len(labels))
        recall = np.zeros(len(labels))
        f1_score = np.zeros(len(labels))
        
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            if tp + fp > 0:
                precision[i] = tp / (tp + fp)
            else:
                precision[i] = 0.0
            
            if tp + fn > 0:
                recall[i] = tp / (tp + fn)
            else:
                recall[i] = 0.0
            
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_score[i] = 0.0
        
        overall_accuracy = np.trace(cm) / cm.sum()
        
        return {
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "overall_accuracy": overall_accuracy,
            "labels": labels
        }
    
    def evaluate_mask_manager(self, pred_manager: MaskManager, gt_manager: MaskManager, 
                             detailed: bool = True) -> Dict:
        """
        Evaluate MaskManager predictions against ground truth.
        
        Args:
            pred_manager: Predicted masks
            gt_manager: Ground truth masks
            detailed: If True, includes per-label metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        if pred_manager.image_shape != gt_manager.image_shape:
            raise ValueError("Predicted and ground truth masks must have the same image shape")
        
        # Get combined masks
        pred_combined = pred_manager.get_combined_mask()
        gt_combined = gt_manager.get_combined_mask()
        
        # Overall metrics
        results = {
            "overall": {
                "pixel_accuracy": self.pixel_accuracy(pred_combined, gt_combined),
                "iou": self.intersection_over_union(pred_combined, gt_combined),
                "dice": self.dice_coefficient(pred_combined, gt_combined)
            }
        }
        
        # Multi-class metrics
        mean_iou, per_class_iou = self.mean_iou_multiclass(pred_combined, gt_combined)
        results["overall"]["mean_iou"] = mean_iou
        
        if detailed:
            # Per-label evaluation
            results["per_label"] = {}
            
            # Find common labels
            all_label_ids = set(pred_manager.labels.keys()) | set(gt_manager.labels.keys())
            
            for label_id in all_label_ids:
                pred_mask = pred_manager.get_mask(label_id) if label_id in pred_manager.labels else np.zeros(pred_manager.image_shape, dtype=np.uint8)
                gt_mask = gt_manager.get_mask(label_id) if label_id in gt_manager.labels else np.zeros(gt_manager.image_shape, dtype=np.uint8)
                
                label_metrics = {
                    "iou": self.intersection_over_union(pred_mask, gt_mask, ignore_background=False),
                    "dice": self.dice_coefficient(pred_mask, gt_mask, ignore_background=False),
                    "pixel_accuracy": self.pixel_accuracy(pred_mask, gt_mask)
                }
                
                # Add label information
                if label_id in pred_manager.labels:
                    label_metrics["pred_label_name"] = pred_manager.labels[label_id].name
                if label_id in gt_manager.labels:
                    label_metrics["gt_label_name"] = gt_manager.labels[label_id].name
                
                results["per_label"][label_id] = label_metrics
            
            # Confusion matrix metrics for multi-class
            cm_metrics = self.confusion_matrix_metrics(pred_combined, gt_combined)
            results["confusion_matrix_metrics"] = {
                "precision": cm_metrics["precision"].tolist(),
                "recall": cm_metrics["recall"].tolist(),
                "f1_score": cm_metrics["f1_score"].tolist(),
                "overall_accuracy": cm_metrics["overall_accuracy"],
                "labels": cm_metrics["labels"].tolist()
            }
        
        return results
    
    def save_evaluation_report(self, results: Dict, filepath: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save the report
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_evaluation_report(self, filepath: str) -> Dict:
        """
        Load evaluation results from a JSON file.
        
        Args:
            filepath: Path to the evaluation report
        
        Returns:
            Evaluation results dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)