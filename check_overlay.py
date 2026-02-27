from PIL import Image
import numpy as np
import os

path = r"c:\projects\16 climbing\prj\hold_seg\bg1_masks"

# Load overlay
overlay = Image.open(os.path.join(path, "bg1_overlay.png")).convert('RGB')
ov_arr = np.array(overlay)

# Load background
bg = Image.open(os.path.join(path, "bg1.jpg")).convert('RGB')
bg_arr = np.array(bg)

# Find all regions where overlay differs from background
diff = np.any(ov_arr != bg_arr, axis=2)

# Count different regions
from scipy import ndimage
labeled, num_features = ndimage.label(diff)

print(f"Background size: {bg.size}")
print(f"Overlay differs from background: {np.sum(diff)} pixels")
print(f"Number of distinct regions in overlay: {num_features}")
print()

# Check if the masks in PNG files match regions in overlay
print("Checking if overlay contains mask positions...")

# Load first mask
mask1 = Image.open(os.path.join(path, "bg1_100_0.90.png")).convert('L')
mask1_arr = np.array(mask1)
mask1_binary = mask1_arr > 50

# Extract the mask shape (ignoring position)
mask1_bbox = Image.fromarray(mask1_arr.astype(np.uint8)).getbbox()
mask1_patch = mask1_arr[mask1_bbox[1]:mask1_bbox[3], mask1_bbox[0]:mask1_bbox[2]]

print(f"Mask patch size: {mask1_patch.shape}")
print(f"This mask should be found somewhere in the overlay image...")
