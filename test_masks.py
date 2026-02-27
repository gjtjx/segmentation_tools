from PIL import Image
import os
import numpy as np

path = r"c:\projects\16 climbing\prj\hold_seg\bg1_masks"

# Test first 5 masks
files = [f for f in os.listdir(path) if f.endswith('.png') and 'overlay' not in f][:5]

print("Checking mask positions:")
print("-" * 80)

for f in files:
    img = Image.open(os.path.join(path, f))
    alpha = img.convert('L')
    
    # Get bbox
    bbox = alpha.getbbox()
    
    # Count white pixels
    arr = np.array(alpha)
    white_pixels = np.sum(arr > 50)
    total_pixels = arr.size
    
    # Get center of bbox
    if bbox:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        print(f"{f}:")
        print(f"  BBox: {bbox}")
        print(f"  Center: ({center_x:.0f}, {center_y:.0f})")
        print(f"  Size: {width}x{height}")
        print(f"  White pixels: {white_pixels} / {total_pixels}")
        print()

# Check if bg1_overlay exists and compare
overlay_path = os.path.join(path, "bg1_overlay.png")
if os.path.exists(overlay_path):
    print("\nChecking bg1_overlay.png:")
    overlay = Image.open(overlay_path)
    print(f"  Size: {overlay.size}")
    print(f"  Mode: {overlay.mode}")
    
    # Sample a few points to see if there's color variation
    ov_arr = np.array(overlay)
    unique_colors = len(np.unique(ov_arr.reshape(-1, ov_arr.shape[-1]), axis=0))
    print(f"  Unique colors: {unique_colors}")
