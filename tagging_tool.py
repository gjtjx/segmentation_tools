import os
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, colorchooser, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageColor, ImageChops
import numpy as np
import cv2

# Configuration
TARGET_SIZE = (1200, 900)  # Max display size

class MaskManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.bg_path = None
        self.masks = [] # List of tuples: (filename, resized_image_alpha_array, original_size_tuple)
        self.scale_factor = 1.0
        self.display_size = (0, 0)
        
        # Auto-detect background image (jpg file)
        self.find_background()
        self.load_background()
        self.load_masks()
        
        # Generate random colors for all masks initially
        self.mask_colors = {} # filename -> (r,g,b)
        for m in self.masks:
             self.mask_colors[m['filename']] = tuple(np.random.randint(0, 256, 3))

    def find_background(self):
        """Find the first jpg file in the directory to use as background"""
        files = os.listdir(self.base_dir)
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not jpg_files:
            raise FileNotFoundError(f"No JPG background image found in {self.base_dir}")
        
        # Use the first jpg file found
        self.bg_path = os.path.join(self.base_dir, jpg_files[0])
        print(f"Found background image: {jpg_files[0]}")

    def load_background(self):
        if not os.path.exists(self.bg_path):
            raise FileNotFoundError(f"Background image not found at {self.bg_path}")
        
        self.original_bg = Image.open(self.bg_path).convert("RGB")
        w, h = self.original_bg.size
        
        # Calculate scale to fit TARGET_SIZE
        scale_w = TARGET_SIZE[0] / w
        scale_h = TARGET_SIZE[1] / h
        self.scale_factor = min(scale_w, scale_h, 1.0) # Do not upscale
        
        self.display_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        self.display_bg = self.original_bg.resize(self.display_size, Image.Resampling.LANCZOS)
        print(f"Original BG: {w}x{h}. Display Size: {self.display_size}. Scale: {self.scale_factor}")

    def load_masks(self):
        files = os.listdir(self.base_dir)
        # Load all PNG files as masks (excluding overlay files)
        mask_files = [f for f in files if f.lower().endswith('.png') and 'overlay' not in f.lower()]
        
        print(f"Loading {len(mask_files)} masks...")
        
        for idx, f in enumerate(mask_files):
            full_path = os.path.join(self.base_dir, f)
            try:
                # Load full-size image (4000x3000)
                img = Image.open(full_path)
                
                # Convert to grayscale mask (white=255 is foreground)
                alpha = img.convert('L')
                
                # Threshold to clean binary mask
                alpha = alpha.point(lambda p: 255 if p > 50 else 0)
                
                # Resize the FULL mask to display size - white pixels will stay in correct position
                resized_alpha = alpha.resize(self.display_size, Image.Resampling.NEAREST)
                
                # Get bbox of the content in the resized mask
                bbox = resized_alpha.getbbox()
                if not bbox:
                    continue # Skip empty masks
                
                self.masks.append({
                    'filename': f,
                    'alpha': resized_alpha, # Full-size resized mask
                    'bbox': bbox # Position where content exists
                })
                
                if idx % 50 == 0:
                    print(f"Loaded {idx}/{len(mask_files)}...", end='\r')

            except Exception as e:
                print(f"Error loading mask {f}: {e}")
        print(f"\nMask loading complete. Loaded {len(self.masks)} masks.")

    def get_mask_at(self, x, y):
        # Very tight hit detection - must click ON the mask
        # Only allow small radius (2px) to help with precision but prevent loose hits
        search_radius = 2
        
        candidates = []
        
        # Iterate all masks
        for mask_data in self.masks:
            bbox = mask_data['bbox']
            if not bbox: continue
            
            # Quick bbox check - must be inside or very close
            if (bbox[0] - search_radius <= x <= bbox[2] + search_radius and 
                bbox[1] - search_radius <= y <= bbox[3] + search_radius):
                
                # Direct check: must find mask pixel very close to click point
                found = False
                
                # Check in a small cross pattern first (most likely to hit)
                check_points = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), 
                               (-2, 0), (2, 0), (0, -2), (0, 2),
                               (-1, -1), (1, -1), (-1, 1), (1, 1)]
                
                for dx, dy in check_points:
                    chk_x, chk_y = x + dx, y + dy
                    
                    # Check in full-size mask coordinates
                    if 0 <= chk_x < self.display_size[0] and 0 <= chk_y < self.display_size[1]:
                         val = mask_data['alpha'].getpixel((chk_x, chk_y))
                         if val > 128: # Higher threshold - must be solid white
                             found = True
                             break
                
                if found:
                    candidates.append(mask_data)

        # If multiple candidates, pick the smallest one (most specific/on top)
        if candidates:
             candidates.sort(key=lambda m: (m['bbox'][2]-m['bbox'][0]) * (m['bbox'][3]-m['bbox'][1]))
             return candidates[0]
             
        return None

class TaggingApp:
    def __init__(self, root, base_dir):
        self.root = root
        self.root.title("Climbing Route Tagger")
        
        self.manager = MaskManager(base_dir)
        
        self.selected_route_masks = [] # List of dict: [{'filename': str, 'type': 'hand'/'foot'}, ...]
        self.current_color = "#FF0000" # Default Red
        self.transparency_val = 150 # Default Alpha
        self.route_name = tk.StringVar()
        self.route_type = tk.StringVar(value="hand") # hand or foot
        
        # Store all saved routes for visualization
        self.saved_routes = {} # Dict: route_folder_name -> {'color': hex, 'masks': [filenames], 'type': 'hand'/'foot'}
        
        # Animation state
        self.is_animating = False
        self.animation_route = None
        self.animation_route_list = [] # List of routes to animate in sequence
        self.animation_route_index = 0 # Current route in the list
        self.animation_index = 0
        self.animation_blink_state = False
        self.animation_blink_count = 0
        self.show_route_arrows = False # Show arrows for completed route
        
        # Animation progress for sliding icons (0.0 to 1.0)
        self.animation_progress = 0.0
        
        # Create icons for hand and foot
        self.hand_icon = self.create_hand_icon()
        self.foot_icon = self.create_foot_icon()
        
        # Merge mode state
        self.is_merge_mode = False
        self.merge_selected_masks = []
        
        try:
            from PIL import ImageFont
            self.font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            self.font = None
        
        self.setup_ui()
        self.load_saved_routes()
        self.refresh_canvas()
        self.refresh_route_list()

    def setup_ui(self):
        # Layout: Left Canvas, Right Controls
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Frame
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, 
                                width=self.manager.display_size[0], 
                                height=self.manager.display_size[1],
                                bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Controls Frame
        self.controls_frame = tk.Frame(main_frame, width=300)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.controls_frame, text="Route Name:").pack(anchor="w")
        tk.Entry(self.controls_frame, textvariable=self.route_name).pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(self.controls_frame, text="Route Type:").pack(anchor="w")
        type_frame = tk.Frame(self.controls_frame)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Radiobutton(type_frame, text="Hand", variable=self.route_type, value="hand").pack(side=tk.LEFT)
        tk.Radiobutton(type_frame, text="Foot", variable=self.route_type, value="foot").pack(side=tk.LEFT)
        
        self.color_btn = tk.Button(self.controls_frame, text="Choose Color", bg=self.current_color, command=self.choose_color)
        self.color_btn.pack(fill=tk.X, pady=5)

        tk.Label(self.controls_frame, text="Transparency:").pack(anchor="w")
        self.transparency_scale = tk.Scale(self.controls_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=self.on_transparency_change)
        self.transparency_scale.set(self.transparency_val)
        self.transparency_scale.pack(fill=tk.X, pady=5)
        
        tk.Button(self.controls_frame, text="Clear Selection", command=self.clear_selection).pack(fill=tk.X, pady=5)
        
        tk.Button(self.controls_frame, text="Merge Masks", command=self.start_merge_mode, bg="lightyellow").pack(fill=tk.X, pady=5)
        
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        tk.Button(self.controls_frame, text="Save Route", command=self.save_route, bg="lightblue").pack(fill=tk.X, pady=5)
        
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        tk.Label(self.controls_frame, text="Existing Routes:").pack(anchor="w")
        self.route_listbox = tk.Listbox(self.controls_frame, height=15)
        self.route_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.route_listbox.bind("<<ListboxSelect>>", self.on_route_select)

        self.highlight_btn = tk.Button(self.controls_frame, text="Highlight Route Masks", command=self.highlight_current_route)
        self.highlight_btn.pack(fill=tk.X, pady=5)

    def choose_color(self):
        color = colorchooser.askcolor(color=self.current_color)[1]
        if color:
            self.current_color = color
            self.color_btn.config(bg=color)
            self.refresh_canvas()

    def on_transparency_change(self, val):
        self.transparency_val = int(float(val))
        self.refresh_canvas()

    def on_click(self, event):
        x, y = event.x, event.y
        print(f"Click at {x}, {y}")
        mask = self.manager.get_mask_at(x, y)
        if mask:
            fname = mask['filename']
            print(f"Hit mask: {fname}")
            
            # Check if in merge mode
            if self.is_merge_mode:
                if fname in self.merge_selected_masks:
                    self.merge_selected_masks.remove(fname)
                else:
                    self.merge_selected_masks.append(fname)
                print(f"Merge selection: {len(self.merge_selected_masks)} masks")
                self.refresh_canvas()
            else:
                # Normal selection mode
                # Check if already selected with the same type
                current_type = self.route_type.get()
                
                # Check if this exact combination (filename + type) exists
                existing_entry = None
                for m in self.selected_route_masks:
                    if isinstance(m, dict):
                        if m['filename'] == fname and m['type'] == current_type:
                            existing_entry = m
                            break
                    elif m == fname:  # Old format compatibility
                        existing_entry = m
                        break
                
                if existing_entry:
                    # Remove this specific filename+type combination
                    self.selected_route_masks.remove(existing_entry)
                    print(f"Removed {fname} as {current_type}")
                else:
                    # Add with current type (allows same mask as both hand and foot)
                    self.selected_route_masks.append({'filename': fname, 'type': current_type})
                    print(f"Added {fname} as {current_type}")
                self.refresh_canvas()
        else:
            print("No mask found at click location.")

    def create_hand_icon(self):
        """Create a modern, tech-style hand icon for animation"""
        size = 60
        icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        
        # Modern hand design - simplified geometric style
        # Using bright cyan/blue tech color scheme
        
        # Outer glow (multiple layers for stronger effect)
        for i in range(8, 0, -1):
            alpha = int(180 * (i / 8))
            draw.ellipse([30-20-i, 30-20-i, 30+20+i, 30+20+i], 
                        fill=(0, 200, 255, alpha // 4))
        
        # Main hand silhouette - simplified palm with 5 fingers
        # Palm (rounded rectangle)
        palm_color = (0, 220, 255, 255)
        draw.ellipse([15, 22, 45, 48], fill=palm_color)
        
        # Fingers - simplified as rounded rectangles
        finger_data = [
            (20, 8, 24, 24),    # Index
            (26, 4, 30, 24),    # Middle
            (32, 4, 36, 24),    # Ring
            (38, 8, 42, 24),    # Pinky
        ]
        
        for fx1, fy1, fx2, fy2 in finger_data:
            draw.rounded_rectangle([fx1, fy1, fx2, fy2], radius=2, fill=palm_color)
        
        # Thumb
        draw.ellipse([12, 26, 20, 36], fill=palm_color)
        
        # Inner bright core
        core_color = (100, 255, 255, 200)
        draw.ellipse([24, 28, 36, 40], fill=core_color)
        
        # Tech border - bright outline
        border_color = (0, 255, 255, 255)
        draw.ellipse([14, 21, 46, 49], outline=border_color, width=3)
        
        # Add glowing particles around
        for px, py in [(18, 18), (42, 18), (18, 42), (42, 42), (30, 12)]:
            draw.ellipse([px-2, py-2, px+2, py+2], fill=(150, 255, 255, 220))
        
        return icon
    
    def create_foot_icon(self):
        """Create a modern, tech-style foot icon for animation"""
        size = 60
        icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        
        # Modern foot design - realistic foot sole shape with arch
        # Using bright orange/red tech color scheme for contrast with hand
        
        # Outer glow (multiple layers for stronger effect)
        for i in range(8, 0, -1):
            alpha = int(180 * (i / 8))
            draw.ellipse([30-22-i, 30-22-i, 30+22+i, 30+22+i], 
                        fill=(255, 100, 0, alpha // 4))
        
        # Main foot silhouette - realistic shape with arch
        foot_color = (255, 120, 0, 255)
        
        # Create realistic foot sole shape using polygon
        # Starting from heel, going around clockwise
        foot_outline = [
            (26, 50),   # Heel bottom left
            (34, 50),   # Heel bottom right
            (38, 48),   # Heel outer edge
            (40, 42),   # Outer midfoot
            (41, 35),   # Outer arch (higher)
            (40, 28),   # Outer forefoot
            (38, 22),   # Start of toes outer
            (36, 18),   # Before pinky toe
            (32, 16),   # Middle toes
            (28, 16),   # Before big toe
            (24, 18),   # Big toe area
            (22, 22),   # Forefoot inner
            (21, 28),   # Inner ball
            (20, 32),   # Inner arch start
            (21, 38),   # Inner arch (curved in)
            (23, 44),   # Inner heel
            (26, 50),   # Back to heel
        ]
        
        draw.polygon(foot_outline, fill=foot_color)
        
        # 5 individual toes at the top
        toe_data = [
            (23, 12, 28, 18),   # Big toe (largest, inner)
            (28, 10, 32, 17),   # Second toe
            (32, 10, 36, 17),   # Middle toe
            (36, 11, 39, 17),   # Fourth toe
            (39, 13, 42, 18),   # Pinky toe (smallest, outer)
        ]
        
        for tx1, ty1, tx2, ty2 in toe_data:
            draw.ellipse([tx1, ty1, tx2, ty2], fill=foot_color)
        
        # Add depth with highlights
        highlight_color = (255, 200, 100, 150)
        # Highlight on forefoot (ball of foot)
        draw.ellipse([25, 20, 37, 28], fill=highlight_color)
        # Highlight on heel
        draw.ellipse([26, 43, 34, 48], fill=highlight_color)
        
        # Inner bright core
        core_color = (255, 200, 100, 200)
        draw.ellipse([27, 30, 35, 38], fill=core_color)
        
        # Tech border - bright orange outline
        border_color = (255, 150, 0, 255)
        # Main sole outline
        draw.polygon(foot_outline, outline=border_color, width=3)
        # Toe outlines for tech effect
        for tx1, ty1, tx2, ty2 in toe_data:
            draw.ellipse([tx1-1, ty1-1, tx2+1, ty2+1], outline=border_color, width=2)
        
        # Add glowing particles around
        for px, py in [(20, 15), (42, 15), (20, 48), (38, 48), (32, 8)]:
            draw.ellipse([px-2, py-2, px+2, py+2], fill=(255, 200, 100, 220))
        
        return icon
    
    def clear_selection(self):
        self.selected_route_masks = []
        self.show_route_arrows = False
        self.animation_route = None
        self.refresh_canvas()

    def refresh_canvas(self):
        # Determine active masks for background processing (Grayscale effect)
        active_filenames = set()
        use_grayscale_bg = False
        
        if (self.is_animating and self.animation_route) or (self.show_route_arrows and self.animation_route):
            use_grayscale_bg = True
            
            # Helper logic to collect active masks
            target_routes = self.animation_route_list if hasattr(self, 'animation_route_list') else [self.animation_route]
            
            for route_name in target_routes:
                route_data = self.saved_routes.get(route_name)
                if not route_data: continue
                
                # Split masks
                h_masks = []
                f_masks = []
                
                for m in route_data['masks']:
                    fname = m['filename'] if isinstance(m, dict) else m
                    m_type = m['type'] if isinstance(m, dict) else route_data.get('type', 'hand')
                    
                    if m_type == 'hand': h_masks.append(fname)
                    else: f_masks.append(fname)
                
                # Determine limit
                if self.is_animating:
                    # Current step (animation_index) IS included in active set
                    # so background at current step is colored.
                    limit_h = min(self.animation_index + 1, len(h_masks))
                    limit_f = min(self.animation_index + 1, len(f_masks))
                else:
                    limit_h = len(h_masks)
                    limit_f = len(f_masks)
                
                active_filenames.update(h_masks[:limit_h])
                active_filenames.update(f_masks[:limit_f])

        # Create base image
        color_bg = self.manager.display_bg.copy().convert("RGBA")
        
        if use_grayscale_bg:
            gray_bg = color_bg.convert("L").convert("RGBA")
            
            if not active_filenames:
                 base = gray_bg
            else:
                 # Create union mask for active areas
                 active_mask_union = Image.new("L", color_bg.size, 0)
                 
                 for fname in active_filenames:
                      m_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                      if m_data:
                          active_mask_union = ImageChops.lighter(active_mask_union, m_data['alpha'])
                 
                 # Composite: Color where mask is white (active), Gray where mask is black
                 base = Image.composite(color_bg, gray_bg, active_mask_union)
        else:
            base = color_bg
        
        # Create a transparent overlay layer
        overlay = Image.new("RGBA", base.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        selected_rgb = ImageColor.getrgb(self.current_color)
        alpha_val = self.transparency_val

        # Store text positions to draw at the end
        text_positions = []
        
        # Check if we are in animation mode
        if self.is_animating and self.animation_route:
            # Draw both hand and foot masks synchronously from single route
            for route_name in self.animation_route_list:
                route_data = self.saved_routes.get(route_name)
                if not route_data:
                    continue
                
                route_masks = route_data['masks']
                route_rgb = ImageColor.getrgb(route_data['color'])
                
                # Separate masks by type and draw them synchronously
                hand_masks = []
                foot_masks = []
                
                # First pass: separate masks by type
                for mask_entry in route_masks:
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                        mask_type = mask_entry['type']
                    else:
                        fname = mask_entry
                        mask_type = route_data.get('type', 'hand')
                    
                    if mask_type == 'hand':
                        hand_masks.append(mask_entry)
                    else:
                        foot_masks.append(mask_entry)
                
                # Draw hand masks up to animation_index
                for i in range(min(self.animation_index + 1, len(hand_masks))):
                    mask_entry = hand_masks[i]
                    
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                    else:
                        fname = mask_entry
                    
                    # If this is the current blinking mask and blink is off, skip drawing
                    should_draw = True
                    if i == self.animation_index and not self.animation_blink_state:
                        should_draw = False
                    
                    if should_draw:
                        mask_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                        if mask_data:
                            bbox = mask_data['bbox']
                            if bbox:
                                mask_w = bbox[2] - bbox[0]
                                mask_h = bbox[3] - bbox[1]
                                
                                if mask_w > 0 and mask_h > 0:
                                    mask_crop = mask_data['alpha'].crop(bbox)
                                    solid_block = Image.new("RGBA", (mask_w, mask_h), route_rgb + (alpha_val,))
                                    overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
                                    
                                    cx = (bbox[0] + bbox[2]) // 2
                                    cy = (bbox[1] + bbox[3]) // 2
                                    text_positions.append((cx, cy, f"h-{i + 1}"))
                
                # Draw foot masks up to animation_index
                for i in range(min(self.animation_index + 1, len(foot_masks))):
                    mask_entry = foot_masks[i]
                    
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                    else:
                        fname = mask_entry
                    
                    # If this is the current blinking mask and blink is off, skip drawing
                    should_draw = True
                    if i == self.animation_index and not self.animation_blink_state:
                        should_draw = False
                    
                    if should_draw:
                        mask_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                        if mask_data:
                            bbox = mask_data['bbox']
                            if bbox:
                                mask_w = bbox[2] - bbox[0]
                                mask_h = bbox[3] - bbox[1]
                                
                                if mask_w > 0 and mask_h > 0:
                                    mask_crop = mask_data['alpha'].crop(bbox)
                                    solid_block = Image.new("RGBA", (mask_w, mask_h), route_rgb + (alpha_val,))
                                    overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
                                    
                                    cx = (bbox[0] + bbox[2]) // 2
                                    cy = (bbox[1] + bbox[3]) // 2
                                    text_positions.append((cx, cy, f"f-{i + 1}"))
                
                
                # Draw arrows for hand masks
                for i in range(min(self.animation_index, len(hand_masks) - 1)):
                    mask_entry1 = hand_masks[i]
                    mask_entry2 = hand_masks[i + 1]
                    
                    fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                    fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                    
                    mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                    mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                    
                    if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                        bbox1 = mask1['bbox']
                        bbox2 = mask2['bbox']
                        
                        cx1 = (bbox1[0] + bbox1[2]) // 2
                        cy1 = (bbox1[1] + bbox1[3]) // 2
                        cx2 = (bbox2[0] + bbox2[2]) // 2
                        cy2 = (bbox2[1] + bbox2[3]) // 2
                        self.draw_arrow(draw, cx1, cy1, cx2, cy2, route_rgb)
                
                # Note: Icons will be drawn at the end on final_img
                
                # Draw arrows for foot masks
                for i in range(min(self.animation_index, len(foot_masks) - 1)):
                    mask_entry1 = foot_masks[i]
                    mask_entry2 = foot_masks[i + 1]
                    
                    fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                    fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                    
                    mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                    mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                    
                    if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                        bbox1 = mask1['bbox']
                        bbox2 = mask2['bbox']
                        
                        cx1 = (bbox1[0] + bbox1[2]) // 2
                        cy1 = (bbox1[1] + bbox1[3]) // 2
                        cx2 = (bbox2[0] + bbox2[2]) // 2
                        cy2 = (bbox2[1] + bbox2[3]) // 2
                        self.draw_arrow(draw, cx1, cy1, cx2, cy2, route_rgb)
                
        elif self.show_route_arrows and self.animation_route:
            # Show completed routes with arrows (after animation) - both hand and foot
            for route_name in self.animation_route_list:
                route_data = self.saved_routes.get(route_name)
                if not route_data:
                    continue
                
                route_masks = route_data['masks']
                route_rgb = ImageColor.getrgb(route_data['color'])
                
                # Build separate counters for hand and foot
                hand_idx = 0
                foot_idx = 0
                
                # Draw all masks in route
                for mask_entry in route_masks:
                    # Handle both old format (string) and new format (dict)
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                        mask_type = mask_entry['type']
                    else:
                        fname = mask_entry
                        mask_type = route_data.get('type', 'hand')  # Fallback to route type
                    
                    mask_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                    if not mask_data:
                        continue
                    
                    bbox = mask_data['bbox']
                    if not bbox:
                        continue
                    
                    mask_w = bbox[2] - bbox[0]
                    mask_h = bbox[3] - bbox[1]
                    
                    if mask_w <= 0 or mask_h <= 0:
                        continue
                    
                    # Draw with route color
                    mask_crop = mask_data['alpha'].crop(bbox)
                    solid_block = Image.new("RGBA", (mask_w, mask_h), route_rgb + (alpha_val,))
                    overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
                    
                    # Add text position using mask's own type with separate counters
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    if mask_type == "hand":
                        hand_idx += 1
                        text_positions.append((cx, cy, f"h-{hand_idx}"))
                    else:
                        foot_idx += 1
                        text_positions.append((cx, cy, f"f-{foot_idx}"))
                
                # Draw arrows only between same type masks
                hand_masks = []
                foot_masks = []
                
                for mask_entry in route_masks:
                    if isinstance(mask_entry, dict):
                        mask_type = mask_entry['type']
                    else:
                        mask_type = route_data.get('type', 'hand')
                    
                    if mask_type == 'hand':
                        hand_masks.append(mask_entry)
                    else:
                        foot_masks.append(mask_entry)
                
                # Draw arrows for hand masks
                for i in range(len(hand_masks) - 1):
                    mask_entry1 = hand_masks[i]
                    mask_entry2 = hand_masks[i + 1]
                    
                    fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                    fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                    
                    mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                    mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                    
                    if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                        bbox1 = mask1['bbox']
                        bbox2 = mask2['bbox']
                        
                        cx1 = (bbox1[0] + bbox1[2]) // 2
                        cy1 = (bbox1[1] + bbox1[3]) // 2
                        cx2 = (bbox2[0] + bbox2[2]) // 2
                        cy2 = (bbox2[1] + bbox2[3]) // 2
                        
                        self.draw_arrow(draw, cx1, cy1, cx2, cy2, route_rgb)
                
                # Draw arrows for foot masks
                for i in range(len(foot_masks) - 1):
                    mask_entry1 = foot_masks[i]
                    mask_entry2 = foot_masks[i + 1]
                    
                    fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                    fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                    
                    mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                    mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                    
                    if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                        bbox1 = mask1['bbox']
                        bbox2 = mask2['bbox']
                        
                        cx1 = (bbox1[0] + bbox1[2]) // 2
                        cy1 = (bbox1[1] + bbox1[3]) // 2
                        cx2 = (bbox2[0] + bbox2[2]) // 2
                        cy2 = (bbox2[1] + bbox2[3]) // 2
                        
                        self.draw_arrow(draw, cx1, cy1, cx2, cy2, route_rgb)
        else:
            # Normal mode: draw all saved routes + current selection
            # First, draw all saved routes (in background)
            # Build separate counters for hand and foot
            hand_counter = {}
            foot_counter = {}
            
            for route_name, route_data in self.saved_routes.items():
                route_type = route_data.get('type', 'hand')
                if route_type == 'hand':
                    hand_counter[route_name] = len(hand_counter)
                else:
                    foot_counter[route_name] = len(foot_counter)
            
            for route_name, route_data in self.saved_routes.items():
                route_color_hex = route_data['color']
                route_masks = route_data['masks']
                route_type = route_data.get('type', 'hand')
                
                try:
                    route_rgb = ImageColor.getrgb(route_color_hex)
                except:
                    route_rgb = (128, 128, 128) # Fallback gray
                
                # Build separate counters for hand and foot
                hand_idx = 0
                foot_idx = 0
                
                for mask_entry in route_masks:
                    # Handle both old format (string) and new format (dict)
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                        mask_type = mask_entry['type']
                    else:
                        fname = mask_entry
                        mask_type = route_data.get('type', 'hand')  # Fallback to route type
                    
                    mask_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                    if not mask_data:
                        continue
                    
                    bbox = mask_data['bbox']
                    if not bbox:
                        continue
                    
                    mask_w = bbox[2] - bbox[0]
                    mask_h = bbox[3] - bbox[1]
                    
                    if mask_w <= 0 or mask_h <= 0:
                        continue
                    
                    # Draw with route color
                    mask_crop = mask_data['alpha'].crop(bbox)
                    solid_block = Image.new("RGBA", (mask_w, mask_h), route_rgb + (alpha_val,))
                    overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
                    
                    # Add text position using mask's own type with separate counters
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    if mask_type == "hand":
                        hand_idx += 1
                        text_positions.append((cx, cy, f"h-{hand_idx}"))
                    else:
                        foot_idx += 1
                        text_positions.append((cx, cy, f"f-{foot_idx}"))
            
            # Then, draw currently selected masks (overwrite if overlapping)
            # Build counters for hand and foot separately
            hand_idx = 0
            foot_idx = 0
            
            for mask_entry in self.selected_route_masks:
                # Handle both old format (string) and new format (dict)
                if isinstance(mask_entry, dict):
                    fname = mask_entry['filename']
                    mask_type = mask_entry['type']
                else:
                    fname = mask_entry
                    mask_type = 'hand'  # Default for old entries
                
                mask_data = next((m for m in self.manager.masks if m['filename'] == fname), None)
                if not mask_data:
                    continue
                
                bbox = mask_data['bbox']
                if not bbox:
                    continue
                
                # Selected route color - FORCE use selected color
                color = selected_rgb
                
                # Prepare text drawing info with type-specific numbering
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                
                if mask_type == 'hand':
                    hand_idx += 1
                    text_positions.append((cx, cy, f"h-{hand_idx}"))
                else:
                    foot_idx += 1
                    text_positions.append((cx, cy, f"f-{foot_idx}"))
                
                # Draw the mask
                mask_w = bbox[2] - bbox[0]
                mask_h = bbox[3] - bbox[1]
                
                if mask_w <= 0 or mask_h <= 0:
                    continue
                
                mask_crop = mask_data['alpha'].crop(bbox)
                solid_block = Image.new("RGBA", (mask_w, mask_h), color + (alpha_val,))
                overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
            
            # Draw merge mode masks
            for mask_data in self.manager.masks:
                fname = mask_data['filename']
                bbox = mask_data['bbox']
                
                if not bbox:
                    continue
                
                # Check if in merge selection
                if fname in self.merge_selected_masks:
                    # Merge mode: highlight selected masks in magenta
                    color = (255, 0, 255) # Magenta
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    text_positions.append((cx, cy, "M"))
                else:
                    # Random color - only if not in any saved route
                    if not any(fname in route_data['masks'] for route_data in self.saved_routes.values()):
                        color = self.manager.mask_colors.get(fname, (255, 255, 255))
                    else:
                        continue # Skip, already drawn above
                
                # Optimization: Create a solid color rect for the bbox size only, then apply mask crop
                # This avoids creating full-size buffer for every mask
                
                mask_w = bbox[2] - bbox[0]
                mask_h = bbox[3] - bbox[1]
                
                if mask_w <=0 or mask_h <= 0: continue
                
                # Crop the alpha mask to the bbox for efficiency
                mask_crop = mask_data['alpha'].crop(bbox)
                
                # Create solid color block matching the cropped size
                solid_block = Image.new("RGBA", (mask_w, mask_h), color + (alpha_val,))
                
                # Paste into overlay at bbox position using the cropped mask
                overlay.paste(solid_block, (bbox[0], bbox[1]), mask_crop)
        
        # 2. Draw all text labels on top
        for cx, cy, text in text_positions:
            draw.text((cx, cy), text, fill="white", stroke_fill="black", stroke_width=2, font=self.font)

        # Composite
        final_img = Image.alpha_composite(base, overlay)
        
        # 3. Draw animated icons on top of everything (if in animation mode)
        if self.is_animating and self.animation_route and self.animation_index > 0:
            # Get hand and foot masks for current route
            for route_name in self.animation_route_list:
                route_data = self.saved_routes.get(route_name)
                if not route_data:
                    continue
                
                route_masks = route_data['masks']
                
                # Separate masks by type
                hand_masks = []
                foot_masks = []
                
                for mask_entry in route_masks:
                    if isinstance(mask_entry, dict):
                        fname = mask_entry['filename']
                        mask_type = mask_entry['type']
                    else:
                        fname = mask_entry
                        mask_type = route_data.get('type', 'hand')
                    
                    if mask_type == 'hand':
                        hand_masks.append(mask_entry)
                    else:
                        foot_masks.append(mask_entry)
                
                # Draw animated hand icon on current arrow (sliding effect)
                if self.animation_index <= len(hand_masks):
                    arrow_idx = self.animation_index - 1
                    if arrow_idx < len(hand_masks) - 1:
                        mask_entry1 = hand_masks[arrow_idx]
                        mask_entry2 = hand_masks[arrow_idx + 1]
                        
                        fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                        fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                        
                        mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                        mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                        
                        if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                            bbox1 = mask1['bbox']
                            bbox2 = mask2['bbox']
                            
                            cx1 = (bbox1[0] + bbox1[2]) // 2
                            cy1 = (bbox1[1] + bbox1[3]) // 2
                            cx2 = (bbox2[0] + bbox2[2]) // 2
                            cy2 = (bbox2[1] + bbox2[3]) // 2
                            
                            # Calculate icon position based on animation progress
                            progress = self.animation_progress
                            icon_x = int(cx1 + (cx2 - cx1) * progress)
                            icon_y = int(cy1 + (cy2 - cy1) * progress)
                            
                            # Paste hand icon on final image (top layer)
                            icon_pos = (icon_x - 30, icon_y - 30)
                            final_img.paste(self.hand_icon, icon_pos, self.hand_icon)
                
                # Draw animated foot icon on current arrow (sliding effect)
                if self.animation_index <= len(foot_masks):
                    arrow_idx = self.animation_index - 1
                    if arrow_idx < len(foot_masks) - 1:
                        mask_entry1 = foot_masks[arrow_idx]
                        mask_entry2 = foot_masks[arrow_idx + 1]
                        
                        fname1 = mask_entry1['filename'] if isinstance(mask_entry1, dict) else mask_entry1
                        fname2 = mask_entry2['filename'] if isinstance(mask_entry2, dict) else mask_entry2
                        
                        mask1 = next((m for m in self.manager.masks if m['filename'] == fname1), None)
                        mask2 = next((m for m in self.manager.masks if m['filename'] == fname2), None)
                        
                        if mask1 and mask2 and mask1['bbox'] and mask2['bbox']:
                            bbox1 = mask1['bbox']
                            bbox2 = mask2['bbox']
                            
                            cx1 = (bbox1[0] + bbox1[2]) // 2
                            cy1 = (bbox1[1] + bbox1[3]) // 2
                            cx2 = (bbox2[0] + bbox2[2]) // 2
                            cy2 = (bbox2[1] + bbox2[3]) // 2
                            
                            # Calculate icon position based on animation progress
                            progress = self.animation_progress
                            icon_x = int(cx1 + (cx2 - cx1) * progress)
                            icon_y = int(cy1 + (cy2 - cy1) * progress)
                            
                            # Paste foot icon on final image (top layer)
                            icon_pos = (icon_x - 30, icon_y - 30)
                            final_img.paste(self.foot_icon, icon_pos, self.foot_icon)
        
        self.tk_img = ImageTk.PhotoImage(final_img)
        
        # Update existing image item or create new one
        if hasattr(self, 'canvas_img_id'):
             self.canvas.itemconfig(self.canvas_img_id, image=self.tk_img)
        else:
             self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def draw_arrow(self, draw, x1, y1, x2, y2, color):
        """Draw an arrow from (x1,y1) to (x2,y2)"""
        import math
        
        # Draw line
        draw.line([(x1, y1), (x2, y2)], fill=color + (255,), width=4)
        
        # Draw arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_length = 20
        arrow_angle = math.pi / 6
        
        # Arrow tip points
        p1_x = x2 - arrow_length * math.cos(angle - arrow_angle)
        p1_y = y2 - arrow_length * math.sin(angle - arrow_angle)
        p2_x = x2 - arrow_length * math.cos(angle + arrow_angle)
        p2_y = y2 - arrow_length * math.sin(angle + arrow_angle)
        
        draw.polygon([(x2, y2), (p1_x, p1_y), (p2_x, p2_y)], fill=color + (255,))

    def save_route(self):
        name = self.route_name.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a Route Name.")
            return
        
        if not self.selected_route_masks:
            messagebox.showwarning("No Selection", "Please select at least one mask.")
            return

        # Folder name: Name_Color_Type
        # Clean color hex string
        color_str = self.current_color.replace("#", "")
        route_type = self.route_type.get()
        folder_name = f"{name}_{color_str}_{route_type}"
        target_dir = os.path.join(self.manager.base_dir, folder_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        # Copy files
        try:
            for idx, mask_entry in enumerate(self.selected_route_masks):
                # Handle both old format (string) and new format (dict)
                if isinstance(mask_entry, dict):
                    fname = mask_entry['filename']
                    mask_type = mask_entry['type']
                else:
                    fname = mask_entry
                    mask_type = 'hand'  # Default for old entries
                
                # Add type prefix (h/f) to filename
                type_prefix = 'h' if mask_type == 'hand' else 'f'
                src = os.path.join(self.manager.base_dir, fname)
                dest_fname = f"{idx+1}-{type_prefix}-{fname}"
                dest = os.path.join(target_dir, dest_fname)
                shutil.copy2(src, dest)
            
            # Generate preview image with gray background and colored selected masks
            try:
                self.generate_route_preview(target_dir, name)
            except Exception as preview_error:
                print(f"Warning: Failed to generate preview image: {preview_error}")
            
            messagebox.showinfo("Success", f"Route saved to {folder_name}")
            self.refresh_route_list()
            self.load_saved_routes() # Reload saved routes
            self.clear_selection()
            self.route_name.set("")
            self.refresh_canvas() # Refresh to show the newly saved route
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save route: {e}")
    
    def generate_route_preview(self, route_dir, route_name):
        """Generate a full-resolution preview image with gray background and colored route holds"""
        # Load original background image (full resolution)
        bg_img = self.manager.original_bg.copy()
        original_size = bg_img.size
        
        # Convert to grayscale
        gray_bg = bg_img.convert("L").convert("RGB")
        
        # Create final image starting with gray background
        final_img = gray_bg.copy()
        
        # Get unique filenames from selected masks (remove duplicates)
        unique_filenames = set()
        for mask_entry in self.selected_route_masks:
            if isinstance(mask_entry, dict):
                fname = mask_entry['filename']
            else:
                fname = mask_entry
            unique_filenames.add(fname)
        
        # Process each unique mask
        for fname in unique_filenames:
            # Load original full-resolution mask
            mask_path = os.path.join(self.manager.base_dir, fname)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found: {mask_path}")
                continue
            
            mask_img = Image.open(mask_path).convert('L')
            
            # Ensure mask is same size as background
            if mask_img.size != original_size:
                mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)
            
            # Create binary mask (white pixels = hold area)
            mask_array = np.array(mask_img)
            mask_binary = mask_array > 50  # Threshold
            
            # Get the corresponding color area from original background
            bg_array = np.array(bg_img)
            final_array = np.array(final_img)
            
            # Replace gray pixels with color pixels where mask is active
            final_array[mask_binary] = bg_array[mask_binary]
            
            final_img = Image.fromarray(final_array)
        
        # Save the preview image
        preview_filename = f"{route_name}_preview.jpg"
        preview_path = os.path.join(route_dir, preview_filename)
        final_img.save(preview_path, "JPEG", quality=95)
        
        print(f"Preview image saved: {preview_path}")

    def refresh_route_list(self):
        self.route_listbox.delete(0, tk.END)
        dirs = [d for d in os.listdir(self.manager.base_dir) if os.path.isdir(os.path.join(self.manager.base_dir, d))]
        for d in dirs:
            self.route_listbox.insert(tk.END, d)

    def load_saved_routes(self):
        """Load all saved routes from subfolders"""
        self.saved_routes = {}
        dirs = [d for d in os.listdir(self.manager.base_dir) if os.path.isdir(os.path.join(self.manager.base_dir, d))]
        
        for route_folder in dirs:
            full_path = os.path.join(self.manager.base_dir, route_folder)
            
            # Parse folder name "RouteName_ColorHex_Type" or old format "RouteName_ColorHex"
            color_hex = "#FF0000" # Default
            route_type = "hand" # Default
            
            if "_" in route_folder:
                parts = route_folder.split("_")
                
                # Check if last part is type (hand/foot)
                if len(parts) >= 2 and parts[-1] in ["hand", "foot"]:
                    route_type = parts[-1]
                    # Color is second to last
                    if len(parts) >= 3:
                        color_candidate = parts[-2]
                    else:
                        color_candidate = parts[-1]
                else:
                    # Old format, no type specified
                    color_candidate = parts[-1]
                
                # Validate color
                if len(color_candidate) in [3, 6]:
                    try:
                        color_hex = f"#{color_candidate}"
                        ImageColor.getrgb(color_hex)
                    except:
                        pass
            
            # Load mask files from folder (sorted by index)
            try:
                files = sorted(os.listdir(full_path), key=lambda x: int(x.split('-')[0]) if '-' in x else 999)
                masks = []  # List of dicts with filename and type
                
                for f in files:
                    if '-' in f and f.endswith('.png'):
                        # Parse filename: "1-h-bg1_100.png" or old format "1-bg1_100.png"
                        parts = f.split('-')
                        
                        if len(parts) >= 3 and parts[1] in ['h', 'f']:
                            # New format with type marker
                            mask_type = 'hand' if parts[1] == 'h' else 'foot'
                            original_name = '-'.join(parts[2:])
                        elif len(parts) >= 2:
                            # Old format without type marker
                            mask_type = route_type  # Use folder type as fallback
                            original_name = '-'.join(parts[1:])
                        else:
                            continue
                        
                        # Verify it exists in our manager
                        if any(m['filename'] == original_name for m in self.manager.masks):
                            masks.append({
                                'filename': original_name,
                                'type': mask_type
                            })
                
                if masks:
                    self.saved_routes[route_folder] = {
                        'color': color_hex,
                        'masks': masks,
                        'type': route_type
                    }
                    print(f"Loaded route: {route_folder} ({route_type}) with {len(masks)} masks")
            except Exception as e:
                print(f"Error loading route {route_folder}: {e}")

    def on_route_select(self, event):
        # When a route is selected in the listbox, do we want to load it into the "Editor"?
        # Or just allow highlighting? For now, let's just enable the highlight button.
        pass

    def highlight_current_route(self):
        selection = self.route_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a route from the list first.")
            return
        
        route_folder = self.route_listbox.get(selection[0])
        
        if route_folder not in self.saved_routes:
            messagebox.showwarning("Route Not Found", f"Route {route_folder} not found.")
            return
        
        route_data = self.saved_routes[route_folder]
        
        # Single route folder contains both hand and foot masks
        # identified by 'h' or 'f' in the filename
        print(f"Animating route: {route_folder}")
        print(f"Total masks in route: {len(route_data['masks'])}")
        
        # Count hand and foot masks
        hand_count = 0
        foot_count = 0
        for mask_entry in route_data['masks']:
            if isinstance(mask_entry, dict):
                if mask_entry['type'] == 'hand':
                    hand_count += 1
                else:
                    foot_count += 1
        
        print(f"Hand masks: {hand_count}, Foot masks: {foot_count}")
        
        routes_to_animate = [route_folder]
        
        # Start animation with synchronized display
        self.is_animating = True
        self.animation_route_list = routes_to_animate
        self.animation_route_index = 0
        self.animation_route = routes_to_animate[0]
        self.animation_index = 0
        self.animation_blink_state = True
        self.animation_blink_count = 0
        
        # Disable controls during animation
        self.highlight_btn.config(state=tk.DISABLED, text="Animating...")
        
        # Start animation loop
        self.animate_route_synchronized()
    
    def animate_route_synchronized(self):
        """Animation loop for highlighting both hand and foot routes synchronously"""
        if not self.is_animating:
            return
        
        # Find max length - the maximum of hand or foot mask count
        # since we animate them in sync
        max_length = 0
        for route_name in self.animation_route_list:
            route_data = self.saved_routes.get(route_name)
            if route_data:
                route_masks = route_data['masks']
                
                # Count hand and foot masks separately
                hand_count = 0
                foot_count = 0
                for mask_entry in route_masks:
                    if isinstance(mask_entry, dict):
                        mask_type = mask_entry['type']
                    else:
                        mask_type = route_data.get('type', 'hand')
                    
                    if mask_type == 'hand':
                        hand_count += 1
                    else:
                        foot_count += 1
                
                # Max is the larger of the two
                max_length = max(max_length, hand_count, foot_count)
                print(f"Route {route_name}: hand={hand_count}, foot={foot_count}, max={max_length}")
        
        # Check if we've finished all masks
        if self.animation_index >= max_length:
            self.stop_animation()
            return
        
        # Blink logic: 4 blinks over 2 seconds = 250ms per blink
        if self.animation_blink_count < 8:  # 4 blinks = 8 state changes (on/off)
            self.animation_blink_state = not self.animation_blink_state
            self.animation_blink_count += 1
            self.refresh_canvas()
            self.root.after(250, self.animate_route_synchronized)
        else:
            # Done blinking, start sliding icons to next mask
            self.animation_blink_state = True
            self.animation_blink_count = 0
            
            # Animate icon sliding (10 steps over 500ms = 50ms per step)
            self.animate_icon_slide()
    
    def animate_icon_slide(self):
        """Animate icons sliding along arrows"""
        if not self.is_animating:
            return
        
        if self.animation_progress < 1.0:
            self.animation_progress += 0.1  # 10 steps
            self.refresh_canvas()
            self.root.after(50, self.animate_icon_slide)
        else:
            # Finished sliding, move to next mask
            self.animation_progress = 0.0
            self.animation_index += 1
            self.refresh_canvas()
            
            # Find max length
            max_length = 0
            for route_name in self.animation_route_list:
                route_data = self.saved_routes.get(route_name)
                if route_data:
                    route_masks = route_data['masks']
                    hand_count = sum(1 for m in route_masks if (m['type'] if isinstance(m, dict) else route_data.get('type', 'hand')) == 'hand')
                    foot_count = sum(1 for m in route_masks if (m['type'] if isinstance(m, dict) else route_data.get('type', 'hand')) == 'foot')
                    max_length = max(max_length, hand_count, foot_count)
            
            if self.animation_index < max_length:
                self.root.after(100, self.animate_route_synchronized)
            else:
                self.stop_animation()
    
    def animate_route(self):
        """Animation loop for highlighting route masks sequentially"""
        if not self.is_animating:
            return
        
        route_data = self.saved_routes.get(self.animation_route)
        if not route_data:
            self.stop_animation()
            return
        
        route_masks = route_data['masks']
        
        # Check if we've finished all masks in current route
        if self.animation_index >= len(route_masks):
            # Move to next route in the list
            self.animation_route_index += 1
            
            if self.animation_route_index >= len(self.animation_route_list):
                # Finished all routes
                self.stop_animation()
                return
            
            # Start next route
            self.animation_route = self.animation_route_list[self.animation_route_index]
            self.animation_index = 0
            self.animation_blink_count = 0
            self.animation_blink_state = True
            
            # Small pause before next route
            self.root.after(1000, self.animate_route)
            return
        
        # Blink logic: 4 blinks over 2 seconds = 250ms per blink
        if self.animation_blink_count < 8:  # 4 blinks = 8 state changes (on/off)
            self.animation_blink_state = not self.animation_blink_state
            self.animation_blink_count += 1
            self.refresh_canvas()
            self.root.after(250, self.animate_route)
        else:
            # Done blinking, move to next mask
            self.animation_blink_state = True
            self.animation_blink_count = 0
            self.animation_index += 1
            self.refresh_canvas()
            
            # Wait a bit before starting next mask (show arrow)
            if self.animation_index < len(route_masks):
                self.root.after(500, self.animate_route)
            else:
                # Check if there's another route
                self.animate_route()
    
    def stop_animation(self):
        """Stop animation and return to normal mode"""
        self.is_animating = False
        # Keep route displayed with arrows
        self.show_route_arrows = True
        self.animation_blink_state = False
        self.animation_blink_count = 0
        
        # Re-enable controls
        self.highlight_btn.config(state=tk.NORMAL, text="Highlight Route Masks")
        
        # Refresh canvas to show route with arrows
        self.refresh_canvas()
    
    def start_merge_mode(self):
        """Enter merge mode to select masks for merging"""
        if self.is_merge_mode:
            # Already in merge mode, perform merge
            if len(self.merge_selected_masks) < 2:
                messagebox.showwarning("Not Enough Masks", "Please select at least 2 masks to merge.")
                return
            
            # Perform merge and reset to allow new merge
            self.perform_merge()
            # After merge, enter merge mode again for next group
            self.is_merge_mode = True
            self.merge_selected_masks = []
            messagebox.showinfo("Merge Mode", "Merge complete! Click masks to select another group for merging, or click 'Merge Masks' again.")
        else:
            # Enter merge mode
            self.is_merge_mode = True
            self.merge_selected_masks = []
            messagebox.showinfo("Merge Mode", "Click on masks to select them for merging. Click 'Merge Masks' again when ready.")
            self.refresh_canvas()
    
    def perform_merge(self):
        """Merge selected masks into one"""
        if len(self.merge_selected_masks) < 2:
            return
        
        try:
            # Load all selected masks as full-size images
            merged_mask = None
            files_to_delete = []
            
            for fname in self.merge_selected_masks:
                mask_path = os.path.join(self.manager.base_dir, fname)
                img = Image.open(mask_path).convert('L')
                files_to_delete.append(mask_path)
                
                if merged_mask is None:
                    merged_mask = np.array(img, dtype=np.uint16)
                else:
                    merged_mask = np.maximum(merged_mask, np.array(img, dtype=np.uint16))
            
            # Convert back to image
            merged_img = Image.fromarray(merged_mask.astype(np.uint8), mode='L')
            
            # Generate new filename
            base_name = self.merge_selected_masks[0].replace('.png', '')
            new_name = f"{base_name}_merged.png"
            counter = 1
            while os.path.exists(os.path.join(self.manager.base_dir, new_name)):
                new_name = f"{base_name}_merged{counter}.png"
                counter += 1
            
            # Save merged mask
            new_path = os.path.join(self.manager.base_dir, new_name)
            merged_img.save(new_path)
            
            print(f"Merged {len(self.merge_selected_masks)} masks into {new_name}")
            
            # Delete old mask files
            for old_file in files_to_delete:
                try:
                    os.remove(old_file)
                    print(f"Deleted {os.path.basename(old_file)}")
                except Exception as e:
                    print(f"Warning: Could not delete {old_file}: {e}")
            
            # Reload masks
            messagebox.showinfo("Success", f"Masks merged into {new_name}. Old masks deleted.")
            self.manager.masks = []
            self.manager.load_masks()
            
            # Generate random color for new mask
            self.manager.mask_colors[new_name] = tuple(np.random.randint(0, 256, 3))
            
            # Clear merge selection but stay in merge mode (handled by start_merge_mode)
            self.merge_selected_masks = []
            
            self.refresh_canvas()
            
        except Exception as e:
            messagebox.showerror("Merge Error", f"Failed to merge masks: {e}")
            print(f"Merge error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Create root window for folder selection
    root = tk.Tk()
    root.withdraw()  # Hide the main window initially
    
    # Ask user to select folder containing background and masks
    folder_path = filedialog.askdirectory(
        title="Select folder containing background (jpg) and masks (png)"
    )
    
    if not folder_path:
        messagebox.showwarning("No Folder Selected", "No folder selected. Exiting.")
        exit()
    
    # Show the main window
    root.deiconify()
    root.geometry("1400x950") # Adjust as needed
    
    try:
        app = TaggingApp(root, folder_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize application:\n{str(e)}")
        exit()
    
    root.mainloop()
