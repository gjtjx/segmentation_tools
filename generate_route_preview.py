import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw
import numpy as np

class RoutePreviewGenerator:
    def __init__(self, root, base_dir):
        self.root = root
        self.root.title("Route Preview Generator")
        self.root.geometry("500x600")
        
        self.base_dir = base_dir
        self.saved_routes_dir = base_dir
        
        # Find background jpg file
        self.bg_path = self.find_background()
        
        # Set output directory
        self.output_dir = os.path.join(base_dir, "route_previews")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load available routes
        self.routes = self.load_saved_routes()
        
        self.setup_ui()
    
    def find_background(self):
        """Find the first jpg file in the directory to use as background"""
        files = os.listdir(self.base_dir)
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not jpg_files:
            raise FileNotFoundError(f"No JPG background image found in {self.base_dir}")
        
        # Use the first jpg file found
        bg_path = os.path.join(self.base_dir, jpg_files[0])
        print(f"Found background image: {jpg_files[0]}")
        return bg_path
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Generate Route Preview with Grayed Background", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        info_text = ("Select a route from the list below.\n"
                    "The script will generate a full-resolution image where:\n"
                    "• Route holds are shown in color\n"
                    "• Everything else is converted to grayscale")
        info_label = tk.Label(self.root, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=5, padx=10)
        
        # Route list
        tk.Label(self.root, text="Available Routes:").pack(anchor="w", padx=10, pady=(10, 0))
        
        list_frame = tk.Frame(self.root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.route_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.route_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.route_listbox.yview)
        
        # Populate route list
        if self.routes:
            for route_name in sorted(self.routes.keys()):
                route_info = self.routes[route_name]
                num_masks = len(route_info['masks'])
                self.route_listbox.insert(tk.END, f"{route_name} ({num_masks} holds)")
        else:
            self.route_listbox.insert(tk.END, "No saved routes found")
        
        # Progress bar
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Generate button
        self.generate_btn = tk.Button(self.root, text="Generate Preview", 
                                      command=self.generate_preview, 
                                      bg="lightblue", font=("Arial", 11, "bold"))
        self.generate_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready", fg="green")
        self.status_label.pack(pady=5)
        
    def load_saved_routes(self):
        """Load all saved routes from the directory"""
        routes = {}
        
        # Look for subdirectories in BG_DIR (each represents a saved route)
        if not os.path.exists(self.saved_routes_dir):
            return routes
            
        for item in os.listdir(self.saved_routes_dir):
            item_path = os.path.join(self.saved_routes_dir, item)
            if os.path.isdir(item_path) and item != "route_previews":
                # This is a route folder
                route_name = item
                
                # Get all mask files in this folder
                mask_files = [f for f in os.listdir(item_path) if f.endswith('.png')]
                
                if mask_files:
                    # Parse mask files to extract type information
                    masks = []
                    for mask_file in mask_files:
                        # Expected format: "idx-type-originalname.png"
                        parts = mask_file.split('-', 2)
                        if len(parts) >= 3:
                            mask_type = parts[1]  # 'h' or 'f'
                            full_type = 'hand' if mask_type == 'h' else 'foot'
                        else:
                            full_type = 'hand'  # default
                        
                        masks.append({
                            'filename': mask_file,
                            'type': full_type,
                            'path': os.path.join(item_path, mask_file)
                        })
                    
                    routes[route_name] = {
                        'masks': masks,
                        'folder': item_path
                    }
        
        return routes
    
    def generate_preview(self):
        """Generate the preview image for selected route"""
        selection = self.route_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a route first.")
            return
            
        # Get selected route name
        selected_text = self.route_listbox.get(selection[0])
        route_name = selected_text.split(" (")[0]  # Remove the count part
        
        if route_name not in self.routes:
            messagebox.showerror("Error", f"Route '{route_name}' not found.")
            return
        
        # Disable button and show progress
        self.generate_btn.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_label.config(text="Generating...", fg="orange")
        self.root.update()
        
        try:
            # Load original background image (full resolution)
            bg_img = Image.open(self.bg_path).convert("RGB")
            original_size = bg_img.size  # (4000, 3000)
            print(f"Original background size: {original_size}")
            
            # Convert to grayscale
            gray_bg = bg_img.convert("L").convert("RGB")
            
            # Create final image starting with gray background
            final_img = gray_bg.copy()
            
            # Get route masks
            route_info = self.routes[route_name]
            masks = route_info['masks']
            
            print(f"Processing {len(masks)} masks for route '{route_name}'")
            
            # Process each mask in the route
            for idx, mask_info in enumerate(masks):
                mask_filename = mask_info['filename']
                
                # Find the original mask file in bg1_masks directory
                # The saved route contains copies, but we need original mask location
                # Extract original filename from the saved name (format: idx-type-originalname.png)
                parts = mask_filename.split('-', 2)
                if len(parts) >= 3:
                    original_mask_name = parts[2]  # Get the original filename part
                else:
                    original_mask_name = mask_filename
                
                original_mask_path = os.path.join(self.base_dir, original_mask_name)
                
                if not os.path.exists(original_mask_path):
                    print(f"Warning: Original mask not found: {original_mask_path}")
                    continue
                
                # Load original full-resolution mask
                mask_img = Image.open(original_mask_path).convert('L')
                
                # Ensure mask is same size as background
                if mask_img.size != original_size:
                    print(f"Warning: Mask size {mask_img.size} differs from background {original_size}")
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
                
                print(f"Processed {idx+1}/{len(masks)}: {original_mask_name}")
            
            # Save the final image
            output_filename = f"{route_name}_preview.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            final_img.save(output_path, "JPEG", quality=95)
            
            self.status_label.config(text=f"Saved: {output_filename}", fg="green")
            print(f"Preview saved to: {output_path}")
            
            messagebox.showinfo("Success", 
                              f"Preview generated successfully!\n\n"
                              f"Saved to:\n{output_path}\n\n"
                              f"Resolution: {original_size[0]}x{original_size[1]}")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Failed to generate preview:\n{str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Re-enable button and stop progress
            self.progress_bar.stop()
            self.generate_btn.config(state=tk.NORMAL)
            if self.status_label.cget("text").startswith("Error"):
                self.status_label.config(text="Ready", fg="green")


if __name__ == "__main__":
    # Create root window for folder selection
    root = tk.Tk()
    root.withdraw()  # Hide the main window initially
    
    # Ask user to select folder containing background and route folders
    folder_path = filedialog.askdirectory(
        title="Select folder containing background (jpg) and route folders"
    )
    
    if not folder_path:
        messagebox.showwarning("No Folder Selected", "No folder selected. Exiting.")
        exit()
    
    # Show the main window
    root.deiconify()
    
    try:
        app = RoutePreviewGenerator(root, folder_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize application:\n{str(e)}")
        import traceback
        traceback.print_exc()
        exit()
    
    root.mainloop()
