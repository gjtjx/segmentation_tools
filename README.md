# segmentation_tools
segmentation_tools：hand label and sam-hq segmentation
open datapath   --images --labels(option)
key: S brush | E erase | F pologon add | G pologon delete|P AI piont (left mouse positive |right mouse negative) | R reset |C change mask/source image/ctrl+Z reback|ctrl+Y redo|A next|D last|T clear

you can set brush size and rotate angle    
tool automatic save result when A / D,save a png
mask in dir(--labels)

exe for windows
PyInstaller --onefile --windowed image_annotation_tool.py
