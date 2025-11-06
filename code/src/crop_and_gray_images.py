import os
from PIL import Image

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw')) # Get absolute path of source directory 
DST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/grey')) # Get absolute path of destination directory 

def process_image(src_path, dst_path): # Process a single image from source to destination 
    try:
        img = Image.open(src_path) # Open the image file 
        w, h = img.size # Get width and height of the image 
        if w < h: # If width is less than height 
            new_w = 128 # Set new width to 128 
            new_h = int(h * (128 / w)) # Calculate new height to maintain aspect ratio 
        else: # If height is less than or equal to width 
            new_h = 128 # Set new height to 128 
            new_w = int(w * (128 / h)) # Calculate new width to maintain aspect ratio 
        img = img.resize((new_w, new_h), Image.LANCZOS) # Resize image with high-quality downsampling 
        img = img.crop((0, 0, 128, 128)) # Crop the image from top-left to 128x128 
        img = img.convert('L') # Convert image to grayscale 
        img.save(dst_path) # Save the processed image to destination 
    except Exception as e: # Handle any exception during processing 
        print(f"An error occurred while processing the image {src_path}: {e}") # Print error message if exception occurs 

def process_folder(src_folder, dst_folder): # Process all images in a folder 
    if not os.path.exists(dst_folder): # If destination folder does not exist 
        os.makedirs(dst_folder) # Create the destination folder 
    for fname in os.listdir(src_folder): # Iterate over all files in the source folder 
        src_path = os.path.join(src_folder, fname) # Get full path of source file 
        dst_path = os.path.join(dst_folder, fname) # Get full path of destination file 
        if os.path.isfile(src_path) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')): # Check if file is an image 
            process_image(src_path, dst_path) # Process the image file 

def main():
    for subdir in os.listdir(SRC_ROOT): # Iterate over all subdirectories in source root 
        src_subdir = os.path.join(SRC_ROOT, subdir) # Get full path of source subdirectory 
        dst_subdir = os.path.join(DST_ROOT, subdir) # Get full path of destination subdirectory 
        if os.path.isdir(src_subdir): # Check if source subdirectory is a directory 
            print(f"Processing Folder: {subdir}") # Print the name of the folder being processed 
            process_folder(src_subdir, dst_subdir) # Process all images in the subdirectory 

if __name__ == '__main__':
    main()
