import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def drpe_encrypt(img_array, phase1, phase2):  # Encrypts an image using DRPE algorithm 
    img_mod = img_array * np.exp(1j * phase1)  # Modulate image with spatial phase mask 
    img_fft = np.fft.fft2(img_mod)  # Apply 2D Fourier transform to modulated image 
    img_fft_mod = img_fft * np.exp(1j * phase2)  # Modulate Fourier spectrum with frequency phase mask 
    encrypted = np.fft.ifft2(img_fft_mod)  # Apply inverse Fourier transform to get encrypted image 
    return encrypted  # Return the complex-valued encrypted image 

def save_encrypted_image(enc_img, save_path):  # Saves encrypted image as a .npy file 
    np.save(save_path, enc_img)  # Save the numpy array to disk 

def save_visualization(enc_img, save_path_prefix):  # Save magnitude and phase visualizations 
    mag = np.abs(enc_img)  # Compute magnitude of encrypted image 
    mag = (mag / mag.max() * 255).astype(np.uint8)  # Normalize magnitude to [0,255] and convert to uint8 
    Image.fromarray(mag).save(save_path_prefix + '_mag.png')  # Save magnitude as PNG image 

    phase = np.angle(enc_img)  # Compute phase of encrypted image 
    phase = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)  # Normalize phase to [0,255] 
    # Image.fromarray(phase).save(save_path_prefix + '_phase.png')  # (Commented out) Save phase as PNG image 

def load_image(img_path):  # Loads an image and converts it to normalized grayscale array 
    img = Image.open(img_path).convert('L')  # Open image and convert to grayscale 
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0,1] 
    return img  # Return normalized image array 

def generate_phase(shape, seed=None):  # Generates a random phase mask 
    rng = np.random.default_rng(seed)  # Initialize random number generator with seed 
    return rng.uniform(0, 2 * np.pi, size=shape)  # Generate random values in [0, 2pi) 

def main(
    src_root = '../../data/128px_grey',
    dst_root = '../../data/drpe_encrypted',
    phase1_seed = 42,
    phase2_seed = 123
):
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), src_root))  # Get absolute path for source root 
    dst_root = os.path.abspath(os.path.join(os.path.dirname(__file__), dst_root))  # Get absolute path for destination root 
    if not os.path.exists(dst_root):  # Check if destination directory exists 
        os.makedirs(dst_root)  # Create destination directory if it does not exist 

    sample_class = os.listdir(src_root)[0]  # Get the first class folder name 
    sample_img_path = os.path.join(src_root, sample_class, os.listdir(os.path.join(src_root, sample_class))[0])  # Get a sample image path 
    img_shape = load_image(sample_img_path).shape  # Load sample image to determine shape 
    phase1 = generate_phase(img_shape, phase1_seed)  # Generate first random phase mask 
    phase2 = generate_phase(img_shape, phase2_seed)  # Generate second random phase mask 

    np.save(os.path.join(dst_root, 'phase1.npy'), phase1)  # Save first phase mask for decryption 
    np.save(os.path.join(dst_root, 'phase2.npy'), phase2)  # Save second phase mask for decryption 

    for cls in tqdm(os.listdir(src_root), desc='Classes'):  # Iterate over class folders 
        src_cls_dir = os.path.join(src_root, cls)  # Source directory for current class 
        dst_cls_dir = os.path.join(dst_root, cls)  # Destination directory for current class 
        if not os.path.isdir(src_cls_dir):  # Skip if not a directory 
            continue  # Continue to next class 
        if not os.path.exists(dst_cls_dir):  # Check if destination class directory exists 
            os.makedirs(dst_cls_dir)  # Create destination class directory if needed 
        for fname in tqdm(os.listdir(src_cls_dir), desc=cls, leave=False):  # Iterate over files in class 
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Skip non-image files 
                continue  # Continue to next file 
            src_img_path = os.path.join(src_cls_dir, fname)  # Path to source image 
            dst_img_path = os.path.join(dst_cls_dir, fname + '.npy')  # Path to save encrypted image 
            img = load_image(src_img_path)  # Load and normalize image 
            enc_img = drpe_encrypt(img, phase1, phase2)  # Encrypt image using DRPE 
            # save_encrypted_image(enc_img, dst_img_path)  # (Commented out) Save encrypted image as .npy 

            save_path_prefix = os.path.join(dst_cls_dir, fname)  # Prefix for visualization files 
            save_visualization(enc_img, save_path_prefix)  # Save magnitude visualization 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DRPE Encrypt Images')
    parser.add_argument('--src_root', type=str, default='../../data/128px_grey', help='Source image root')
    parser.add_argument('--dst_root', type=str, default='../../data/drpe_encrypted', help='Encrypted image root')
    parser.add_argument('--phase1_seed', type=int, default=42, help='Seed for spatial phase')
    parser.add_argument('--phase2_seed', type=int, default=123, help='Seed for frequency phase')
    args = parser.parse_args()
    main(args.src_root, args.dst_root, args.phase1_seed, args.phase2_seed)
