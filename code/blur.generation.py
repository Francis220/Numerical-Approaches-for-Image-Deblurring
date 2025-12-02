import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import cv2
from PIL import Image
import os

def create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian blur kernel.
    
    Args:
        size: kernel size (should be odd)
        sigma: standard deviation of Gaussian
    
    Returns:
        Normalized Gaussian kernel
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def create_motion_kernel(length, angle):
    """
    Create a motion blur kernel.
    
    Args:
        length: length of motion blur in pixels
        angle: angle of motion in degrees
    
    Returns:
        Motion blur kernel
    """
    kernel = np.zeros((length, length))
    center = length // 2
    angle_rad = np.deg2rad(angle)
    
    for i in range(length):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    return kernel / np.sum(kernel)

def apply_blur(image, kernel):
    """
    Apply blur kernel to image using convolution.
    
    Args:
        image: input image (2D array)
        kernel: blur kernel
    
    Returns:
        Blurred image
    """
    return convolve2d(image, kernel, mode='same', boundary='wrap')

def add_gaussian_noise(image, snr_db):
    """
    Add Gaussian noise to image based on SNR.
    
    Args:
        image: input image
        snr_db: signal-to-noise ratio in decibels
    
    Returns:
        Noisy image
    """
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    return image + noise

def load_image_grayscale(filepath):
    """
    Load image and convert to grayscale, normalize to [0, 1].
    
    Args:
        filepath: path to image file
    
    Returns:
        Grayscale image as float array
    """
    img = Image.open(filepath).convert('L')
    max_side = 64
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BICUBIC)
    return np.array(img, dtype=np.float64) / 255.0

def save_image(image, filepath):
    """
    Save image array to file.
    
    Args:
        image: image array (values in [0, 1])
        filepath: output path
    """
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filepath)

def generate_blurred_dataset(input_dir, output_dir, blur_configs):
    """
    Generate dataset of blurred images with various configurations.
    
    Args:
        input_dir: directory containing original images
        output_dir: directory to save blurred images
        blur_configs: list of dictionaries with blur parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, img_file)
            img = load_image_grayscale(img_path)
            img_name = os.path.splitext(img_file)[0]
            
            for config in blur_configs:
                blur_type = config['type']
                
                if blur_type == 'gaussian':
                    kernel = create_gaussian_kernel(config['size'], config['sigma'])
                    suffix = f"_gauss_s{config['sigma']}"
                elif blur_type == 'motion':
                    kernel = create_motion_kernel(config['length'], config['angle'])
                    suffix = f"_motion_l{config['length']}_a{config['angle']}"
                
                blurred = apply_blur(img, kernel)
                
                if 'snr' in config:
                    blurred = add_gaussian_noise(blurred, config['snr'])
                    suffix += f"_snr{config['snr']}"
                
                output_path = os.path.join(output_dir, f"{img_name}{suffix}.png")
                save_image(blurred, output_path)