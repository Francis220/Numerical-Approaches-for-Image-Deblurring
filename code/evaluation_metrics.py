import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_mse(original, reconstructed):
    """
    Compute Mean Squared Error.
    
    Args:
        original: original image
        reconstructed: reconstructed image
    
    Returns:
        MSE value
    """
    return np.mean((original - reconstructed) ** 2)

def compute_psnr(original, reconstructed, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        original: original image
        reconstructed: reconstructed image
        data_range: range of data (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    return psnr(original, reconstructed, data_range=data_range)

def compute_ssim(original, reconstructed, data_range=1.0):
    """
    Compute Structural Similarity Index.
    
    Args:
        original: original image
        reconstructed: reconstructed image
        data_range: range of data
    
    Returns:
        SSIM value
    """
    return ssim(original, reconstructed, data_range=data_range)

def compute_relative_error(original, reconstructed):
    """
    Compute relative reconstruction error.
    
    Args:
        original: original image
        reconstructed: reconstructed image
    
    Returns:
        Relative error
    """
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

def compute_all_metrics(original, reconstructed):
    """
    Compute all quality metrics.
    
    Args:
        original: original image
        reconstructed: reconstructed image
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'mse': compute_mse(original, reconstructed),
        'psnr': compute_psnr(original, reconstructed),
        'ssim': compute_ssim(original, reconstructed),
        'relative_error': compute_relative_error(original, reconstructed)
    }