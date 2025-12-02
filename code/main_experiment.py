import numpy as np
import os
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("blur_generation", os.path.join(os.path.dirname(__file__), "blur.generation.py"))
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

import tikhonov_deblur as td
import tsvd_deblur as tsvd
import parameter_selection as ps
import evaluation_metrics as em
import visualization as vis
import compare_methods as cm
import blur_operators as bo

def setup_directories():
    """
    Create necessary directory structure for the project.
    """
    directories = [
        'data/original',
        'data/blurred',
        'data/deblurred',
        'results/figures',
        'results/metrics'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_test_data():
    """
    Generate synthetic test images and blur configurations.
    """
    print("Generating test data...")
    
    blur_configs = [
        {'type': 'gaussian', 'size': 15, 'sigma': 1.0, 'snr': 40},
        {'type': 'gaussian', 'size': 15, 'sigma': 2.0, 'snr': 30},
        {'type': 'gaussian', 'size': 21, 'sigma': 3.0, 'snr': 30},
        {'type': 'gaussian', 'size': 25, 'sigma': 5.0, 'snr': 20},
        {'type': 'motion', 'length': 15, 'angle': 0, 'snr': 30},
        {'type': 'motion', 'length': 15, 'angle': 45, 'snr': 30},
        {'type': 'motion', 'length': 21, 'angle': 90, 'snr': 20},
    ]
    
    bg.generate_blurred_dataset('data/original', 'data/blurred', blur_configs)
    
    print("Test data generated successfully!")

def test_tikhonov_method():
    """
    Test Tikhonov regularization with parameter selection.
    """
    print("\n=== Testing Tikhonov Regularization ===")
    
    image = bg.load_image_grayscale('data/original/cameraman.png')
    kernel = bg.create_gaussian_kernel(15, 2.0)
    blurred = bg.apply_blur(image, kernel)
    blurred_noisy = bg.add_gaussian_noise(blurred, 30)
    
    lambda_range = np.logspace(-6, -1, 20)
    
    print("Computing L-curve...")
    image_shape = image.shape
    b = blurred_noisy.flatten()
    A = bo.create_convolution_matrix(kernel, image_shape, boundary='periodic')
    
    lcurve_data = ps.compute_lcurve(A, b, lambda_range)
    optimal_lambda = ps.find_lcurve_corner(lcurve_data)
    
    print(f"Optimal lambda: {optimal_lambda:.2e}")
    
    vis.plot_lcurve(lcurve_data, optimal_lambda, save_path='results/figures/lcurve.png')
    
    print("Deblurring with optimal parameter...")
    restored = td.tikhonov_fft(blurred_noisy, kernel, optimal_lambda)
    restored = np.clip(restored, 0, 1)
    
    metrics = em.compute_all_metrics(image, restored)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"MSE: {metrics['mse']:.6f}")
    
    bg.save_image(restored, 'data/deblurred/tikhonov_cameraman.png')

def test_tsvd_method():
    """
    Test TSVD method with truncation parameter selection.
    """
    print("\n=== Testing TSVD Method ===")
    
    image = bg.load_image_grayscale('data/original/cameraman.png')
    kernel = bg.create_gaussian_kernel(15, 2.0)
    blurred = bg.apply_blur(image, kernel)
    blurred_noisy = bg.add_gaussian_noise(blurred, 30)
    
    print("Analyzing singular value decay...")
    image_shape = image.shape
    b = blurred_noisy.flatten()
    A = bo.create_convolution_matrix(kernel, image_shape, boundary='periodic')
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    vis.plot_singular_values(s, k_truncate=1000, save_path='results/figures/singular_values.png')
    
    print("Testing different truncation levels...")
    k_range = [100, 500, 1000, 2000, 5000]
    
    for k in k_range:
        if k < len(s):
            restored = tsvd.tsvd_deblur_image(blurred_noisy, kernel, k, method='randomized')
            restored = np.clip(restored, 0, 1)
            metrics = em.compute_all_metrics(image, restored)
            print(f"k={k}: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
    
    optimal_k = 1000
    restored = tsvd.tsvd_deblur_image(blurred_noisy, kernel, optimal_k, method='randomized')
    restored = np.clip(restored, 0, 1)
    
    bg.save_image(restored, 'data/deblurred/tsvd_cameraman.png')

def run_comparative_study():
    """
    Run comprehensive comparative study of both methods.
    """
    print("\n=== Running Comparative Study ===")
    
    blur_configs = [
        {'blur_type': 'gaussian', 'kernel_size': 15, 'sigma': 2.0, 'snr': 40, 'lambda': 0.0005, 'k_truncate': 1500},
        {'blur_type': 'gaussian', 'kernel_size': 15, 'sigma': 2.0, 'snr': 30, 'lambda': 0.001, 'k_truncate': 1000},
        {'blur_type': 'gaussian', 'kernel_size': 21, 'sigma': 3.0, 'snr': 30, 'lambda': 0.002, 'k_truncate': 800},
        {'blur_type': 'gaussian', 'kernel_size': 21, 'sigma': 3.0, 'snr': 20, 'lambda': 0.005, 'k_truncate': 500},
        {'blur_type': 'motion', 'length': 15, 'angle': 0, 'snr': 30, 'lambda': 0.001, 'k_truncate': 1000},
        {'blur_type': 'motion', 'length': 15, 'angle': 45, 'snr': 30, 'lambda': 0.001, 'k_truncate': 1000},
    ]
    
    results_df = cm.run_experiment_suite(
        'data/original/cameraman.png',
        blur_configs,
        'results'
    )
    
    print("\n=== Summary Statistics ===")
    print(results_df.groupby('method')[['psnr', 'ssim', 'runtime']].mean())

def main():
    """
    Main execution function for the image deblurring project.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    
    print("=" * 60)
    print("Image Deblurring Using Regularization and Spectral Methods")
    print("=" * 60)
    
    setup_directories()
    
    test_tikhonov_method()
    
    test_tsvd_method()
    
    run_comparative_study()
    
    print("\n" + "=" * 60)
    print("Experiments completed successfully!")
    print("Results saved in 'results/' directory")
    print("=" * 60)

if __name__ == "__main__":
    main()