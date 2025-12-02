import numpy as np
import time
import importlib.util
import sys
import os

# Import blur.generation module
spec = importlib.util.spec_from_file_location("blur_generation", os.path.join(os.path.dirname(__file__), "blur.generation.py"))
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

import tikhonov_deblur as td
import tsvd_deblur as tsvd
import evaluation_metrics as em
import visualization as vis
import blur_operators as bo

def compare_deblurring_methods(original_image, kernel, noise_snr, lambda_reg, k_truncate):
    """
    Compare Tikhonov and TSVD deblurring methods.
    
    Args:
        original_image: clean image
        kernel: blur kernel
        noise_snr: SNR for added noise (in dB)
        lambda_reg: Tikhonov regularization parameter
        k_truncate: TSVD truncation level
    
    Returns:
        Dictionary with results for both methods
    """
    blurred = bg.apply_blur(original_image, kernel)
    blurred_noisy = bg.add_gaussian_noise(blurred, noise_snr)
    
    results = {}
    
    start_time = time.time()
    tikhonov_result = td.tikhonov_fft(blurred_noisy, kernel, lambda_reg)
    tikhonov_time = time.time() - start_time
    
    tikhonov_result = np.clip(tikhonov_result, 0, 1)
    
    results['Tikhonov'] = {
        'image': tikhonov_result,
        'metrics': em.compute_all_metrics(original_image, tikhonov_result),
        'runtime': tikhonov_time,
        'parameter': lambda_reg
    }
    
    start_time = time.time()
    tsvd_result = tsvd.tsvd_deblur_image(blurred_noisy, kernel, k_truncate, method='randomized')
    tsvd_time = time.time() - start_time
    
    tsvd_result = np.clip(tsvd_result, 0, 1)
    
    results['TSVD'] = {
        'image': tsvd_result,
        'metrics': em.compute_all_metrics(original_image, tsvd_result),
        'runtime': tsvd_time,
        'parameter': k_truncate
    }
    
    results['blurred'] = blurred_noisy
    results['original'] = original_image
    
    return results

def run_experiment_suite(image_path, blur_configs, output_dir):
    """
    Run comprehensive experiments with multiple configurations.
    
    Args:
        image_path: path to test image
        blur_configs: list of blur configuration dictionaries
        output_dir: directory to save results
    """
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    original = bg.load_image_grayscale(image_path)
    
    all_results = []
    
    for config_idx, config in enumerate(blur_configs):
        print(f"Running experiment {config_idx + 1}/{len(blur_configs)}")
        
        if config['blur_type'] == 'gaussian':
            kernel = bg.create_gaussian_kernel(config['kernel_size'], config['sigma'])
        elif config['blur_type'] == 'motion':
            kernel = bg.create_motion_kernel(config['length'], config['angle'])
        
        results = compare_deblurring_methods(
            original, 
            kernel, 
            config['snr'],
            config['lambda'],
            config['k_truncate']
        )
        
        experiment_name = f"exp_{config_idx}"
        
        for method in ['Tikhonov', 'TSVD']:
            result_dict = {
                'experiment': experiment_name,
                'blur_type': config['blur_type'],
                'method': method,
                'mse': results[method]['metrics']['mse'],
                'psnr': results[method]['metrics']['psnr'],
                'ssim': results[method]['metrics']['ssim'],
                'runtime': results[method]['runtime'],
                'parameter': results[method]['parameter']
            }
            
            if config['blur_type'] == 'gaussian':
                result_dict['sigma'] = config['sigma']
            else:
                result_dict['length'] = config['length']
                result_dict['angle'] = config['angle']
            
            result_dict['snr'] = config['snr']
            
            all_results.append(result_dict)
        
        recon_dict = {
            'Tikhonov': results['Tikhonov']['image'],
            'TSVD': results['TSVD']['image']
        }
        
        vis.plot_comparison(
            results['original'],
            results['blurred'],
            recon_dict,
            save_path=os.path.join(output_dir, f"{experiment_name}_comparison.png")
        )
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'experiment_results.csv'), index=False)
    
    print(f"Results saved to {output_dir}")
    
    return df