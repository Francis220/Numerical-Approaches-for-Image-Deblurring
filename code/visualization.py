import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_comparison(original, blurred, reconstructed_dict, titles=None, save_path=None):
    """
    Plot comparison of original, blurred, and reconstructed images.
    
    Args:
        original: original image
        blurred: blurred image
        reconstructed_dict: dictionary of {method_name: reconstructed_image}
        titles: optional custom titles
        save_path: path to save figure
    """
    n_methods = len(reconstructed_dict)
    n_cols = 2 + n_methods
    
    fig = plt.figure(figsize=(4*n_cols, 5))
    
    plt.subplot(1, n_cols, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=1)
    plt.title('Original', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, n_cols, 2)
    plt.imshow(blurred, cmap='gray', vmin=0, vmax=1)
    plt.title('Blurred', fontsize=12)
    plt.axis('off')
    
    for idx, (method_name, recon_img) in enumerate(reconstructed_dict.items()):
        plt.subplot(1, n_cols, 3 + idx)
        plt.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        plt.title(method_name, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_lcurve(lcurve_data, corner_lambda=None, save_path=None):
    """
    Plot L-curve for Tikhonov regularization.
    
    Args:
        lcurve_data: dictionary from compute_lcurve
        corner_lambda: optimal lambda to highlight
        save_path: path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.loglog(lcurve_data['residual_norms'], lcurve_data['solution_norms'], 'b-o', linewidth=2, markersize=4)
    
    if corner_lambda is not None:
        idx = np.argmin(np.abs(lcurve_data['lambda_values'] - corner_lambda))
        plt.loglog(lcurve_data['residual_norms'][idx], lcurve_data['solution_norms'][idx], 
                   'r*', markersize=15, label=f'Corner: λ={corner_lambda:.2e}')
    
    plt.xlabel('Residual Norm ||Ax - b||', fontsize=12)
    plt.ylabel('Solution Norm ||x||', fontsize=12)
    plt.title('L-Curve for Parameter Selection', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if corner_lambda is not None:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_singular_values(s, k_truncate=None, save_path=None):
    """
    Plot singular value decay for TSVD analysis.
    
    Args:
        s: array of singular values
        k_truncate: truncation level to highlight
        save_path: path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(s, 'b-', linewidth=2, label='Singular values')
    
    if k_truncate is not None:
        plt.axvline(x=k_truncate, color='r', linestyle='--', linewidth=2, label=f'Truncation: k={k_truncate}')
    
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Singular Value', fontsize=12)
    plt.title('Singular Value Decay', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot bar chart comparing metrics across methods.
    
    Args:
        metrics_dict: dictionary of {method_name: metrics_dict}
        save_path: path to save figure
    """
    methods = list(metrics_dict.keys())
    metric_names = ['MSE', 'PSNR', 'SSIM']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['mse', 'psnr', 'ssim']):
        values = [metrics_dict[method][metric] for method in methods]
        axes[idx].bar(methods, values, color=['blue', 'green', 'red'][:len(methods)])
        axes[idx].set_title(metric_names[idx], fontsize=14)
        axes[idx].set_ylabel('Value', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_error_vs_parameter(param_values, errors, param_name, optimal_value=None, save_path=None):
    """
    Plot reconstruction error vs parameter value.
    
    Args:
        param_values: array of parameter values
        errors: array of corresponding errors
        param_name: name of parameter (for labels)
        optimal_value: optimal parameter to highlight
        save_path: path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(param_values, errors, 'b-o', linewidth=2, markersize=6)
    
    if optimal_value is not None:
        idx = np.argmin(np.abs(param_values - optimal_value))
        plt.plot(param_values[idx], errors[idx], 'r*', markersize=15, 
                label=f'Optimal: {param_name}={optimal_value}')
    
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.title(f'Error vs {param_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if optimal_value is not None:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_noise_sensitivity(noise_levels, metrics_by_method, save_path=None):
    """
    Plot how methods perform under different noise levels.
    
    Args:
        noise_levels: array of SNR values
        metrics_by_method: dict of {method_name: {metric_name: values}}
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for method_name, metrics in metrics_by_method.items():
        axes[0].plot(noise_levels, metrics['psnr'], '-o', linewidth=2, label=method_name, markersize=6)
        axes[1].plot(noise_levels, metrics['ssim'], '-o', linewidth=2, label=method_name, markersize=6)
    
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR vs Noise Level', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Noise Level', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_runtime_comparison(methods, runtimes, save_path=None):
    """
    Plot runtime comparison between methods.
    
    Args:
        methods: list of method names
        runtimes: list of runtime values (in seconds)
        save_path: path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(methods, runtimes, color=colors[:len(methods)], alpha=0.7, edgecolor='black')
    
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.3f}s',
                ha='center', va='bottom', fontsize=11)
    
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Computational Time Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_blur_strength_analysis(blur_params, metrics_by_method, param_name='Blur Strength', save_path=None):
    """
    Plot how methods perform under different blur strengths.
    
    Args:
        blur_params: array of blur parameter values (e.g., sigma or length)
        metrics_by_method: dict of {method_name: {metric_name: values}}
        param_name: name of blur parameter
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for method_name, metrics in metrics_by_method.items():
        axes[0].plot(blur_params, metrics['psnr'], '-o', linewidth=2, label=method_name, markersize=6)
        axes[1].plot(blur_params, metrics['ssim'], '-o', linewidth=2, label=method_name, markersize=6)
    
    axes[0].set_xlabel(param_name, fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title(f'PSNR vs {param_name}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel(param_name, fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title(f'SSIM vs {param_name}', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_results_summary_table(results_df, save_path=None):
    """
    Create and display a summary table of results.
    
    Args:
        results_df: pandas DataFrame with experimental results
        save_path: path to save table image
    """
    import pandas as pd
    
    summary = results_df.groupby('method').agg({
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Method', 'PSNR (mean±std)', 'SSIM (mean±std)', 'Runtime (mean±std)'])
    
    for method in summary.index:
        psnr_mean = summary.loc[method, ('psnr', 'mean')]
        psnr_std = summary.loc[method, ('psnr', 'std')]
        ssim_mean = summary.loc[method, ('ssim', 'mean')]
        ssim_std = summary.loc[method, ('ssim', 'std')]
        runtime_mean = summary.loc[method, ('runtime', 'mean')]
        runtime_std = summary.loc[method, ('runtime', 'std')]
        
        table_data.append([
            method,
            f'{psnr_mean:.2f} ± {psnr_std:.2f}',
            f'{ssim_mean:.4f} ± {ssim_std:.4f}',
            f'{runtime_mean:.3f} ± {runtime_std:.3f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary of Experimental Results', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return summary