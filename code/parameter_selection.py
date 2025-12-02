import numpy as np
from scipy.optimize import minimize_scalar
import tikhonov_deblur as td
import tsvd_deblur as tsvd

def compute_lcurve(A, b, lambda_range):
    """
    Compute L-curve for Tikhonov regularization parameter selection.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        lambda_range: array of lambda values to test
    
    Returns:
        Dictionary with residual norms, solution norms, and lambda values
    """
    residual_norms = []
    solution_norms = []
    
    for lambda_reg in lambda_range:
        x = td.tikhonov_direct(A, b, lambda_reg)
        
        residual = np.linalg.norm(A @ x - b)
        solution = np.linalg.norm(x)
        
        residual_norms.append(residual)
        solution_norms.append(solution)
    
    return {
        'lambda_values': lambda_range,
        'residual_norms': np.array(residual_norms),
        'solution_norms': np.array(solution_norms)
    }

def find_lcurve_corner(lcurve_data):
    """
    Find corner of L-curve using maximum curvature method.
    
    Args:
        lcurve_data: dictionary from compute_lcurve
    
    Returns:
        Optimal lambda value
    """
    rho = np.log(lcurve_data['residual_norms'])
    eta = np.log(lcurve_data['solution_norms'])
    
    drho = np.gradient(rho)
    deta = np.gradient(eta)
    
    d2rho = np.gradient(drho)
    d2eta = np.gradient(deta)
    
    curvature = (drho * d2eta - deta * d2rho) / (drho**2 + deta**2)**1.5
    
    corner_idx = np.argmax(np.abs(curvature))
    
    return lcurve_data['lambda_values'][corner_idx]

def gcv_function(lambda_reg, A, b):
    """
    Compute Generalized Cross-Validation score.
    
    Args:
        lambda_reg: regularization parameter
        A: blurring matrix
        b: blurred image vector
    
    Returns:
        GCV score
    """
    n = A.shape[1]
    
    x = td.tikhonov_direct(A, b, lambda_reg)
    
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)**2
    
    AtA = A.T @ A
    I = np.eye(n)
    inv_matrix = np.linalg.inv(AtA + lambda_reg * I)
    influence_matrix = A @ inv_matrix @ A.T
    
    trace = np.trace(I[:len(b), :len(b)] - influence_matrix)
    
    gcv_score = residual_norm / (trace**2)
    
    return gcv_score

def select_lambda_gcv(A, b, lambda_range):
    """
    Select regularization parameter using GCV.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        lambda_range: tuple (min, max) for lambda search
    
    Returns:
        Optimal lambda value
    """
    result = minimize_scalar(
        lambda x: gcv_function(x, A, b),
        bounds=lambda_range,
        method='bounded'
    )
    
    return result.x

def discrepancy_principle(A, b, noise_level, lambda_range):
    """
    Select regularization parameter using discrepancy principle.
    
    Args:
        A: blurring matrix
        b: blurred image vector (noisy)
        noise_level: estimate of noise ||eta||
        lambda_range: array of lambda values to test
    
    Returns:
        Optimal lambda value
    """
    target = noise_level
    
    best_lambda = lambda_range[0]
    best_diff = float('inf')
    
    for lambda_reg in lambda_range:
        x = td.tikhonov_direct(A, b, lambda_reg)
        residual_norm = np.linalg.norm(A @ x - b)
        
        diff = abs(residual_norm - target)
        if diff < best_diff:
            best_diff = diff
            best_lambda = lambda_reg
    
    return best_lambda

def select_tsvd_truncation_visual(A, b, max_k):
    """
    Compute reconstruction error for different TSVD truncation levels.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        max_k: maximum truncation level to test
    
    Returns:
        Dictionary with truncation levels and errors
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    k_values = np.arange(10, min(max_k, len(s)), 10)
    errors = []
    
    for k in k_values:
        x = tsvd.tsvd_direct(A, b, k)
        error = np.linalg.norm(A @ x - b)
        errors.append(error)
    
    return {
        'k_values': k_values,
        'errors': np.array(errors)
    }