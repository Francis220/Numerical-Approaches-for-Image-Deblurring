import numpy as np
from scipy.linalg import solve, cholesky
from scipy.sparse.linalg import cg
import blur_operators as bo

def tikhonov_direct(A, b, lambda_reg):
    """
    Solve Tikhonov regularization using direct method (normal equations).
    
    Args:
        A: blurring matrix
        b: blurred image vector
        lambda_reg: regularization parameter
    
    Returns:
        Reconstructed image vector
    """
    n = A.shape[1]
    AtA = A.T @ A
    Atb = A.T @ b
    
    regularized_matrix = AtA + lambda_reg * np.eye(n)
    x = solve(regularized_matrix, Atb, assume_a='pos')
    
    return x

def tikhonov_fft(blurred_image, kernel, lambda_reg):
    """
    Solve Tikhonov regularization using FFT (for shift-invariant blur).
    
    Args:
        blurred_image: blurred image (2D array)
        kernel: blur kernel (2D array)
        lambda_reg: regularization parameter
    
    Returns:
        Reconstructed image (2D array)
    """
    B_freq = np.fft.fft2(blurred_image)
    K_freq = bo.get_fft_blur_operator(kernel, blurred_image.shape)
    
    K_conj = np.conj(K_freq)
    denominator = K_conj * K_freq + lambda_reg
    
    X_freq = (K_conj * B_freq) / denominator
    
    restored = np.real(np.fft.ifft2(X_freq))
    
    return restored

def tikhonov_iterative(A, b, lambda_reg, max_iter=1000, tol=1e-6):
    """
    Solve Tikhonov regularization using conjugate gradient method.
    
    Args:
        A: blurring matrix (can be sparse)
        b: blurred image vector
        lambda_reg: regularization parameter
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        Reconstructed image vector
    """
    n = A.shape[1]
    
    def matvec(x):
        return A.T @ (A @ x) + lambda_reg * x
    
    from scipy.sparse.linalg import LinearOperator
    A_op = LinearOperator((n, n), matvec=matvec)
    
    Atb = A.T @ b
    x, info = cg(A_op, Atb, maxiter=max_iter, tol=tol)
    
    return x

def tikhonov_deblur_image(blurred_image, kernel, lambda_reg, method='fft'):
    """
    Deblur image using Tikhonov regularization.
    
    Args:
        blurred_image: blurred image (2D array)
        kernel: blur kernel
        lambda_reg: regularization parameter
        method: 'fft', 'direct', or 'iterative'
    
    Returns:
        Reconstructed image (2D array)
    """
    if method == 'fft':
        return tikhonov_fft(blurred_image, kernel, lambda_reg)
    
    image_shape = blurred_image.shape
    b = blurred_image.flatten()
    
    if method == 'direct':
        A = bo.create_convolution_matrix(kernel, image_shape, boundary='periodic')
        x = tikhonov_direct(A, b, lambda_reg)
    elif method == 'iterative':
        A = bo.create_convolution_matrix(kernel, image_shape, boundary='periodic')
        x = tikhonov_iterative(A, b, lambda_reg)
    
    return x.reshape(image_shape)