import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import blur_operators as bo

def tsvd_direct(A, b, k_truncate):
    """
    Solve using Truncated SVD with direct SVD computation.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        k_truncate: number of singular values to keep
    
    Returns:
        Reconstructed image vector
    """
    U, s, Vt = svd(A, full_matrices=False)
    
    k = min(k_truncate, len(s))
    
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    coeffs = (U_k.T @ b) / s_k
    x = Vt_k.T @ coeffs
    
    return x

def tsvd_randomized(A, b, k_truncate, n_oversamples=10):
    """
    Solve using Truncated SVD with randomized SVD for large matrices.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        k_truncate: number of singular values to keep
        n_oversamples: oversampling parameter for randomized SVD
    
    Returns:
        Reconstructed image vector
    """
    from sklearn.utils.extmath import randomized_svd
    
    k = min(k_truncate, min(A.shape) - 1)
    U, s, Vt = randomized_svd(A, n_components=k, n_oversamples=n_oversamples, random_state=42)
    
    coeffs = (U.T @ b) / s
    x = Vt.T @ coeffs
    
    return x

def tsvd_sparse(A, b, k_truncate):
    """
    Solve using Truncated SVD with sparse SVD computation.
    
    Args:
        A: blurring matrix (sparse)
        b: blurred image vector
        k_truncate: number of singular values to keep
    
    Returns:
        Reconstructed image vector
    """
    k = min(k_truncate, min(A.shape) - 2)
    
    U, s, Vt = svds(A, k=k, which='LM')
    
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]
    
    coeffs = (U.T @ b) / s
    x = Vt.T @ coeffs
    
    return x

def tsvd_deblur_image(blurred_image, kernel, k_truncate, method='randomized'):
    """
    Deblur image using Truncated SVD.
    
    Args:
        blurred_image: blurred image (2D array)
        kernel: blur kernel
        k_truncate: number of singular values to keep
        method: 'direct', 'randomized', or 'sparse'
    
    Returns:
        Reconstructed image (2D array)
    """
    image_shape = blurred_image.shape
    b = blurred_image.flatten()
    
    A = bo.create_convolution_matrix(kernel, image_shape, boundary='periodic')
    
    if method == 'direct':
        x = tsvd_direct(A, b, k_truncate)
    elif method == 'randomized':
        x = tsvd_randomized(A, b, k_truncate)
    elif method == 'sparse':
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix(A)
        x = tsvd_sparse(A_sparse, b, k_truncate)
    
    return x.reshape(image_shape)

def compute_picard_condition(A, b, plot=False):
    """
    Analyze Picard condition to determine optimal truncation level.
    
    Args:
        A: blurring matrix
        b: blurred image vector
        plot: whether to create plot
    
    Returns:
        Dictionary with singular values and Fourier coefficients
    """
    U, s, Vt = svd(A, full_matrices=False)
    
    utb = np.abs(U.T @ b)
    
    picard_ratio = utb / s
    
    result = {
        'singular_values': s,
        'fourier_coefficients': utb,
        'picard_ratio': picard_ratio
    }
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogy(s, 'b-', label='Singular values')
        plt.semilogy(utb, 'r--', label='|U^T b|')
        plt.xlabel('Index')
        plt.ylabel('Magnitude')
        plt.title('Picard Condition Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return result