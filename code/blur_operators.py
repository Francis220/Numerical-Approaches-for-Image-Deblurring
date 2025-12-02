import numpy as np
from scipy.linalg import toeplitz, block_diag
from scipy.sparse import csr_matrix, kron, eye

def create_convolution_matrix(kernel, image_shape, boundary='periodic'):
    """
    Create convolution matrix from blur kernel.
    
    Args:
        kernel: blur kernel (2D array)
        image_shape: tuple (height, width) of image
        boundary: boundary condition ('periodic', 'zero', 'symmetric')
    
    Returns:
        Convolution matrix as 2D array or sparse matrix
    """
    m, n = image_shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    if boundary == 'periodic':
        return create_bttb_matrix(kernel, image_shape)
    else:
        total_size = m * n
        A = np.zeros((total_size, total_size))
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                for ki in range(k_h):
                    for kj in range(k_w):
                        ii = i + ki - pad_h
                        jj = j + kj - pad_w
                        
                        if boundary == 'zero':
                            if 0 <= ii < m and 0 <= jj < n:
                                idx2 = ii * n + jj
                                A[idx, idx2] = kernel[ki, kj]
                        elif boundary == 'symmetric':
                            ii = np.abs(ii) if ii < 0 else (2*m - ii - 2 if ii >= m else ii)
                            jj = np.abs(jj) if jj < 0 else (2*n - jj - 2 if jj >= n else jj)
                            idx2 = ii * n + jj
                            A[idx, idx2] = kernel[ki, kj]
        
        return A

def create_bttb_matrix(kernel, image_shape):
    """
    Create Block Toeplitz with Toeplitz Blocks matrix for periodic boundary.
    
    Args:
        kernel: blur kernel
        image_shape: tuple (height, width)
    
    Returns:
        BTTB matrix
    """
    m, n = image_shape
    k_h, k_w = kernel.shape
    
    padded_kernel = np.zeros((m, n))
    padded_kernel[:k_h, :k_w] = kernel
    padded_kernel = np.roll(padded_kernel, -k_h//2, axis=0)
    padded_kernel = np.roll(padded_kernel, -k_w//2, axis=1)
    
    first_col = padded_kernel[:, 0]
    first_row = padded_kernel[0, :]
    
    T = toeplitz(first_col, first_row)
    
    blocks = []
    for i in range(m):
        row_blocks = []
        for j in range(m):
            shift = (j - i) % m
            row_blocks.append(np.roll(T, shift, axis=1))
        blocks.append(row_blocks)
    
    return np.block(blocks)

def get_fft_blur_operator(kernel, image_shape):
    """
    Get FFT of blur kernel padded to image size for frequency domain operations.
    
    Args:
        kernel: blur kernel
        image_shape: tuple (height, width)
    
    Returns:
        FFT of padded kernel
    """
    m, n = image_shape
    k_h, k_w = kernel.shape
    
    padded_kernel = np.zeros((m, n))
    padded_kernel[:k_h, :k_w] = kernel
    padded_kernel = np.roll(padded_kernel, -k_h//2, axis=0)
    padded_kernel = np.roll(padded_kernel, -k_w//2, axis=1)
    
    return np.fft.fft2(padded_kernel)