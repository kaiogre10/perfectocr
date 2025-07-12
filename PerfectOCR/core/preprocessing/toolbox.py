# PerfectOCR/core/preprocessing/toolbox.py
import cv2
import numpy as np
import logging
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

def apply_moire_removal_fft(image: np.ndarray, cutoff_freq: float, filter_strength: float) -> np.ndarray:
    """Aplica un filtro FFT anti-moiré con parámetros específicos."""
    
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    
    h, w = image.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_dist = min(h, w) // 2
    if max_dist == 0: return image
    normalized_distance = distance / max_dist
    filter_mask = np.exp(-(normalized_distance / cutoff_freq)**2)
    final_mask = np.clip(1.0 - (filter_strength * (1.0 - filter_mask)), 0, 1)
    f_filtered = f_shifted * final_mask
    f_ishifted = ifftshift(f_filtered)
    filtered_image = np.real(ifft2(f_ishifted))
    
    return np.clip(filtered_image, 0, 255).astype(np.uint8)

def apply_moire_removal_hybrid(image: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Aplica un filtro híbrido anti-moiré con parámetros específicos."""
    smoothed = gaussian_filter(image, sigma=sigma)
    beta = 1.0 - alpha
    result = cv2.addWeighted(image, alpha, smoothed, beta, 0)
    return result.astype(np.uint8)

def apply_deskew(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Aplica una rotación a la imagen para corregir la inclinación."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return result

def apply_bilateral_filter(image: np.ndarray, d: int, sigma_color: int, sigma_space: int) -> np.ndarray:
    """Aplica un filtro bilateral de eliminación de ruido."""
    result = cv2.bilateralFilter(image, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return result

def apply_clahe_contrast(image: np.ndarray, clip_limit: float, grid_size: tuple) -> np.ndarray:
    """Aplica mejora de contraste con CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    result = clahe.apply(image)
    return result

def apply_binarization(image: np.ndarray, block_size: int, c_value: int, invert: bool = False) -> np.ndarray:
    """
    Aplica binarización adaptativa.
    """
    
    if block_size <= 1: 
        block_size = 3
    elif block_size % 2 == 0: 
        block_size += 1
    
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    result = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, c_value)
    return result

def apply_median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aplica un filtro de mediana de OpenCV para eliminar ruido sal y pimienta."""
    # cv2.medianBlur requiere un tamaño de kernel impar.
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)