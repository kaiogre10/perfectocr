# PerfectOCR/core/preprocessing/toolbox.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_deskew(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Aplica una rotación a la imagen para corregir la inclinación."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def apply_denoise(image: np.ndarray, strength: int) -> np.ndarray:
    """Aplica un filtro de eliminación de ruido."""
    return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)

def apply_clahe_contrast(image: np.ndarray, clip_limit: float, grid_size: tuple) -> np.ndarray:
    """Aplica mejora de contraste con CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def apply_binarization(image: np.ndarray, block_size: int, c_value: int, invert: bool = False) -> np.ndarray:
    """
    Aplica binarización adaptativa.
    
    Args:
        image: Imagen en escala de grises
        block_size: Tamaño del área vecina para calcular el umbral
        c_value: Constante sustraída de la media
        invert: Si True, texto blanco sobre fondo negro. Si False, texto negro sobre fondo blanco.
    """
    if block_size <= 1: 
        block_size = 3
    elif block_size % 2 == 0: 
        block_size += 1
    
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, c_value)