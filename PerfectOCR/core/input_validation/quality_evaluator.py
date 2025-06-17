# PerfectOCR/core/input_validation/quality_evaluator.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ImageQualityEvaluator:
    """
    Analiza la calidad de una imagen y genera planes de corrección personalizados
    para diferentes motores de OCR, utilizando reglas definidas en un archivo de configuración.
    """
    def __init__(self, config: Dict):
        self.rules = config  # Recibe la sección 'quality_assessment_rules'

    def _detect_skew_angle(self, gray_image: np.ndarray) -> float:
        """Detecta el ángulo de inclinación de la imagen."""
        deskew_rules = self.rules.get('deskew', {})
        try:
            # Parámetros desde la configuración con valores por defecto robustos
            canny_params = deskew_rules.get('canny_thresholds', [50, 150])
            hough_threshold = deskew_rules.get('hough_threshold', 150)
            max_line_gap = deskew_rules.get('hough_max_line_gap_px', 20)
            angle_range = deskew_rules.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
            
            # El límite de longitud mínima de la línea se adapta al tamaño de la imagen
            min_line_len_cap = deskew_rules.get('hough_min_line_length_cap_px', 300)
            min_line_len = min(gray_image.shape[1] // 3, min_line_len_cap)

            edges = cv2.Canny(gray_image, canny_params[0], canny_params[1])
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=hough_threshold,
                                    minLineLength=min_line_len,
                                    maxLineGap=max_line_gap)
            if lines is None:
                return 0.0

            angles = [math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
            filtered_angles = [angle for angle in angles if angle_range[0] < angle < angle_range[1]]

            return np.median(filtered_angles) if filtered_angles else 0.0
            
        except Exception as e:
            logger.error(f"Error en la detección de inclinación: {e}", exc_info=True)
            return 0.0

    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estima el nivel de "ruido" o "borrosidad" usando la varianza del Laplaciano."""
        try:
            # La varianza del Laplaciano es un buen indicador de la nitidez de los bordes.
            # Valores bajos sugieren imágenes borrosas.
            return cv2.Laplacian(gray_image, cv2.CV_64F).var()
        except Exception as e:
            logger.error(f"Error en estimación de ruido/nitidez: {e}", exc_info=True)
            return 0.0

    def _estimate_contrast(self, gray_image: np.ndarray) -> Tuple[float, float]:
        """Estima el nivel de contraste usando la desviación estándar de los píxeles."""
        try:
            # La desviación estándar indica cuánto varían los niveles de gris.
            # Un valor bajo significa bajo contraste.
            return np.std(gray_image), np.mean(gray_image)
        except Exception as e:
            logger.error(f"Error en estimación de contraste: {e}", exc_info=True)
            return 255.0, 128.0

    def _get_adaptive_denoise_strength(self, sharpness_level: float, denoise_rules: dict) -> int:
        """Determina la fuerza de eliminación de ruido a aplicar."""
        thresholds = denoise_rules.get('sharpness_thresholds', [60.0, 120.0])
        strengths = denoise_rules.get('strengths_map', [7, 5, 3])
        
        if sharpness_level < thresholds[0]: return strengths[0]  # Borrosa -> Fuerte
        if sharpness_level < thresholds[1]: return strengths[1]  # Normal -> Moderado
        return strengths[2]  # Nítida -> Suave

    def _get_adaptive_contrast_params(self, image_dims: Tuple[int, int], contrast_rules: dict) -> dict:
        """Determina los parámetros de CLAHE a aplicar."""
        dim_thresholds = contrast_rules.get('dimension_thresholds_px', [1000, 2500])
        grid_sizes = contrast_rules.get('grid_sizes_map', [[8, 8], [10, 10], [12, 12]])
        clip_limit = contrast_rules.get('clahe_clip_limit', 1.5)
        
        max_dim = max(image_dims)
        if max_dim < dim_thresholds[0]: grid_size = grid_sizes[0]
        elif max_dim < dim_thresholds[1]: grid_size = grid_sizes[1]
        else: grid_size = grid_sizes[2]
        
        return {'clip_limit': clip_limit, 'grid_size': tuple(grid_size)}

    def _get_adaptive_binarization_params(self, image_height: int, binarization_rules: dict) -> dict:
        """Determina los parámetros de binarización adaptativa."""
        h_thresholds = binarization_rules.get('height_thresholds_px', [800, 1500, 2500])
        block_sizes = binarization_rules.get('block_sizes_map', [31, 41, 51, 61])
        c_value = binarization_rules.get('adaptive_c_value', 7)
        
        if image_height < h_thresholds[0]: block_size = block_sizes[0]
        elif image_height < h_thresholds[1]: block_size = block_sizes[1]
        elif image_height < h_thresholds[2]: block_size = block_sizes[2]
        else: block_size = block_sizes[3]
            
        return {'block_size': block_size, 'c_value': c_value}

    def evaluate_and_create_correction_plan(self, image: np.ndarray) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Función principal que evalúa la imagen y crea planes de corrección para cada motor."""
        observations = []
        correction_plans = {}
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        img_h, img_w = gray_image.shape
        
        # --- Análisis Universal de la Imagen ---
        skew_angle = self._detect_skew_angle(gray_image)
        sharpness = self._estimate_noise_level(gray_image)
        contrast_std_dev, _ = self._estimate_contrast(gray_image)
        
        # --- Generar Plan Específico para cada Motor Y ANÁLISIS ---
        for component in ['tesseract', 'paddleocr', 'spatial_analysis']:
            engine_rules = self.rules.get(component, {})
            plan = {}
            engine_obs = []

            # 1. Plan de inclinación (Deskew)
            min_angle_thresh = self.rules.get('deskew', {}).get('min_angle_for_correction', 0.1)
            if abs(skew_angle) > min_angle_thresh:
                plan['deskew'] = {'angle': skew_angle}
                engine_obs.append(f"Inclinación de {skew_angle:.2f}° detectada (se corregirá).")

            # 2. Plan de ruido/nitidez (Denoise)
            if 'denoise' in engine_rules:
                strength = self._get_adaptive_denoise_strength(sharpness, engine_rules['denoise'])
                plan['denoise'] = {'strength': strength}
                engine_obs.append(f"Nitidez: {sharpness:.1f}. Fuerza de denoise calculada: {strength}.")

            # 3. Plan de contraste (Contrast)
            if 'contrast_enhancement' in engine_rules:
                clahe_params = self._get_adaptive_contrast_params((img_h, img_w), engine_rules['contrast_enhancement'])
                plan['contrast'] = {'clahe_params': clahe_params}
                engine_obs.append(f"Contraste STD: {contrast_std_dev:.1f}. Grid CLAHE: {clahe_params['grid_size']}.")

            # 4. Plan de binarización (solo para Tesseract)
            if component == 'tesseract' and 'binarization' in engine_rules:
                plan['binarization'] = self._get_adaptive_binarization_params(img_h, engine_rules['binarization'])
                engine_obs.append(f"Params de binarización: Bloque={plan['binarization']['block_size']}, C={plan['binarization']['c_value']}.")

            correction_plans[component] = plan
            if engine_obs:
                observations.extend([f"[{component.upper()}] {msg}" for msg in engine_obs])

        return observations, correction_plans