# PerfectOCR/core/preprocessing/quality_validator.py
"""Validador de calidad de imagen utilizado dentro del PreprocessingCoordinator.
   Genera dinámicamente un plan de corrección (dict) basado en métricas simples.
"""
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, Tuple, List
from . import toolbox 
from scipy.fft import fft2, fftshift

logger = logging.getLogger(__name__)

class ImageQualityEvaluator:
    """Evalúa nitidez, ruido, contraste e inclinación y aplica las correcciones necesarias."""
    def __init__(self, config: Dict[str, Any]):
        self.rules = config or {}
        logger.debug("ImageQualityEvaluator (worker) inicializado.")

    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        deskew_rules = self.rules.get('deskew', {})
        canny = deskew_rules.get('canny_thresholds', [50, 150])
        hough_thresh = deskew_rules.get('hough_threshold', 150)
        max_gap = deskew_rules.get('hough_max_line_gap_px', 20)
        angle_range = deskew_rules.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        min_len_cap = deskew_rules.get('hough_min_line_length_cap_px', 300)
        min_len = min(gray.shape[1] // 3, min_len_cap)
        edges = cv2.Canny(gray, canny[0], canny[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                                minLineLength=min_len, maxLineGap=max_gap)
        if lines is None:
            return 0.0
        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered = [a for a in angles if angle_range[0] < a < angle_range[1]]
        return np.median(filtered) if filtered else 0.0

    def _estimate_gaussian_noise(self, gray: np.ndarray) -> float:
        """Estima el ruido general o desenfoque usando la varianza del Laplaciano."""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _estimate_salt_pepper_noise(self, gray: np.ndarray) -> float:
        """Estima el ruido 'sal y pimienta' contando la proporción de píxeles en los extremos."""
        total_pixels = gray.size
        if total_pixels == 0: return 0.0
        sp_pixels = np.sum((gray == 0) | (gray == 255))
        return sp_pixels / total_pixels

    def _estimate_contrast(self, gray: np.ndarray) -> Tuple[float, float]:
        return np.std(gray), np.mean(gray)

    def _detect_moire_patterns(self, gray: np.ndarray) -> dict:
        """Analiza el espectro de frecuencia para detectar moiré y su intensidad."""
        f_transform = fft2(gray)
        f_shifted = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shifted)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        magnitude_spectrum[center_y-8:center_y+8, center_x-8:center_x+8] = 0
        
        max_magnitude = np.max(magnitude_spectrum)
        mean_magnitude = np.mean(magnitude_spectrum)
        
        peak_ratio = max_magnitude / mean_magnitude if mean_magnitude > 0 else 0
        
        plan = {}
        if peak_ratio > 4.0:
            logger.info(f"-> Moiré FUERTE detectado (ratio: {peak_ratio:.2f})")
            plan = {'method': 'fft', 'params': {'cutoff_freq': 0.15, 'filter_strength': 0.6}}
        elif peak_ratio > 2.5:
            logger.info(f"-> Moiré MODERADO detectado (ratio: {peak_ratio:.2f})")
            plan = {'method': 'hybrid', 'params': {'sigma': 1.0, 'alpha': 0.6}}

        return plan

    # --- Adaptadores ---
    def _get_adaptive_denoise_strength(self, sharp: float, denoise_rules: dict) -> int:
        thresholds = denoise_rules.get('sharpness_thresholds', [60.0, 120.0])
        strengths = denoise_rules.get('strengths_map', [7, 5, 3])
        if sharp < thresholds[0]:
            return strengths[0]
        if sharp < thresholds[1]:
            return strengths[1]
        return strengths[2]

    def _get_adaptive_contrast_params(self, dims: Tuple[int,int], rules: dict) -> dict:
        dim_thr = rules.get('dimension_thresholds_px', [1000, 2500])
        grid_sizes = rules.get('grid_sizes_map', [[6,6],[8,8],[10,10]])
        clip_limit = rules.get('clahe_clip_limit', 2.0)
        max_dim = max(dims)
        if max_dim < dim_thr[0]:
            grid = grid_sizes[0]
        elif max_dim < dim_thr[1]:
            grid = grid_sizes[1]
        else:
            grid = grid_sizes[2]
        return {'clip_limit': clip_limit, 'grid_size': tuple(grid)}

    # --- NUEVO: Métodos de corrección que usan el toolbox ---
    def _apply_moire_removal(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Aplica la eliminación de moiré llamando a la herramienta adecuada."""
        method = params.get('method')
        method_params = params.get('params')
        if method == 'fft':
            logger.info(f"-> Aplicando filtro Moiré: FFT (cutoff={method_params.get('cutoff_freq')}, strength={method_params.get('filter_strength')}).")
            return toolbox.apply_moire_removal_fft(image, **method_params)
        elif method == 'hybrid':
            logger.info(f"-> Aplicando filtro Moiré: Híbrido (sigma={method_params.get('sigma')}, alpha={method_params.get('alpha')}).")
            return toolbox.apply_moire_removal_hybrid(image, **method_params)
        return image
    
    def _apply_deskew(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Aplica corrección de inclinación usando el toolbox."""
        angle = params.get('angle', 0.0)
        if abs(angle) > 0.1:  # Umbral para evitar micro-correcciones
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            return toolbox.apply_deskew(image, angle)
        return image

    def _apply_denoise(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Aplica filtros de ruido usando el toolbox."""
        if 'median_filter' in params:
            # Asumimos que si median_filter existe, se debe aplicar
            kernel = int(params['median_filter'].get('kernel_size', 3))
            logger.info(f"-> Aplicando filtro de ruido: Mediana (kernel={kernel}).")
            image = toolbox.apply_median_filter(image, kernel)
        if 'bilateral_params' in params:
            # Asumimos que si bilateral_params existe, se debe aplicar
            p = params['bilateral_params']
            logger.info(f"-> Aplicando filtro de ruido: Bilateral (d={p['d']}, sigmaColor={p['sigma_color']}).")
            image = toolbox.apply_bilateral_filter(image, p['d'], p['sigma_color'], p['sigma_space'])
        return image

    def _apply_contrast(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Aplica mejora de contraste usando el toolbox."""
        clip_limit = params.get('clahe_clip_limit', 2.0)
        grid_size = tuple(params.get('tileGridSize', (8, 8)))
        logger.info(f"-> Aplicando mejora de contraste: CLAHE (clip_limit={clip_limit}, grid_size={grid_size}).")
        return toolbox.apply_clahe_contrast(image, clip_limit, grid_size)
    
    def process_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Aplica el pipeline de evaluación y corrección 'ping-pong' a la imagen.
        Devuelve la imagen final corregida.
        """
        logger.info("QualityWorker: Iniciando pipeline de preprocesamiento...")
        corrected_image = gray_image.copy()

        # 1. Eliminar moiré primero, ya que afecta a todas las demás métricas
        moire_rules = self.rules.get('moire_removal', {})
        if moire_rules.get('enabled', False):
            logger.debug("Evaluando paso: 'moire_removal'...")
            moire_plan = self._detect_moire_patterns(corrected_image)
            if moire_plan:
                logger.info("Paso 'moire_removal' necesario. Aplicando corrección.")
                corrected_image = self._apply_moire_removal(corrected_image, moire_plan)
            else:
                logger.debug("Paso 'moire_removal' no fue necesario. Saltando.")
        else:
            logger.info("Paso 'moire_removal' deshabilitado por configuración.")

        # 2. Continuar con el resto del pipeline sobre la imagen (posiblemente) ya corregida de moiré
        #    La secuencia se ajusta al orden definido en `secuencia_lógica.txt`
        pipeline_steps = ['denoise', 'deskew', 'contrast']

        for step in pipeline_steps:
            step_rules = self.rules.get(step, {})
            if not step_rules.get('enabled', False):
                logger.info(f"Paso '{step}' deshabilitado por configuración.")
                continue

            logger.debug(f"Evaluando paso: '{step}'...")
            # 1. Medir y decidir
            observations, plan = self.evaluate_and_create_correction_plan(corrected_image, step_to_evaluate=step)
            
            # 2. Actuar si hay un plan
            if plan:
                if observations:
                    logger.info(f"Paso '{step}' necesario. Motivos: {'; '.join(observations)}. Aplicando corrección.")
                else:
                    logger.info(f"Paso '{step}' necesario. Aplicando corrección.")
                
                correction_type = list(plan.keys())[0]
                params = plan[correction_type]
                
                if correction_type == 'deskew':
                    corrected_image = self._apply_deskew(corrected_image, params)
                elif correction_type == 'denoise':
                    corrected_image = self._apply_denoise(corrected_image, params)
                elif correction_type == 'contrast':
                    corrected_image = self._apply_contrast(corrected_image, params)
            else:
                logger.debug(f"Paso '{step}' no fue necesario. Saltando.")
        
        logger.info("QualityWorker: Pipeline de preprocesamiento finalizado.")
        return corrected_image

    def evaluate_and_create_correction_plan(self, gray: np.ndarray, step_to_evaluate: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Evalúa UN ÚNICO paso y devuelve el plan si es necesario.
        """
        observations: List[str] = []
        plan: Dict[str, Any] = {}

        if step_to_evaluate == 'deskew':
            skew = self._detect_skew_angle(gray)
            logger.debug(f"  Inclinación medida: {skew:.4f}")
            min_angle = self.rules.get('deskew', {}).get('min_angle_for_correction', 0.1)
            if abs(skew) > min_angle:
                plan['deskew'] = {'angle': skew}
                observations.append(f'Inclinación {skew:.2f}°')
        
        elif step_to_evaluate == 'denoise':
            denoise_rules = self.rules.get('denoise', {})
            if not denoise_rules:
                return observations, plan

            # 1. Detección
            sp_noise = self._estimate_salt_pepper_noise(gray)
            gauss_noise = self._estimate_gaussian_noise(gray)
            logger.debug(f"  Ruido S&P medido: {sp_noise:.4%}")
            logger.debug(f"  Ruido Gaussiano/Desenfoque (Laplacian Var): {gauss_noise:.2f}")

            # 2. Planificación
            denoise_plan = {}
            median_rules = denoise_rules.get('median_filter', {})
            bilateral_rules = denoise_rules.get('bilateral_params', {})

            if sp_noise > median_rules.get('salt_pepper_threshold', 0.001):
                denoise_plan['median_filter'] = median_rules
                observations.append(f"Detectado ruido Sal y Pimienta ({sp_noise:.2%})")

            if gauss_noise < bilateral_rules.get('laplacian_variance_threshold', 80.0):
                denoise_plan['bilateral_params'] = bilateral_rules
                observations.append(f"Detectado desenfoque/ruido general ({gauss_noise:.2f})")

            if denoise_plan:
                plan['denoise'] = denoise_plan
        
        elif step_to_evaluate == 'contrast':
            contrast_std, _ = self._estimate_contrast(gray)
            logger.debug(f"  Contraste (Std Dev) medido: {contrast_std:.4f}")
            contrast_rules = self.rules.get('contrast_enhancement', {})
            if contrast_rules and contrast_std < 40.0:
                img_h, img_w = gray.shape
                params = self._get_adaptive_contrast_params((img_h,img_w), contrast_rules)
                plan['contrast'] = {
                    'clahe_clip_limit': params['clip_limit'],
                    'tileGridSize': params['grid_size']
                }
                observations.append('Mejora de contraste')

        return observations, plan