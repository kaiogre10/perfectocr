# PerfectOCR/core/spatial_analysis/gradient_calculator.py
import logging
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any, Union
from scipy.ndimage import label, gaussian_filter1d
import os
import json

logger = logging.getLogger(__name__)

def calculate_gradient_maps(binary_image: np.ndarray, 
                           normalize_input: bool = True) -> Dict[str, np.ndarray]:
    """
    Calcula mapas de gradientes horizontal y vertical para detección de líneas artificiales.
    
    Args:
        binary_image: Imagen binaria de entrada (0-255 o 0-1)
        normalize_input: Si True, normaliza la entrada a rango 0-1
        
    Returns:
        Diccionario con gradient_x, gradient_y, gradient_magnitude
    """
    if binary_image is None or binary_image.size == 0:
        logger.warning("calculate_gradient_maps: Imagen de entrada vacía o None.")
        return {}
    
    # Normalizar entrada si es necesario
    if normalize_input:
        if binary_image.dtype == np.uint8:
            # Convertir de 0-255 a 0-1, asumiendo que texto es blanco (255)
            working_image = binary_image.astype(np.float32) / 255.0
        else:
            working_image = binary_image.astype(np.float32)
    else:
        working_image = binary_image.astype(np.float32)
    
    try:
        # Calcular gradientes usando operadores Sobel
        gradient_x = cv2.Sobel(working_image, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(working_image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitud del gradiente
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        logger.debug(f"Gradientes calculados. Shape: {gradient_magnitude.shape}, "
                    f"Max magnitude: {np.max(gradient_magnitude):.3f}")
        
        return {
            'gradient_x': gradient_x,
            'gradient_y': gradient_y,
            'gradient_magnitude': gradient_magnitude
        }
        
    except Exception as e:
        logger.error(f"Error calculando gradientes: {e}", exc_info=True)
        return {}

def detect_horizontal_lines_by_consistency(
    binary_image: np.ndarray,
    min_consistency_ratio: float = 0.50,
    max_vertical_variance: float = 0.6,
    fusion_radius_px: int = 3,
) -> List[List[int]]:
    """
    Detecta líneas horizontales artificiales.
    1. Encuentra filas con alta consistencia horizontal de píxeles.
    2. Valida que la varianza vertical de los píxeles en esa fila sea BAJA
       (para distinguir líneas de `====` de líneas de texto).
    3. Fusiona las filas validadas que estén cerca.

    Args:
        binary_image: Imagen binaria (texto/líneas son oscuras).
        min_consistency_ratio: Proporción de columnas que deben tener un píxel.
        max_vertical_variance: Varianza vertical máxima para ser considerada una línea artificial.
                               Valores bajos (<1.0) indican líneas muy rectas.
        fusion_radius_px: Radio para fusionar líneas detectadas.
    """
    if binary_image is None or binary_image.size == 0:
        return []

    h, w = binary_image.shape
    is_pixel_present = (binary_image < 128).astype(np.uint8)

    # 1. Identificar filas candidatas por consistencia Y artificialidad
    artificial_line_rows_indices = []
    for y in range(h):
        row_pixels_x = np.where(is_pixel_present[y, :])[0]
        
        if not row_pixels_x.any():
            continue

        consistency = len(row_pixels_x) / w
        if consistency < min_consistency_ratio:
            continue

        # 2. Criterio de Artificialidad: Varianza Vertical
        # Para una fila dada 'y', revisamos un pequeño vecindario vertical
        # para calcular la desviación de los píxeles.
        y_min_scan = max(0, y - 1)
        y_max_scan = min(h, y + 2)
        
        pixel_y_coords = np.where(is_pixel_present[y_min_scan:y_max_scan, :])[0]
        
        if not pixel_y_coords.any():
            continue

        # Si la desviación estándar de las posiciones Y es muy baja, es una línea recta.
        vertical_variance = np.std(pixel_y_coords)
        
        if vertical_variance < max_vertical_variance:
            artificial_line_rows_indices.append(y)
    
    if not artificial_line_rows_indices:
        logger.info("No se encontraron líneas artificiales (consistentes y con baja varianza vertical).")
        return []

    logger.debug(f"Se encontraron {len(artificial_line_rows_indices)} filas de líneas artificiales candidatas.")

    # 3. Fusionar las filas validadas que estén cerca
    regions = []
    if not artificial_line_rows_indices:
        return regions

    current_group_start = artificial_line_rows_indices[0]
    current_group_end = artificial_line_rows_indices[0]

    for i in range(1, len(artificial_line_rows_indices)):
        current_y = artificial_line_rows_indices[i]
        
        if current_y - current_group_end <= fusion_radius_px:
            current_group_end = current_y
        else:
            regions.append([0, current_group_start, w - 1, current_group_end])
            current_group_start = current_y
            current_group_end = current_y

    regions.append([0, current_group_start, w - 1, current_group_end])
    return regions


def detect_noise_regions_combined(
    binary_image: np.ndarray,
    **kwargs,
) -> List[List[int]]:
    """
    Wrapper principal que utiliza el detector de consistencia y artificialidad.
    """
    if binary_image is None or binary_image.size == 0:
        return []
        
    config = {
        "min_consistency_ratio": kwargs.get("min_consistency_ratio", 0.50),
        "max_vertical_variance": kwargs.get("max_vertical_variance", 0.6),
        "fusion_radius_px": kwargs.get("fusion_radius_px", 3)
    }
    
    regions = detect_horizontal_lines_by_consistency(binary_image, **config)
    
    logger.info(f"Detector de consistencia y artificialidad encontró y fusionó {len(regions)} regiones de ruido.")
    
    return regions

def _merge_overlapping_regions(regions: List[List[int]], 
                              overlap_threshold: float = 0.5) -> List[List[int]]:
    """
    Combina regiones que se solapan significativamente.
    
    Args:
        regions: Lista de regiones [xmin, ymin, xmax, ymax]
        overlap_threshold: Umbral de solapamiento vertical para combinar
        
    Returns:
        Lista de regiones combinadas
    """
    if not regions:
        return []
    
    # Ordenar por coordenada Y
    sorted_regions = sorted(regions, key=lambda r: r[1])
    merged = []
    
    for current in sorted_regions:
        if not merged:
            merged.append(current)
            continue
        
        last = merged[-1]
        
        # Verificar solapamiento vertical
        y_overlap = max(0, min(current[3], last[3]) - max(current[1], last[1]) + 1)
        min_height = min(current[3] - current[1] + 1, last[3] - last[1] + 1)
        
        if y_overlap / min_height >= overlap_threshold:
            # Combinar regiones
            merged[-1] = [
                min(current[0], last[0]),  # xmin
                min(current[1], last[1]),  # ymin
                max(current[2], last[2]),  # xmax
                max(current[3], last[3])   # ymax
            ]
        else:
            merged.append(current)
    
    return merged

def analyze_gradient_statistics(binary_image: np.ndarray) -> Dict[str, Any]:
    """
    Analiza estadísticas de gradientes para debugging y ajuste de parámetros.
    
    Args:
        binary_image: Imagen binaria de entrada
        
    Returns:
        Diccionario con estadísticas de gradientes
    """
    gradients = calculate_gradient_maps(binary_image, normalize_input=True)
    if not gradients:
        return {}
    
    gradient_magnitude = gradients['gradient_magnitude']
    gradient_y = gradients['gradient_y']
    
    # Proyección por filas del gradiente Y
    row_gradient_profile = np.mean(np.abs(gradient_y), axis=1)
    
    stats = {
        'gradient_magnitude_stats': {
            'mean': float(np.mean(gradient_magnitude)),
            'std': float(np.std(gradient_magnitude)),
            'max': float(np.max(gradient_magnitude)),
            'min': float(np.min(gradient_magnitude))
        },
        'row_gradient_profile_stats': {
            'mean': float(np.mean(row_gradient_profile)),
            'std': float(np.std(row_gradient_profile)),
            'max': float(np.max(row_gradient_profile)),
            'peaks_above_2std': int(np.sum(row_gradient_profile > 
                                        (np.mean(row_gradient_profile) + 2 * np.std(row_gradient_profile))))
        },
        'image_info': {
            'shape': gradient_magnitude.shape,
            'dtype': str(gradient_magnitude.dtype)
        }
    }
    
    logger.debug(f"Estadísticas de gradiente: {stats}")
    return stats

def analyze_line_artificiality(line_region: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """
    Analiza qué tan 'artificial' es una línea horizontal detectada.
    
    Args:
        line_region: Región de la imagen que contiene la línea candidata
        verbose: Si True, incluye métricas detalladas en el resultado
        
    Returns:
        Dict con score de artificialidad (0-1) y métricas componentes
    """
    if line_region is None or line_region.size == 0:
        return {"artificiality_score": 0.0}
    
    try:
        # Normalizar región si es necesario
        if line_region.dtype == np.uint8:
            working_region = line_region.astype(np.float32) / 255.0
        else:
            working_region = line_region.astype(np.float32)
        
        # 1. UNIFORMIDAD HORIZONTAL (líneas ===== son muy uniformes)
        horizontal_profile = np.mean(working_region, axis=0)
        horizontal_variance = np.var(horizontal_profile)
        uniformity_score = 1.0 / (1.0 + horizontal_variance * 10)  # Escalar para sensibilidad
        
        # 2. REPETITIVIDAD DE PATRÓN
        if horizontal_profile.size > 3:
            autocorr = np.correlate(horizontal_profile, horizontal_profile, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            peak_consistency = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
            pattern_repetition = min(1.0, peak_consistency)
        else:
            pattern_repetition = 0.0
        
        # 3. DENSIDAD CONSTANTE VS VARIABLE
        vertical_profile = np.mean(working_region, axis=1)
        if len(vertical_profile) > 1:
            density_std = np.std(vertical_profile)
            density_consistency = 1.0 / (1.0 + density_std * 5)
        else:
            density_consistency = 1.0
        
        # 4. INTENSIDAD PROMEDIO (líneas artificiales tienden a ser muy densas)
        avg_intensity = np.mean(working_region)
        intensity_factor = min(1.0, avg_intensity * 2)  # Asume que líneas densas son más artificiales
        
        # SCORE FINAL COMBINADO
        artificiality_score = (
            uniformity_score * 0.35 + 
            pattern_repetition * 0.25 + 
            density_consistency * 0.25 +
            intensity_factor * 0.15
        )
        
        result = {"artificiality_score": float(artificiality_score)}
        
        if verbose:
            result.update({
                "uniformity_score": float(uniformity_score),
                "pattern_repetition": float(pattern_repetition), 
                "density_consistency": float(density_consistency),
                "intensity_factor": float(intensity_factor),
                "horizontal_variance": float(horizontal_variance),
                "avg_intensity": float(avg_intensity)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error en análisis de artificialidad: {e}")
        return {"artificiality_score": 0.0}

def apply_hessian_validation(line_region: np.ndarray, 
                           basic_score: float,
                           ambiguous_range: Tuple[float, float] = (0.4, 0.8),
                           edge_threshold: float = 0.15) -> Tuple[bool, Dict[str, float]]:
    """
    Aplica validación con Hessiana solo para casos ambiguos.
    
    Args:
        line_region: Región de imagen a analizar
        basic_score: Score de artificialidad básico
        ambiguous_range: Rango (min, max) para considerar ambiguo
        edge_threshold: Umbral para considerar bordes artificiales
        
    Returns:
        Tupla (es_artificial, métricas_hessiana)
    """
    metrics = {"hessian_applied": False, "edge_sharpness": 0.0}
    
    # Solo aplicar Hessiana en zona ambigua
    if not (ambiguous_range[0] < basic_score < ambiguous_range[1]):
        # Decisión directa sin Hessiana
        is_artificial = basic_score > 0.7
        return is_artificial, metrics
    
    try:
        # Normalizar región
        if line_region.dtype == np.uint8:
            working_region = line_region.astype(np.float32) / 255.0
        else:
            working_region = line_region.astype(np.float32)
        
        # Calcular Hessiana (segunda derivada horizontal)
        # Las líneas artificiales tienen transiciones muy bruscas
        first_deriv = cv2.Sobel(working_region, cv2.CV_32F, 1, 0, ksize=3)
        hessian_xx = cv2.Sobel(first_deriv, cv2.CV_32F, 1, 0, ksize=3)
        
        # Medir la "agudeza" de los bordes
        edge_sharpness = np.mean(np.abs(hessian_xx))
        
        # Líneas artificiales (como ====) tienen bordes muy definidos
        is_artificial = edge_sharpness > edge_threshold
        
        metrics.update({
            "hessian_applied": True,
            "edge_sharpness": float(edge_sharpness),
            "threshold_used": edge_threshold
        })
        
        logger.debug(f"Hessiana aplicada: edge_sharpness={edge_sharpness:.4f}, "
                    f"umbral={edge_threshold}, artificial={is_artificial}")
        
        return is_artificial, metrics
        
    except Exception as e:
        logger.error(f"Error en validación Hessiana: {e}")
        # Fallback a decisión básica
        return basic_score > 0.7, metrics

def detect_intelligent_noise_regions(binary_image: np.ndarray,
                                   config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Detecta regiones de ruido usando análisis inteligente de gradientes + Hessiana selectiva.
    
    Args:
        binary_image: Imagen binaria de entrada
        config: Configuración opcional
        
    Returns:
        Lista de regiones con información detallada sobre la detección
    """
    if binary_image is None or binary_image.size == 0:
        logger.warning("detect_intelligent_noise_regions: Imagen vacía.")
        return []
    
    # AÑADIR ESTA LÍNEA DE DEBUG
    debug_gradient_detection(binary_image)
    
    # Configuración por defecto
    default_config = {
        'min_line_length_ratio': 0.2,
        'max_line_height_px': 10,
        'artificiality_threshold': 0.6,
        'hessian_ambiguous_range': [0.4, 0.8],
        'hessian_edge_threshold': 0.15,
        'expansion_factor': 2.0
    }
    
    if config:
        default_config.update(config)
    
    # 1. Detectar líneas candidatas usando método de gradientes existente
    candidate_regions = detect_horizontal_lines_by_consistency(binary_image, **default_config)
    
    if not candidate_regions:
        logger.info("No se encontraron líneas candidatas.")
        return []
    
    # 2. Analizar cada candidata con métricas de artificialidad
    intelligent_regions = []
    h, w = binary_image.shape
    
    for i, (xmin, ymin, xmax, ymax) in enumerate(candidate_regions):
        # Extraer región
        line_region = binary_image[ymin:ymax+1, xmin:xmax+1]
        
        if line_region.size == 0:
            continue
        
        # Análisis de artificialidad
        artificiality_analysis = analyze_line_artificiality(line_region, verbose=True)
        basic_score = artificiality_analysis["artificiality_score"]
        
        # Validación con Hessiana si es necesario
        is_artificial, hessian_metrics = apply_hessian_validation(
            line_region, 
            basic_score,
            tuple(default_config['hessian_ambiguous_range']),
            default_config['hessian_edge_threshold']
        )
        
        # Decidir si es ruido basándose en análisis combinado
        final_decision = is_artificial and basic_score > default_config['artificiality_threshold']
        
        if final_decision:
            # Expandir región para crear máscara efectiva
            expansion = int((ymax - ymin) * default_config['expansion_factor'])
            expanded_ymin = max(0, ymin - expansion)
            expanded_ymax = min(h - 1, ymax + expansion)
            
            region_info = {
                "bbox": [xmin, expanded_ymin, xmax, expanded_ymax],
                "original_bbox": [xmin, ymin, xmax, ymax],
                "artificiality_score": basic_score,
                "is_artificial": final_decision,
                "analysis_method": "hessian" if hessian_metrics["hessian_applied"] else "gradient_only",
                "metrics": {**artificiality_analysis, **hessian_metrics}
            }
            
            intelligent_regions.append(region_info)
            
            logger.debug(f"Región de ruido inteligente detectada #{i+1}: "
                        f"Y:[{ymin}-{ymax}] → expandida a Y:[{expanded_ymin}-{expanded_ymax}], "
                        f"score: {basic_score:.3f}, método: {region_info['analysis_method']}")
    
    logger.info(f"Detección inteligente completada: {len(candidate_regions)} candidatas → "
               f"{len(intelligent_regions)} confirmadas como ruido.")
    
    return intelligent_regions

# Función de conveniencia para mantener compatibilidad
def detect_noise_regions_intelligent(binary_image: np.ndarray,
                                   config: Optional[Dict[str, Any]] = None) -> List[List[int]]:
    """
    Versión simplificada que retorna solo bounding boxes para compatibilidad.
    """
    detailed_regions = detect_intelligent_noise_regions(binary_image, config)
    return [region["bbox"] for region in detailed_regions]

def debug_gradient_detection(binary_image: np.ndarray) -> None:
    """Función de debug para entender por qué no se detectan líneas."""
    if binary_image is None:
        logger.debug("[Spatial Debug] Imagen binaria es None")
        return

    logger.debug(f"[Spatial Debug] Shape: {binary_image.shape}")
    logger.debug(f"[Spatial Debug] Dtype: {binary_image.dtype}")
    logger.debug(f"[Spatial Debug] Valores únicos (trim): {np.unique(binary_image)[:10]}")

    # Guardar imagen para inspección manual (solo la primera vez en sesión)
    try:
        debug_path = os.path.join(os.getcwd(), 'output', 'spatial_debug.png')
        if not os.path.exists(debug_path):
            cv2.imwrite(debug_path, binary_image)
            logger.info(f"[Spatial Debug] Imagen binaria guardada en {debug_path}")
    except Exception as e:
        logger.error(f"[Spatial Debug] No se pudo guardar imagen de debug: {e}")

    vertical_projection = np.mean(binary_image, axis=1)
    logger.debug(
        f"[Spatial Debug] Proyección vertical - min:{np.min(vertical_projection):.3f}, "
        f"max:{np.max(vertical_projection):.3f}")

    threshold = np.mean(vertical_projection) + 2 * np.std(vertical_projection)
    peaks = vertical_projection > threshold
    logger.debug(f"[Spatial Debug] Umbral 2σ: {threshold:.3f}, Filas con picos: {np.sum(peaks)}")

    if np.sum(peaks) > 0:
        peak_rows = np.where(peaks)[0]
        logger.debug(f"[Spatial Debug] Primeras filas con picos: {peak_rows[:10]}")

def detect_consistent_horizontal_lines(binary_image: np.ndarray, 
                                      min_density: float = 0.35, 
                                      max_gap: int = 2) -> List[List[int]]:
    """
    Detecta regiones horizontales consistentes (líneas) fusionando todas las filas
    con alta densidad de píxeles.
    """
    h, w = binary_image.shape
    bin_f = binary_image.astype(np.float32) / 255.0

    # 1. Proyección horizontal: densidad por fila
    row_density = np.mean(bin_f, axis=1)

    # 2. Fila candidata si su densidad supera el umbral
    candidate_rows = row_density > min_density

    # 3. Agrupa filas consecutivas candidatas en regiones
    from scipy.ndimage import label
    labels, num_labels = label(candidate_rows)

    regions = []
    for i in range(1, num_labels + 1):
        rows = np.where(labels == i)[0]
        if rows.size == 0:
            continue
        y0, y1 = int(rows.min()), int(rows.max())
        regions.append([0, y0, w - 1, y1])

    # 4. Fusiona regiones cercanas (separadas por <= max_gap)
    merged = []
    for region in regions:
        if not merged:
            merged.append(region)
        else:
            last = merged[-1]
            if region[1] - last[3] <= max_gap:
                # Fusiona
                merged[-1] = [min(last[0], region[0]), min(last[1], region[1]), 
                              max(last[2], region[2]), max(last[3], region[3])]
            else:
                merged.append(region)
    return merged

def detect_artificial_lines_optimized(binary_image: np.ndarray,
                                     min_consistency_ratio: float = 0.4,
                                     max_vertical_variance: float = 2.0,
                                     fusion_radius_px: int = 3) -> List[List[int]]:
    """
    Versión optimizada usando operaciones vectorizadas de NumPy.
    """
    h, w = binary_image.shape
    is_pixel_present = binary_image > 0
    
    # Vectorizar el cálculo de consistencia
    row_consistencies = np.mean(is_pixel_present, axis=1)
    candidate_rows = np.where(row_consistencies >= min_consistency_ratio)[0]
    
    if len(candidate_rows) == 0:
        return []
    
    # Vectorizar el cálculo de varianza vertical
    artificial_rows = []
    for y in candidate_rows:
        y_min_scan = max(0, y - 1)
        y_max_scan = min(h, y + 2)
        
        # Usar operaciones vectorizadas
        region_pixels = is_pixel_present[y_min_scan:y_max_scan, :]
        if np.any(region_pixels):
            pixel_positions = np.where(region_pixels)
            if len(pixel_positions[0]) > 0:
                vertical_variance = np.std(pixel_positions[0])
                if vertical_variance < max_vertical_variance:
                    artificial_rows.append(y)
    
    # Fusionar regiones usando algoritmo optimizado
    return _merge_regions_optimized(artificial_rows, fusion_radius_px, w)

def _merge_regions_optimized(artificial_rows: List[int], fusion_radius_px: int, w: int) -> List[List[int]]:
    """
    Función auxiliar para fusionar regiones de líneas artificiales.
    """
    if not artificial_rows:
        return []
    
    regions = []
    current_group_start = artificial_rows[0]
    current_group_end = artificial_rows[0]

    for i in range(1, len(artificial_rows)):
        current_y = artificial_rows[i]
        
        if current_y - current_group_end <= fusion_radius_px:
            current_group_end = current_y
        else:
            regions.append([0, current_group_start, w - 1, current_group_end])
            current_group_start = current_y
            current_group_end = current_y

    regions.append([0, current_group_start, w - 1, current_group_end])
    return regions