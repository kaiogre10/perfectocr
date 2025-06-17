"""
Utilidades para manipulación de matrices binarizadas y datos JSON espaciales.
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import copy
from utils.geometric import get_polygon_bounds

logger = logging.getLogger(__name__)

def crop_binary_matrix(binary_matrix: np.ndarray, 
                      y_start: int, 
                      y_end: Optional[int] = None) -> np.ndarray:
    """
    Recorta una matriz binarizada en el eje Y.
    
    Args:
        binary_matrix: Matriz binarizada original (H x W)
        y_start: Coordenada Y de inicio (inclusive)
        y_end: Coordenada Y de fin (exclusive), None = hasta el final
        
    Returns:
        Matriz recortada
    """
    if binary_matrix is None or binary_matrix.size == 0:
        logger.warning("crop_binary_matrix: Matriz vacía o None recibida")
        return np.array([])
    
    h, w = binary_matrix.shape
    
    # Validar y ajustar coordenadas
    y_start = max(0, min(y_start, h - 1))
    if y_end is None:
        y_end = h
    else:
        y_end = max(y_start + 1, min(y_end, h))
    
    # Recortar
    cropped = binary_matrix[y_start:y_end, :]
    
    logger.debug(f"crop_binary_matrix: Original {h}x{w}, recortado desde Y={y_start} hasta Y={y_end}, "
                f"resultado {cropped.shape[0]}x{cropped.shape[1]}")
    
    return cropped

def crop_json_lines(lines: List[Dict[str, Any]], 
                   y_start: float, 
                   y_end: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Recorta las líneas basándose en coordenadas Y, preservando las coordenadas originales.
    """
    filtered_lines = []
    for line in lines:
        # Obtener coordenada Y de la línea
        y_val = get_line_y_coordinate(line)
        if y_val is None:
            continue
            
        # Verificar si la línea está en el rango
        if y_val >= y_start and (y_end is None or y_val <= y_end):
            # Crear una copia de la línea para no modificar la original
            line_copy = line.copy()
            
            # Preservar coordenadas Y originales
            if 'polygon_line_bbox' in line_copy:
                y_coords = [p[1] for p in line_copy['polygon_line_bbox']]
                if y_coords:
                    line_copy['ymin'] = min(y_coords)
                    line_copy['ymax'] = max(y_coords)
                    line_copy['cy'] = (line_copy['ymin'] + line_copy['ymax']) / 2
                    line_copy['y'] = line_copy['cy']
            
            # Preservar coordenadas Y en palabras constituyentes
            if 'constituent_elements_ocr_data' in line_copy:
                for word in line_copy['constituent_elements_ocr_data']:
                    if 'polygon_coords' in word:
                        y_coords = [p[1] for p in word['polygon_coords']]
                        if y_coords:
                            word['ymin'] = min(y_coords)
                            word['ymax'] = max(y_coords)
                            word['cy'] = (word['ymin'] + word['ymax']) / 2
                            word['y'] = word['cy']
            
            filtered_lines.append(line_copy)
    
    return filtered_lines

def get_line_y_coordinate(line: Dict[str, Any]) -> Optional[float]:
    """
    Obtiene la coordenada Y de una línea usando múltiples fuentes.
    """
    # Intentar obtener Y de múltiples fuentes
    y_val = None
    
    # 1. Intentar obtener de geometric_properties_line
    geom_props = line.get('geometric_properties_line', {})
    if geom_props:
        y_val = geom_props.get('cy_avg')
        if y_val is not None:
            return float(y_val)
    
    # 2. Intentar obtener del polígono
    if 'polygon_final' in line:
        polygon = line['polygon_final']
        if polygon and len(polygon) >= 4:
            y_coords = [p[1] for p in polygon]
            y_val = sum(y_coords) / len(y_coords)
            if y_val is not None:
                return float(y_val)
    
    # 3. Intentar obtener de coordenadas individuales
    if 'ymin' in line and line['ymin'] is not None:
        y_val = line['ymin']
    elif 'y' in line and line['y'] is not None:
        y_val = line['y']
    elif 'cy' in line and line['cy'] is not None:
        y_val = line['cy']
    
    return float(y_val) if y_val is not None else None