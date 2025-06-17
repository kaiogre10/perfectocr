# PerfectOCR/utils/geometry_transformers.py
import logging
from typing import List, Dict, Any, Optional, Union
from utils.geometric import get_shapely_polygon

logger = logging.getLogger(__name__)

def _calculate_centroid_from_polygon(polygon_coords: List[List[Union[float, int]]]) -> Optional[List[float]]:
    """Calcula el centroide de un polígono. Devuelve [cx, cy] o None."""
    if not polygon_coords or not isinstance(polygon_coords, list) or len(polygon_coords) < 3:
        return None
    try:
        # Asegurar que las coordenadas sean floats para Shapely
        float_coords = [[float(p[0]), float(p[1])] for p in polygon_coords]
        shapely_poly = get_shapely_polygon(float_coords) # Usa la utilidad robusta
        if shapely_poly: # get_shapely_polygon ya valida y puede devolver None
            centroid = shapely_poly.centroid
            return [float(centroid.x), float(centroid.y)]
        return None
    except Exception as e:
        logger.error(f"Error calculando centroide para polígono {polygon_coords}: {e}", exc_info=False)
        return None

def _calculate_centroid_from_bbox_list(bbox: List[Union[float, int]]) -> Optional[List[float]]:
    """Calcula el centroide de un bbox [xmin, ymin, xmax, ymax]. Devuelve [cx, cy] o None."""
    if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if xmax > xmin and ymax > ymin:
            return [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0]
        return None
    except (TypeError, ValueError) as e:
        logger.warning(f"Error convirtiendo bbox a float para centroide: {bbox}, error: {e}")
        return None


def convert_to_vector_representation(element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modifica un elemento para usar una representación vectorial (centroide).
    Mueve la geometría detallada original a 'original_geometry_data'.
    Retorna una NUEVA COPIA del elemento modificado.
    """
    if not isinstance(element, dict):
        logger.warning("convert_to_vector_representation recibió una entrada que no es un diccionario.")
        return element 
    
    processed_element = element.copy()
    original_geom_backup: Dict[str, Any] = {}
    centroid_vector: Optional[List[float]] = None
    
    geom_keys_to_backup = ['polygon_coords', 'bbox', 'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'cx', 'cy']
    
    for key in geom_keys_to_backup:
        if key in processed_element:
            original_geom_backup[key] = processed_element.pop(key) # Elimina del nivel superior

    # Calcular el centroide (vector_representation) desde los datos respaldados
    # Prioridad 1: Desde 'cx', 'cy' si estaban explícitamente en el original_geom_backup
    if 'cx' in original_geom_backup and 'cy' in original_geom_backup and \
       isinstance(original_geom_backup['cx'], (int, float)) and \
       isinstance(original_geom_backup['cy'], (int, float)):
        centroid_vector = [float(original_geom_backup['cx']), float(original_geom_backup['cy'])]
    
    if not centroid_vector and 'polygon_coords' in original_geom_backup and original_geom_backup['polygon_coords']:
        centroid_vector = _calculate_centroid_from_polygon(original_geom_backup['polygon_coords'])
    
    if not centroid_vector and 'bbox' in original_geom_backup and original_geom_backup['bbox']:
        centroid_vector = _calculate_centroid_from_bbox_list(original_geom_backup['bbox'])
    
    if centroid_vector:
        processed_element['vector_representation'] = centroid_vector
    else: # Si no se pudo calcular ningún centroide, es una advertencia
        logger.warning(f"No se pudo determinar un vector para el elemento: {element.get('text', 'N/A')[:30]}... "
                       f"Geometría original respaldada: {original_geom_backup}")
            
    processed_element['original_geometry_data'] = original_geom_backup
    return processed_element

def restore_from_vector_representation(element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restaura la geometría completa desde 'original_geometry_data' al nivel superior
    del elemento y elimina 'vector_representation' y 'original_geometry_data'.
    Retorna una NUEVA COPIA del elemento modificado.
    """
    if not isinstance(element, dict):
        logger.warning("restore_from_vector_representation recibió una entrada que no es un diccionario.")
        return element

    processed_element = element.copy()
    if 'original_geometry_data' in processed_element:
        original_geom = processed_element.pop('original_geometry_data')
        if isinstance(original_geom, dict):
            # Restaurar solo las claves que no sobreescriban las existentes,
            # a menos que se desee explícitamente. Por ahora, un simple update.
            processed_element.update(original_geom)
        else:
            logger.error(f"original_geometry_data no era un diccionario para el elemento: {element.get('text', 'N/A')[:30]}...")
            processed_element['original_geometry_data_error'] = original_geom # Guardar dato problemático

    processed_element.pop('vector_representation', None) 
    return processed_element

def vectorize_element_list(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aplica convert_to_vector_representation a una lista de elementos."""
    if not isinstance(elements, list):
        logger.warning("vectorize_element_list recibió una entrada que no es una lista.")
        return elements # O lanzar error
    return [convert_to_vector_representation(el) for el in elements]

def devectorize_element_list(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aplica restore_from_vector_representation a una lista de elementos."""
    if not isinstance(elements, list):
        logger.warning("devectorize_element_list recibió una entrada que no es una lista.")
        return elements # O lanzar error
    return [restore_from_vector_representation(el) for el in elements]
