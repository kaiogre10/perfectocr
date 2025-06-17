# PerfectOCR/utils/data_preparation.py
import logging
from typing import List, Dict, Any, Optional, Union
from .geometric import ( # Importar desde el mismo paquete utils
    get_polygon_bounds, get_shapely_polygon,
)

logger = logging.getLogger(__name__)

def _calculate_geometry_from_bbox_list(bbox: List[Union[int, float]]) -> Optional[Dict[str, Any]]:
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        logger.warning(f"Bbox inválido proporcionado para _calculate_geometry_from_bbox_list: {bbox}")
        return None
    try:
        xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if xmax <= xmin or ymax <= ymin:
            logger.debug(f"Bbox con área no positiva: {bbox}")
            return None
        return {
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "cx": (xmin + xmax) / 2.0,
            "cy": (ymin + ymax) / 2.0,
            "height": ymax - ymin,
            "width": xmax - xmin,
            "polygon_coords": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]] # Bbox como polígono
        }
    except (TypeError, ValueError) as e:
        logger.warning(f"Error convirtiendo bbox a float en _calculate_geometry_from_bbox_list: {bbox}, error: {e}")
        return None


def _calculate_geometry_from_polygon_coords(polygon_coords: List[List[Union[int,float]]]) -> Optional[Dict[str, Any]]:
    if not polygon_coords or len(polygon_coords) < 3:
        logger.debug(f"Polígono inválido (menos de 3 puntos) para _calculate_geometry_from_polygon_coords: {polygon_coords}")
        return None
    
    try:
        float_polygon_coords = [[float(p[0]), float(p[1])] for p_idx, p in enumerate(polygon_coords) 
                                if isinstance(p, (list, tuple)) and len(p) == 2]
        if len(float_polygon_coords) < 3: # Re-chequear después de la conversión
             logger.debug(f"Insuficientes puntos válidos después de conversión para polígono: {float_polygon_coords}")
             return None
    except (IndexError, TypeError, ValueError) as e:
        logger.warning(f"Coordenadas de polígono no válidas para conversión a float: {polygon_coords}. Error: {e}")
        return None

    try:
        xmin, ymin, xmax, ymax = get_polygon_bounds(float_polygon_coords) # Usa la utilidad robusta
        height = ymax - ymin 
        width = xmax - xmin   
        
        if height <= 0.01 or width <= 0.01: # Umbral pequeño para evitar problemas de precisión flotante
            logger.debug(f"Polígono con altura o anchura no positiva. Alt: {height}, Anch: {width}, Coords: {float_polygon_coords}")
            return None

        shapely_p = get_shapely_polygon(float_polygon_coords) # Usa la utilidad robusta
        if not shapely_p : # get_shapely_polygon ya devuelve None si es inválido/vacío
            logger.warning(f"Polígono vacío o inválido de Shapely para coords: {float_polygon_coords}")
            return None
        centroid = shapely_p.centroid
        cx, cy = float(centroid.x), float(centroid.y)
        
        return {
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "cx": cx, "cy": cy,
            "height": height,
            "width": width,
            "polygon_coords": float_polygon_coords # Devolver las coordenadas validadas y convertidas a float
        }
    except Exception as e:
        logger.error(f"Error crítico calculando geometría desde polígono {float_polygon_coords}: {e}", exc_info=True)
        return None

def prepare_unified_text_elements(ocr_output_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepara una lista unificada de elementos de texto (palabras o líneas)
    a partir de los resultados crudos de múltiples motores OCR, asegurando
    que cada elemento tenga una estructura geométrica completa y consistente.
    """
    logger.debug("Preparando elementos de texto unificados con geometría completa...")
    unified_elements: List[Dict[str, Any]] = []
    
    if not isinstance(ocr_output_data, dict):
        logger.error("ocr_output_data no es un diccionario válido.")
        return unified_elements
        
    ocr_raw_results = ocr_output_data.get("ocr_raw_results", {})
    if not isinstance(ocr_raw_results, dict):
        logger.error("ocr_raw_results no es un diccionario válido o no existe.")
        return unified_elements


    # Tesseract (palabras)
    tesseract_data = ocr_raw_results.get("tesseract", {})
    tesseract_words_raw = tesseract_data.get("words", []) if isinstance(tesseract_data, dict) else []
    
    for idx, tess_word_data in enumerate(tesseract_words_raw):
        if not isinstance(tess_word_data, dict): continue # Saltar si no es un dict

        text = str(tess_word_data.get('text', '')).strip()
        if not text: continue

        element_dict: Dict[str, Any] = { # Especificar tipo para claridad
            "text": text,
            "confidence": float(tess_word_data.get('confidence', 0.0)),
            "source_ocr": "tesseract",
            "original_id": f"tess_word_{idx}" 
        }
        
        geometry_info = None
        if 'bbox' in tess_word_data and isinstance(tess_word_data['bbox'], list):
            geometry_info = _calculate_geometry_from_bbox_list(tess_word_data['bbox'])
        
        if geometry_info:
            element_dict.update(geometry_info)
            unified_elements.append(element_dict)
        else:
            logger.warning(f"Tesseract: No se pudo calcular geometría para palabra: '{text}' con datos: {tess_word_data}")

    # PaddleOCR (líneas, pero las tratamos como elementos individuales)
    paddle_data = ocr_raw_results.get("paddleocr", {})
    paddle_items_raw = paddle_data.get("lines", []) if isinstance(paddle_data, dict) else []

    for idx, padd_item_data in enumerate(paddle_items_raw):
        if not isinstance(padd_item_data, dict): continue

        text = str(padd_item_data.get('text', '')).strip()
        if not text: continue

        element_dict = {
            "text": text,
            "confidence": float(padd_item_data.get('confidence', 0.0)), 
            "source_ocr": "paddleocr",
            "original_id": f"padd_item_{idx}"
        }
        
        geometry_info = None
        if 'polygon_coords' in padd_item_data and isinstance(padd_item_data['polygon_coords'], list) :
            geometry_info = _calculate_geometry_from_polygon_coords(padd_item_data['polygon_coords'])
        # Podría haber un fallback a 'bbox' de Paddle si 'polygon_coords' no está o es inválido,
        # pero el wrapper actual de Paddle debería priorizar 'polygon_coords'.
        
        if geometry_info:
            element_dict.update(geometry_info)
            unified_elements.append(element_dict)
        else:
            logger.warning(f"PaddleOCR: No se pudo calcular geometría para elemento: '{text}' con datos: {padd_item_data}")
            
    logger.info(f"Preparados {len(unified_elements)} elementos de texto unificados desde OCR.")
    return unified_elements
