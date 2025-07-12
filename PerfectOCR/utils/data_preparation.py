# PerfectOCR/utils/data_preparation.py
import logging
from typing import Dict, Any, List, Union, Optional
from utils.geometric import get_polygon_bounds, get_shapely_polygon

logger = logging.getLogger(__name__)

def _calculate_geometry_from_bbox_list(bbox: List[Union[int, float]]) -> Optional[Dict[str, Any]]:
    """
    Calcula la geometría completa desde una bbox con formato [xmin, ymin, xmax, ymax].
    """
    try:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.debug(f"Bbox no es una lista/tupla de 4 elementos: {bbox}")
            return None
        
        xmin, ymin, xmax, ymax = map(float, bbox)
        if xmin >= xmax or ymin >= ymax:
            logger.debug(f"Bbox inválida con coordenadas: {bbox}")
            return None
        
        width = xmax - xmin
        height = ymax - ymin
        cx = xmin + (width / 2)
        cy = ymin + (height / 2)
        
        # Para bboxes, las coordenadas del polígono son simplemente las esquinas del rectángulo
        polygon_coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        
        return {
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "cx": cx, "cy": cy,
            "height": height,
            "width": width,
            "polygon_coords": polygon_coords
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculando geometría desde bbox {bbox}: {e}")
        return None

def _calculate_geometry_from_polygon_coords(polygon_coords: List[List[Union[int,float]]]) -> Optional[Dict[str, Any]]:
    """
    Calcula toda la información geométrica necesaria desde las coordenadas del polígono.
    Utiliza las utilidades espaciales robustas para validar y procesar los polígonos.
    """
    try:
        if not isinstance(polygon_coords, list) or len(polygon_coords) < 3:
            logger.debug(f"Coordenadas de polígono insuficientes: {polygon_coords}")
            return None
        
        # Convertir todos los puntos a float y validar
        float_polygon_coords = []
        for point in polygon_coords:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                logger.debug(f"Punto de polígono inválido: {point}")
                return None
            try:
                float_point = [float(point[0]), float(point[1])]
                float_polygon_coords.append(float_point)
            except (ValueError, TypeError) as e:
                logger.debug(f"Error convirtiendo punto {point} a float: {e}")
                return None
        
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
    Prepara una lista unificada de elementos de texto (líneas)
    a partir de los resultados crudos de PaddleOCR, asegurando
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

def prepare_header_ml_data(
    base_name: str,
    page_dimensions: Dict,
    all_lines: List[Dict],
    header_words: List[Dict]
) -> Optional[Dict]:
    """
    Genera un diccionario con todas las palabras del OCR y una bandera `is_header`.
    """
    logger.info(f"prepare_header_ml_data: Iniciando generación para {base_name}")
    
    if not page_dimensions or 'width' not in page_dimensions or 'height' not in page_dimensions:
        logger.warning("prepare_header_ml_data: Faltan dimensiones de página.")
        return None

    # Crear un conjunto de identificadores únicos para las palabras de encabezado
    header_word_ids = set()
    for hw in header_words:
        # Usar múltiples campos para crear un identificador único
        uid = (
            hw.get('text_raw', hw.get('text', '')),
            hw.get('xmin'),
            hw.get('ymin'),
            hw.get('xmax'),
            hw.get('ymax')
        )
        header_word_ids.add(uid)

    ml_words = []
    total_words_processed = 0
    
    # Procesar todas las líneas y sus palabras
    for line in all_lines:
        words_in_line = line.get('constituent_elements_ocr_data', [])
        for word in words_in_line:
            total_words_processed += 1
            
            # Intentar obtener coordenadas de diferentes campos posibles
            xmin = word.get('xmin') or word.get('left')
            ymin = word.get('ymin') or word.get('top')
            xmax = word.get('xmax') or word.get('right')
            ymax = word.get('ymax') or word.get('bottom')
            
            # Si no tenemos coordenadas, intentar calcularlas desde otros campos
            if xmin is None and 'cx' in word and 'width' in word:
                xmin = word['cx'] - (word['width'] / 2)
                xmax = word['cx'] + (word['width'] / 2)
            if ymin is None and 'cy' in word and 'height' in word:
                ymin = word['cy'] - (word['height'] / 2)
                ymax = word['cy'] + (word['height'] / 2)
            
            if xmin is None or ymin is None or xmax is None or ymax is None:
                logger.debug(f"prepare_header_ml_data: Palabra sin coordenadas válidas, omitiendo: {word.get('text_raw', 'N/A')}")
                continue

            # Crear identificador único para esta palabra
            word_uid = (
                word.get('text_raw', word.get('text', '')),
                xmin,
                ymin,
                xmax,
                ymax
            )
            
            # Determinar si es encabezado
            is_header = word_uid in header_word_ids

            # Construir el diccionario de palabra para ML con formato exacto
            ml_word = {
                "text": word.get('text', word.get('text_raw', '')),
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "is_header": is_header
            }
            ml_words.append(ml_word)
    
    logger.info(f"prepare_header_ml_data: Procesadas {len(ml_words)} palabras válidas de {total_words_processed} totales")
    
    if not ml_words:
        logger.warning("prepare_header_ml_data: No se pudieron procesar palabras válidas.")
        return None

    output = {
        "page_info": {
            "document_name": base_name,
            "page_w": page_dimensions['width'],
            "page_h": page_dimensions['height']
        },
        "words": ml_words
    }
    
    logger.info(f"prepare_header_ml_data: JSON generado exitosamente con {len(ml_words)} palabras")
    return output
