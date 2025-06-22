# PerfectOCR/core/geo_matrix/line_reconstructor.py
import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union
from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils.geometric import get_shapely_polygon, tighten_geometry
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LineReconstructor:
    def __init__(self, page_width: int, page_height: int, config_line_reconstructor_params: Dict):
        self.page_width = page_width
        self.page_height = page_height
        
        self.vertical_overlap_threshold = float(config_line_reconstructor_params.get('vertical_overlap_threshold', 0.5))

        # NUEVO: parámetros para "tighten"
        self.use_tighten_geometry = bool(config_line_reconstructor_params.get('tighten_geometry', False))
        self.tighten_ratio = float(config_line_reconstructor_params.get('tighten_ratio', 0.03))

        logger.info(f"LineReconstructor inicializado. Método Agrupamiento: Solapamiento Vertical (Umbral Global: {self.vertical_overlap_threshold:.2f}). Tighten_geometry: {self.use_tighten_geometry} (ratio={self.tighten_ratio:.3f})")

    def _calculate_vertical_overlap_ratio(self, el1: Dict, el2: Dict) -> bool:
        y1_min = float(el1.get('ymin', 0.0))
        y1_max = float(el1.get('ymax', 0.0))
        y2_min = float(el2.get('ymin', 0.0))
        y2_max = float(el2.get('ymax', 0.0))

        # Usar altura H-1 si existe
        h1 = el1.get('height_h1', el1.get('height', 0.0))
        h2 = el2.get('height_h1', el2.get('height', 0.0))

        if h1 <= 1e-5 or h2 <= 1e-5: return False
        overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if overlap_abs <= 1e-5: return False
            
        min_h = min(h1, h2)
        if min_h <= 1e-5 : return False
        overlap_rel = overlap_abs / min_h
        
        return overlap_rel >= self.vertical_overlap_threshold

    def _group_elements_by_vertical_overlap(self, elements: List[Dict]) -> List[List[Dict]]:
        if not elements: return []
        elements_sorted = sorted(elements, key=lambda el: float(el.get('ymin', float('inf'))))
        
        groups: List[List[Dict]] = []
        if not elements_sorted: return groups

        current_group: List[Dict] = [elements_sorted[0]]
        for i in range(1, len(elements_sorted)):
            word_to_check = elements_sorted[i]
            can_be_grouped = any(self._calculate_vertical_overlap_ratio(word_to_check, member) for member in current_group)
            
            if can_be_grouped:
                current_group.append(word_to_check)
            else:
                groups.append(current_group)
                current_group = [word_to_check]
        if current_group: groups.append(current_group)
        
        return groups

    def _normalize_raw_confidence(self, raw_confidence: Union[float, int, str]) -> float:
        try:
            conf = float(raw_confidence)
            return min(max(conf, 0.0), 100.0)
        except (ValueError, TypeError):
            return 0.0

    def _prepare_ocr_element(self, ocr_item: Dict, element_idx: int, engine_name: str, item_type: str) -> Optional[Dict]:
        try:
            text = str(ocr_item.get("text", "")).strip()
            if not text: return None

            raw_confidence = ocr_item.get("confidence", 0.0)
            polygon_coords_raw: Optional[List[List[Union[int, float]]]] = None

            if engine_name.lower() == "tesseract":
                bbox = ocr_item.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
                    logger.warning(f"Tesseract item {element_idx} ('{text[:20]}...') con bbox inválido: {bbox}")
                    return None
                xmin, ymin, xmax_coord, ymax_coord = bbox[0], bbox[1], bbox[2], bbox[3] # Asumiendo que el JSON de Tesseract tiene [xmin, ymin, xmax, ymax]
                polygon_coords_raw = [[xmin, ymin], [xmax_coord, ymin], [xmax_coord, ymax_coord], [xmin, ymax_coord]]

            elif engine_name.lower() == "paddleocr":
                polygon_coords_raw = ocr_item.get("polygon_coords")
                if not (isinstance(polygon_coords_raw, list) and len(polygon_coords_raw) >= 3):
                    logger.warning(f"PaddleOCR item {element_idx} ('{text[:20]}...') con polygon_coords inválido: {polygon_coords_raw}")
                    return None
            else:
                logger.warning(f"Motor OCR '{engine_name}' no soportado en _prepare_ocr_element.")
                return None

            if not polygon_coords_raw: 
                logger.warning(f"polygon_coords_raw es None o vacío para {engine_name} item {element_idx} ('{text[:20]}...').")
                return None

            float_poly_coords: List[List[float]] = []
            try:
                if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polygon_coords_raw):
                    logger.warning(f"Formato de punto inesperado en polygon_coords_raw para {engine_name} item {element_idx}. Coords: {polygon_coords_raw}")
                    return None
                float_poly_coords = [[float(p[0]), float(p[1])] for p in polygon_coords_raw]
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"No se pudieron convertir polygon_coords a float para {engine_name} item {element_idx} ('{text[:20]}...'). Coords: {polygon_coords_raw}. Error: {e}")
                return None
            
            if len(float_poly_coords) < 3:
                logger.warning(f"Pocos puntos válidos ({len(float_poly_coords)}) después de convertir para {engine_name} item {element_idx} ('{text[:20]}...').")
                return None

            # ---------- NUEVO: contracción opcional ----------
            height_h1: Optional[float] = None
            shapely_poly = None
            if self.use_tighten_geometry:
                tight_res = tighten_geometry(float_poly_coords, shrink_ratio=self.tighten_ratio)
                if tight_res:
                    float_poly_coords = tight_res['polygon_coords']
                    shapely_poly = tight_res['shapely_polygon']
                    height_h1 = tight_res['height_h1']
            # -----------------------------------------------

            if shapely_poly is None:  # sin contracción o falló
                shapely_poly = get_shapely_polygon(float_poly_coords)

            if not shapely_poly or shapely_poly.is_empty or not shapely_poly.is_valid:
                logger.warning(f"Shapely polygon inválido/vacío para {engine_name} item {element_idx} ('{text[:20]}...'). Coords: {float_poly_coords}")
                return None

            min_x, min_y, max_x, max_y = shapely_poly.bounds
            height = max_y - min_y
            width = max_x - min_x

            if height <= 1e-5 or width <= 1e-5:
                logger.debug(f"Elemento {engine_name} ID {element_idx} ('{text[:20]}...') con altura/anchura no positiva: h={height:.2f}, w={width:.2f}")
                return None

            centroid = shapely_poly.centroid
            
            if height_h1 is None:
                height_h1 = max(height - 1.0, 0.0)

            return {
                "internal_id": f"{engine_name}_{item_type}_{element_idx:04d}",
                "text_raw": text,
                "shapely_polygon": shapely_poly,
                "polygon_coords": float_poly_coords,
                "confidence": self._normalize_raw_confidence(raw_confidence),
                "engine_source": engine_name,
                "cx": float(centroid.x), "cy": float(centroid.y),
                "xmin": float(min_x), "ymin": float(min_y), 
                "xmax": float(max_x), "ymax": float(max_y),
                "height": float(height), "width": float(width),
                "height_h1": float(height_h1)
            }
        except Exception as e:
            logger.error(f"Error crítico preparando elemento {engine_name} ID {element_idx} ('{ocr_item.get('text','N/A')}'): {e}", exc_info=True)
            return None

    def _build_line_output(self, group: List[Dict], line_idx: int, engine_name: str) -> Optional[Dict[str, Any]]:
        if not group: return None
        group_sorted = sorted(group, key=lambda el: float(el.get('xmin', self.page_width + 1)))
        text = " ".join([el.get('text_raw', '') for el in group_sorted]).strip()
        if not text: return None

        bbox_coords: List[List[float]] = []
        line_geom_props: Dict[str, Optional[float]] = {
            "cy_avg": None, "ymin_line": None, "ymax_line": None, "xmin_line": None, 
            "xmax_line": None, "height_line": None, "width_line": None
        }

        valid_polys = [el.get('shapely_polygon') for el in group_sorted if el.get('shapely_polygon') and el.get('shapely_polygon').is_valid]
        if valid_polys:
            try:
                merged_geom = unary_union(valid_polys)
                if not merged_geom.is_empty and merged_geom.is_valid:
                    final_poly = merged_geom.convex_hull if hasattr(merged_geom, 'convex_hull') and merged_geom.convex_hull.is_valid and not merged_geom.convex_hull.is_empty else merged_geom
                    if final_poly and not final_poly.is_empty and final_poly.is_valid:
                        bbox_coords = [list(coord) for coord in final_poly.exterior.coords[:-1]]
                        b_min_x, b_min_y, b_max_x, b_max_y = final_poly.bounds
                        centroid = final_poly.centroid # Mantener esta línea
                        line_geom_props.update({
                            "cx_avg": float(centroid.x) if centroid else (float(b_min_x) + float(b_max_x)) / 2.0, # AÑADIDO/MODIFICADO
                            "cy_avg": float(centroid.y) if centroid else (float(b_min_y) + float(b_max_y)) / 2.0,
                            "ymin_line": float(b_min_y), "ymax_line": float(b_max_y),
                            "xmin_line": float(b_min_x), "xmax_line": float(b_max_x),
                            "height_line": float(b_max_y - b_min_y), "width_line": float(b_max_x - b_min_x)
                        })

            except Exception as e_union:
                logger.warning(f"Excepción en unary_union para línea {engine_name} idx {line_idx} ('{text[:30]}...'): {e_union}. Usando fallback.")
                bbox_coords = [] 

        if not bbox_coords: 
            xmins = [float(el['xmin']) for el in group_sorted if el.get('xmin') is not None]
            ymins = [float(el['ymin']) for el in group_sorted if el.get('ymin') is not None]
            xmaxs = [float(el['xmax']) for el in group_sorted if el.get('xmax') is not None]
            ymaxs = [float(el['ymax']) for el in group_sorted if el.get('ymax') is not None]

            if xmins and ymins and xmaxs and ymaxs:
                min_x_val, max_x_val = min(xmins), max(xmaxs)
                min_y_val, max_y_val = min(ymins), max(ymaxs)
                if max_x_val > min_x_val and max_y_val > min_y_val:
                    bbox_coords = [
                        [min_x_val, min_y_val], [max_x_val, min_y_val],
                        [max_x_val, max_y_val], [min_x_val, max_y_val]
                    ]
                    cxs_fb = [float(el.get('cx',0.0)) for el in group_sorted if el.get('cx') is not None] # AÑADIDO
                    cys_fb = [float(el.get('cy',0.0)) for el in group_sorted if el.get('cy') is not None]
                    line_geom_props.update({
                        "cx_avg": np.mean(cxs_fb) if cxs_fb else (min_x_val + max_x_val) / 2.0, # AÑADIDO
                        "cy_avg": np.mean(cys_fb) if cys_fb else (min_y_val + max_y_val) / 2.0,
                        "ymin_line": min_y_val, "ymax_line": max_y_val,
                        "xmin_line": min_x_val, "xmax_line": max_x_val,
                        "height_line": max_y_val - min_y_val, "width_line": max_x_val - min_x_val
                    })
                    
                else: 
                    logger.warning(f"Fallback de Bbox degenerado para línea {engine_name} idx {line_idx} ('{text[:30]}...').")
                    bbox_coords = [] 
            else:
                logger.warning(f"No se pudo determinar geometría (fallback) para línea {engine_name} idx {line_idx} ('{text[:30]}...').")
                bbox_coords = []
        
        if not bbox_coords:
            logger.error(f"No se pudo determinar la geometría para la línea {engine_name} idx {line_idx} con texto: '{text[:50]}...'")
            return None

        confidences = [float(el.get('confidence', 0.0)) for el in group_sorted if el.get('confidence') is not None]
        avg_confidence = round(np.mean(confidences) if confidences else 0.0, 2)
        
        output_elements = []
        for el in group_sorted:
            clean_el = el.copy()
            clean_el.pop('shapely_polygon', None)
            output_elements.append(clean_el)

        return {
            'line_id': f"{engine_name}_line_{line_idx:04d}", 
            'text_raw': text, 
            'polygon_line_bbox': bbox_coords,
            'avg_constituent_confidence': avg_confidence,
            'constituent_elements_ocr_data': output_elements,
            'geometric_properties_line': line_geom_props, 
            'reconstruction_source': f"{engine_name}_overlap_grouping"
        }

    def _reconstruct_for_engine(self, raw_ocr_elements: List[Dict], engine_name: str, item_type: str) -> List[Dict[str, Any]]:
        logger.info(f"Iniciando reconstrucción de líneas para el motor: {engine_name}")
        prepared_elements = [el for el in [self._prepare_ocr_element(item, i, engine_name, item_type) 
                                           for i, item in enumerate(raw_ocr_elements)] if el]
        
        if not prepared_elements:
            logger.warning(f"No se prepararon elementos OCR para {engine_name}.")
            return []

        groups = self._group_elements_by_vertical_overlap(prepared_elements)
        
        reconstructed_lines: List[Dict[str, Any]] = []
        for i, group in enumerate(groups):
            line_obj = self._build_line_output(group, i, engine_name)
            if line_obj:
                reconstructed_lines.append(line_obj)

        return reconstructed_lines

    def reconstruct_all_ocr_outputs_parallel(self, 
                                            tesseract_raw_words: List[Dict], 
                                            paddle_raw_segments: List[Dict]) -> Dict[str, List[Dict]]:
        results = {
            "tesseract_lines": [],
            "paddle_lines": []
        }
        # Usar max_workers=2 para procesar Tesseract y PaddleOCR en paralelo
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='LineRecon') as executor:
            future_tess = None
            if tesseract_raw_words:
                #logger.info("Enviando tarea de reconstrucción de Tesseract al pool de hilos...")
                future_tess = executor.submit(self._reconstruct_for_engine, tesseract_raw_words, "tesseract", "word")
            
            future_padd = None
            if paddle_raw_segments:
                #logger.info("Enviando tarea de reconstrucción de PaddleOCR al pool de hilos...")
                future_padd = executor.submit(self._reconstruct_for_engine, paddle_raw_segments, "paddleocr", "segment")

            if future_tess:
                try:
                    results["tesseract_lines"] = future_tess.result()
                   # logger.info(f"Reconstrucción de Tesseract completada en hilo. {len(results['tesseract_lines'])} líneas.")
                except Exception as e:
                    logger.error(f"Error reconstruyendo líneas de Tesseract en paralelo: {e}", exc_info=True)
            
            if future_padd:
                try:
                    results["paddle_lines"] = future_padd.result()
                   # logger.info(f"Reconstrucción de PaddleOCR completada en hilo. {len(results['paddle_lines'])} líneas.")
                except Exception as e:
                    logger.error(f"Error reconstruyendo líneas de PaddleOCR en paralelo: {e}", exc_info=True)
        
        logger.info(f"Reconstrucción paralela por motor completada. Tesseract: {len(results['tesseract_lines'])} líneas, PaddleOCR: {len(results['paddle_lines'])} líneas.")
        return results