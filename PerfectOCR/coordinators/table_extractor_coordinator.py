# PerfectOCR/coordinators/table_extractor_coordinator.py
import logging
import os
import yaml
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.geo_matrix.geometric_table_structurer import GeometricTableStructurer
from core.geo_matrix.lineal_reconstructor import LineReconstructor
from core.geo_matrix.header_detector import HeaderDetector
from utils.output_handlers import JsonOutputHandler
from utils.spatial_utils import get_line_y_coordinate
from utils.geometric import get_polygon_bounds
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class TableExtractorCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root
        self.line_reconstructor_params = self.config.get('line_reconstructor_params', {})
        self.header_detector_config = self.config.get('header_detector_config', {})
        self.geometric_structurer_config = self.config.get('geometric_structurer_config', {})
        self.max_validation_config = self.config.get('max_validation_config', {})
        self.line_reconstructor: Optional[LineReconstructor] = None
        self.header_detector: Optional[HeaderDetector] = None
        self.geometric_structurer: Optional[GeometricTableStructurer] = None
        self.json_output_handler = JsonOutputHandler()
        logger.info("TableExtractorCoordinator inicializado.")

    def _load_semantic_keywords(self) -> Dict[str, List[str]]:
        semantic_keywords_path = self.header_detector_config.get('table_header_keywords_list', {}).get('semantic_keywords_path')
        if not semantic_keywords_path:
            logger.error("No se encontró la ruta al archivo de palabras clave semánticas en la configuración")
            return {}
        if not os.path.isabs(semantic_keywords_path):
            semantic_keywords_path = os.path.join(self.project_root, semantic_keywords_path)
        try:
            with open(semantic_keywords_path, 'r', encoding='utf-8') as f:
                semantic_keywords = yaml.safe_load(f)
            # Validar que semantic_keywords sea un diccionario
            if not isinstance(semantic_keywords, dict):
                logger.error(f"El archivo de palabras clave semánticas no tiene el formato esperado. Se esperaba un diccionario, se obtuvo {type(semantic_keywords)}")
                return {}
            
            # Preservar la estructura de categorías y normalizar las palabras clave
            processed_keywords = {}
            for category, keywords in semantic_keywords.items():
                if isinstance(keywords, list):
                    # Normalizar todas las palabras clave a mayúsculas y limpiar espacios
                    processed_keywords[category.lower()] = [str(kw).upper().strip() for kw in keywords if kw]
            
            logger.debug(f"Palabras clave semánticas cargadas: {processed_keywords}")
            return processed_keywords
        except Exception as e:
            logger.error(f"Error cargando palabras clave semánticas: {e}")
            return {}
            
    def _build_error_response(self, code: str, message: str) -> Dict[str, Any]:
        return {"status": code, "message": message, "outputs": {}}

    def _save_simplified_matrix(self, matrix_data: List[List[Dict]], header_elements: List[Dict], base_name: str, output_dir: str, suffix: str):
        """Guarda una versión simplificada de solo texto de una matriz estructurada."""
        if not matrix_data:
            return

        headers = [h.get("text_raw", "") for h in header_elements]
        semantic_types = [h.get("semantic_type", "descriptivo") for h in header_elements]
        matrix_texts = [[cell.get("cell_text", "") for cell in row] for row in matrix_data]

        simplified_dict = {
            "headers": headers,
            "semantic_types": semantic_types,
            "matrix": matrix_texts
        }

        output_filename = f"{base_name}_{suffix}.json"
        output_path = os.path.join(output_dir, output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(simplified_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Matriz simplificada de depuración guardada en: {output_path}")
        except Exception as e:
            logger.error(f"Error guardando la matriz simplificada de depuración en {output_path}: {e}")

    def _binarize_line(self, line: Dict[str, Any], page_width: int) -> np.ndarray:
        """
        Binariza una línea de texto creando una máscara 2D unificada de todos los polígonos.
        Retorna una matriz 2D donde cada fila representa una fila de píxeles de la línea completa.
        """
        # Obtener las dimensiones verticales de la línea
        segments = line.get('constituent_elements_ocr_data', [])
        if not segments:
            return np.zeros((1, page_width), dtype=np.uint8)
        
        # Calcular el bounding box vertical de toda la línea
        y_coords = []
        for segment in segments:
            y_coords.extend([segment.get('ymin', 0), segment.get('ymax', 0)])
        
        if not y_coords:
            return np.zeros((1, page_width), dtype=np.uint8)
            
        line_ymin = int(min(y_coords))
        line_ymax = int(max(y_coords))
        line_height = max(1, line_ymax - line_ymin)
        
        # Crear máscara 2D
        binary_line_2d = np.zeros((line_height, page_width), dtype=np.uint8)
        
        # Marcar todas las regiones de tinta de los polígonos de PaddleOCR
        for segment in segments:
            xmin = int(segment.get('xmin', 0))
            xmax = int(segment.get('xmax', 0))
            ymin = int(segment.get('ymin', 0)) - line_ymin  # Relativo al inicio de la línea
            ymax = int(segment.get('ymax', 0)) - line_ymin  # Relativo al inicio de la línea
            
            # Validar coordenadas
            if (xmin < xmax and xmin >= 0 and xmax <= page_width and 
                ymin >= 0 and ymax <= line_height and ymin < ymax):
                binary_line_2d[ymin:ymax, xmin:xmax] = 1
        
        return binary_line_2d

    def _analyze_overlapped_rows(self, binary_line_2d: np.ndarray) -> np.ndarray:
        """
        Analiza el solapamiento de filas para encontrar espacios consistentes.
        Para cada columna X, cuenta cuántas filas tienen espacio en blanco (fondo).
        Los espacios más consistentes tendrán valores más altos.
        """
        if binary_line_2d.size == 0:
            return np.array([])
        
        # Para cada columna X, contar cuántas filas tienen fondo (0=fondo/espacio, 1=tinta)
        # Los espacios consistentes aparecerán como columnas con muchos 0s
        space_profile = np.sum(binary_line_2d == 0, axis=0)
        return space_profile

    def _find_column_boundaries_by_spaces(self, binary_line_2d: np.ndarray, num_columns: int) -> List[int]:
        """
        Encuentra los límites de columna usando análisis de solapamiento de filas.
        Busca los H-1 espacios más consistentes y largos en la máscara 2D.
        """
        H = num_columns
        if H <= 1 or binary_line_2d.size == 0:
            return []

        # Obtener el perfil de espacios solapados
        space_profile = self._analyze_overlapped_rows(binary_line_2d)
        if space_profile.size == 0:
            return []
        
        # Encontrar regiones de espacios consistentes
        # Un espacio es consistente si aparece en muchas filas
        line_height = binary_line_2d.shape[0]
        min_consistency_threshold = max(1, int(line_height * 0.3))  # Al menos 30% de las filas
        
        # Identificar columnas que son espacios consistentes
        consistent_spaces = space_profile >= min_consistency_threshold
        
        # Encontrar grupos consecutivos de espacios
        espacios = []
        in_space = False
        space_start = 0
        
        for x, is_space in enumerate(consistent_spaces):
            if is_space and not in_space:
                # Inicio de un espacio
                in_space = True
                space_start = x
            elif not is_space and in_space:
                # Fin de un espacio
                space_width = x - space_start
                space_center = (space_start + x - 1) // 2
                # Usar la consistencia promedio como peso del espacio
                avg_consistency = np.mean(space_profile[space_start:x])
                espacios.append((space_width * avg_consistency, space_center, space_start, x - 1))
                in_space = False
        
        # Manejar el caso donde el espacio llega hasta el final
        if in_space:
            space_width = len(consistent_spaces) - space_start
            space_center = (space_start + len(consistent_spaces) - 1) // 2
            avg_consistency = np.mean(space_profile[space_start:])
            espacios.append((space_width * avg_consistency, space_center, space_start, len(consistent_spaces) - 1))
        
        if len(espacios) < H - 1:
            logger.debug(f"Solo se encontraron {len(espacios)} espacios consistentes, se necesitan {H-1}")
            return []
        
        # Seleccionar los H-1 espacios más importantes (por peso)
        espacios_mayores = sorted(espacios, reverse=True)[:H-1]
        # Ordenar por posición X para obtener los cortes en orden
        espacios_mayores = sorted(espacios_mayores, key=lambda x: x[1])
        
        # Extraer las posiciones de corte (centros de los espacios)
        cortes = [espacio[1] for espacio in espacios_mayores]
        
        #logger.debug(f"Espacios detectados por solapamiento: {len(espacios)}, seleccionados: {len(cortes)}")
        return sorted(cortes)

    def extract_table(
        self,
        ocr_results: Dict,
        base_name: str,
        output_dir: str
    ) -> Dict[str, Any]:
        page_dimensions = ocr_results.get("metadata", {}).get("dimensions", {})
        if not page_dimensions.get('width') or not page_dimensions.get('height'):
            return self._build_error_response("error_no_page_dims", "Dimensiones de página no disponibles.")

        logger.info("Paso 1: Reconstruyendo líneas.")
        line_reconstructor = LineReconstructor(page_dimensions['width'], page_dimensions['height'], self.line_reconstructor_params)
        reconstructed_lines_by_engine = line_reconstructor.reconstruct_all_ocr_outputs_parallel(
            ocr_results.get("ocr_raw_results", {}).get("tesseract", {}).get("words", []),
            ocr_results.get("ocr_raw_results", {}).get("paddleocr", {}).get("lines", [])
        )
        self.json_output_handler.save(
            reconstructed_lines_by_engine,
            output_dir,
            f"{base_name}_reconstructed_lines.json",
            output_type="reconstructed_lines"
        )

        logger.info("Paso 2: Detectando cabecera usando texto de PaddleOCR.")
        semantic_keywords = self._load_semantic_keywords()
        if not semantic_keywords:
            logger.warning("No se pudieron cargar las palabras clave semánticas. Se usarán valores por defecto.")
        
        # Crear lista plana de todas las palabras clave para el HeaderDetector
        all_keywords_flat = []
        if semantic_keywords:
            for keywords_list in semantic_keywords.values():
                all_keywords_flat.extend(keywords_list)

        self.header_detector = HeaderDetector(
            config=self.header_detector_config,
            header_keywords_list=all_keywords_flat,
            page_dimensions=page_dimensions
        )
        header_words, y_min_band, y_max_band = self.header_detector.identify_header_band_and_words(
            formed_lines=reconstructed_lines_by_engine.get('paddle_lines', []),
            semantic_keywords=semantic_keywords  # Pasar el diccionario estructurado
        )
        if not header_words or y_max_band is None:
            return self._build_error_response("error_no_header", "No se pudo detectar un encabezado de tabla confiable.")
        table_end_keywords = self.header_detector_config.get('table_end_keywords',[])
        y_min_table_end = page_dimensions['height']
        lines_after_header = [line for line in reconstructed_lines_by_engine.get('paddle_lines', []) if get_line_y_coordinate(line) > y_max_band]
        for line in sorted(lines_after_header, key=lambda l: get_line_y_coordinate(l)):
            if any(keyword.upper() in line.get("text_raw", "").upper() for keyword in table_end_keywords):
                polygon = line.get('polygon_line_bbox')
                if polygon:
                    try:
                        _, ymin_line, _, _ = get_polygon_bounds(polygon)
                        y_min_table_end = ymin_line
                        logger.info(f"Palabra clave de fin encontrada: '{line.get('text_raw', '')}'. Límite inferior de tabla establecido en Y={y_min_table_end:.2f}")
                        break
                    except Exception as e:
                        logger.warning(f"Error obteniendo límites de línea con palabra clave: {e}")
                        y_min_table_end = get_line_y_coordinate(line)
                        break
                else:
                    y_min_table_end = get_line_y_coordinate(line)
                    break
        table_body_tesseract_lines = [line for line in reconstructed_lines_by_engine.get('tesseract_lines', []) if y_max_band < get_line_y_coordinate(line) < y_min_table_end]
        table_body_paddle_lines = [line for line in reconstructed_lines_by_engine.get('paddle_lines', []) if y_max_band < get_line_y_coordinate(line) < y_min_table_end]
        lines_for_structuring = table_body_paddle_lines
        tesseract_fallback_used = False
        if not lines_for_structuring and table_body_tesseract_lines:
            logger.warning("No se encontraron líneas de tabla de PaddleOCR. Usando Tesseract como fallback.")
            lines_for_structuring = table_body_tesseract_lines
            tesseract_fallback_used = True
        self.json_output_handler.save(
            {
                "header_band_y_coordinates": [y_min_band, y_max_band],
                "table_end_y_coordinate": y_min_table_end,
                "tesseract_table_body_lines": table_body_tesseract_lines,
                "paddle_table_body_lines": table_body_paddle_lines
            },
            output_dir,
            f"{base_name}_table_body_lines.json",
            output_type="table_body_lines"
        )
        
        logger.info("Paso 3: Estructuración geométrica usando GeometricTableStructurer con cortes por valles.")
        self.geometric_structurer = GeometricTableStructurer(config=self.geometric_structurer_config)
        
        # Generar cortes por valles para cada línea
        pixel_gap_cuts_per_line = []
        for line in lines_for_structuring:
            binary_line_2d = self._binarize_line(line, page_dimensions['width'])
            cuts_x = self._find_column_boundaries_by_spaces(binary_line_2d, len(header_words))
            pixel_gap_cuts_per_line.append(cuts_x)
        
        # Usar GeometricTableStructurer con los cortes calculados
        final_matrix = self.geometric_structurer.structure_table_from_pixel_gaps(
            lines_table_only=lines_for_structuring,
            main_header_line_elements=header_words,
            pixel_gap_cuts_per_line=pixel_gap_cuts_per_line
        )
        
        logger.info(f"Matriz final obtenida usando GeometricTableStructurer: {len(final_matrix)} filas.")
        self._save_simplified_matrix(final_matrix, header_words, base_name, output_dir, "final_matrix_only")
        final_payload = {
            "document_id": base_name,
            "status": "success_structured_binary_cuts",
            "message": "Tabla estructurada usando binarización directa y GeometricTableStructurer.",
            "outputs": {
                "table_matrix": final_matrix,
                "header_elements": header_words,
            }
        }
        if tesseract_fallback_used:
            final_payload.setdefault("summary", {})["warning"] = (
                "PaddleOCR no devolvió líneas; se usó Tesseract como fallback."
            )
        return final_payload