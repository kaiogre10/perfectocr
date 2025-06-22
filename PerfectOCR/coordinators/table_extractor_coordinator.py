# PerfectOCR/coordinators/table_extractor_coordinator.py
import logging
import os
import yaml
import json
from typing import Dict, Any, Optional, List
from core.geo_matrix.geometric_table_structurer import GeometricTableStructurer
from core.geo_matrix.lineal_reconstructor import LineReconstructor
from core.geo_matrix.header_detector import HeaderDetector
from utils.output_handlers import JsonOutputHandler
from utils.spatial_utils import get_line_y_coordinate
from utils.geometric import get_polygon_bounds
from utils.data_preparation import prepare_header_ml_data
import re
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import process, fuzz


logger = logging.getLogger(__name__)

class TableExtractorCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root
        self.line_reconstructor_params = self.config.get('line_reconstructor_params', {})
        self.header_detector_config = self.config.get('header_detector_config', {})
        self.geometric_structurer_config = self.config.get('geometric_structurer_config', {})

        self.line_reconstructor: Optional[LineReconstructor] = None
        self.header_detector: Optional[HeaderDetector] = None
        self.geometric_structurer: Optional[GeometricTableStructurer] = None
        self.json_output_handler = JsonOutputHandler()
        
    def reconstruct_lines(self, ocr_results: Dict, base_name: str, output_dir: str) -> Dict[str, list]:
        """
        Reconstruye las líneas a partir de los resultados de OCR y las guarda.
        No realiza limpieza de texto.
        """
        metadata = ocr_results.get("metadata", {})
        page_dimensions = metadata.get("dimensions") or metadata.get("page_dimensions") or {}
        width = page_dimensions.get('width')
        height = page_dimensions.get('height')
        if width is None or height is None or width <= 0 or height <= 0:
            logger.error(f"Dimensiones de página inválidas")
            return {}

        line_reconstructor = LineReconstructor(width, height, self.line_reconstructor_params)
        reconstructed_lines_by_engine = line_reconstructor.reconstruct_all_ocr_outputs_parallel(
            ocr_results.get("ocr_raw_results", {}).get("tesseract", {}).get("words", []),
            ocr_results.get("ocr_raw_results", {}).get("paddleocr", {}).get("lines", [])
        )
                
        reconstructed_lines_by_engine['page_dimensions'] = page_dimensions

        self.json_output_handler.save(
            reconstructed_lines_by_engine,
            output_dir,
            f"{base_name}_reconstructed_lines.json",
            output_type="reconstructed_lines"
        )
        return reconstructed_lines_by_engine

    def extract_table_from_cleaned_lines(self, cleaned_lines: Dict[str, list], base_name: str, output_dir: str) -> Dict[str, any]:
        """
        Recibe las líneas ya limpias y realiza la detección de cabecera, delimitación y estructuración de la tabla.
        Permite cambiar entre el detector de cabeceras de ML y el clásico.
        """
        page_dimensions = cleaned_lines.get('page_dimensions', {})
        if not page_dimensions:
            for engine in ['paddle_lines', 'tesseract_lines']:
                if cleaned_lines.get(engine) and cleaned_lines[engine]:
                    page_dimensions = cleaned_lines[engine][0].get('page_dimensions', {})
                    break

        if not page_dimensions or not page_dimensions.get('width') or not page_dimensions.get('height'):
            logger.error("No se encontraron dimensiones de página en las líneas limpias.")
            return self._build_error_response("error_no_page_dims", "Dimensiones de página no disponibles.")

        use_ml_detector = self.header_detector_config.get('use_ml_detector', False)
        header_words, y_min_band, y_max_band = None, None, None
        lines_for_header_detection = cleaned_lines.get('paddle_lines', [])
        
        # Inicializar variables que se usarán después
        semantic_keywords = {}
        all_keywords_flat = []

        # Cargar palabras clave semánticas ANTES de cualquier detección
        semantic_keywords = self._load_semantic_keywords()
        all_keywords_flat = [kw for kws in semantic_keywords.values() for kw in kws] if semantic_keywords else []

        if use_ml_detector:
            logger.info("Usando MLHeaderDetector para la detección de cabecera.")
            model_path = self.header_detector_config.get('ml_model_path')
            if not model_path:
                logger.error("Ruta del modelo de ML para cabecera no especificada en la configuración.")
                return self._build_error_response("error_config", "Ruta del modelo de ML no configurada.")
            
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.project_root, model_path)

            try:
                from core.geo_matrix.ml_header_detector import MLHeaderDetector
                logger.debug("MLHeaderDetector importado correctamente.")
                
                ml_detector = MLHeaderDetector(model_path=model_path)
                logger.debug(f"MLHeaderDetector inicializado con el modelo: {model_path}")
                
                all_words = [word for line in lines_for_header_detection for word in line.get('constituent_elements_ocr_data', [])]
                logger.debug(f"Se pasaron {len(all_words)} palabras al detector de ML.")
                
                header_words = ml_detector.detect(
                    all_words=all_words,
                    page_w=page_dimensions['width'],
                    page_h=page_dimensions['height']
                )
                logger.info(f"MLHeaderDetector predijo {len(header_words)} palabras de cabecera.")

                if header_words:
                    ymins = [w['ymin'] for w in header_words if 'ymin' in w]
                    ymaxs = [w['ymax'] for w in header_words if 'ymax' in w]
                    y_min_band = min(ymins) if ymins else None
                    y_max_band = max(ymaxs) if ymaxs else None
                    logger.info(f"Banda de cabecera de ML detectada: Y=[{y_min_band}, {y_max_band}]")
                else:
                    logger.warning("MLHeaderDetector no detectó ninguna palabra de cabecera.")

            except FileNotFoundError as e:
                logger.error(f"Archivo de modelo de ML no encontrado: {e}. Volviendo al detector clásico.")
                use_ml_detector = False
            except ImportError as e:
                logger.error(f"Fallo de importación para MLHeaderDetector, ¿faltan dependencias como pandas o joblib? Error: {e}. Volviendo al detector clásico.")
                use_ml_detector = False
            except Exception as e:
                logger.error(f"Error inesperado al ejecutar MLHeaderDetector: {e}. Volviendo al detector clásico.", exc_info=True)
                use_ml_detector = False

        # Inicializar el detector clásico SIEMPRE (para fallback o uso directo)
        if not use_ml_detector or not header_words or y_max_band is None:
            logger.info("Usando HeaderDetector clásico para la detección de cabecera.")
            
            # Verificar que tenemos palabras clave
            if not all_keywords_flat:
                logger.warning("No se pudieron cargar palabras clave semánticas. Usando palabras clave por defecto.")
                # Palabras clave por defecto para casos de emergencia
                all_keywords_flat = [
                    'CANTIDAD', 'CANT', 'QTY', 'UNIDADES', 'UNI',
                    'DESCRIPCION', 'DESC', 'PRODUCTO', 'ARTICULO', 'ITEM',
                    'PRECIO', 'P.U.', 'PU', 'PRECIO UNITARIO',
                    'IMPORTE', 'TOTAL', 'SUBTOTAL', 'MONTO'
                ]

        # Crear el detector clásico
        self.header_detector = HeaderDetector(
            config=self.header_detector_config,
            header_keywords_list=all_keywords_flat,
            page_dimensions=page_dimensions
        )
        
        # Usar el detector clásico si no se detectó cabecera con ML
        if not header_words or y_max_band is None:
            logger.info("Intentando detección de cabecera con HeaderDetector clásico...")
            header_words, y_min_band, y_max_band = self.header_detector.identify_header_band_and_words(
                formed_lines=lines_for_header_detection,
                semantic_keywords=semantic_keywords
            )
        
        if not header_words or y_max_band is None:
            logger.error("No se pudo detectar un encabezado de tabla confiable con ningún método")
            return self._build_error_response("error_no_header", "No se pudo detectar un encabezado de tabla confiable con ningún método.")
        
        # --- Búsqueda de límites de tabla ---
        table_end_keywords = self.header_detector_config.get('table_end_keywords', [])
        y_min_table_end = page_dimensions.get('height', 0)
        document_grand_total = None

        lines_after_header = [line for line in lines_for_header_detection if get_line_y_coordinate(line) > y_max_band]
        
        for line in sorted(lines_after_header, key=lambda l: get_line_y_coordinate(l)):
            line_text_raw = line.get("text_raw", "")
            if any(keyword.upper() in line_text_raw.upper() for keyword in table_end_keywords):
                polygon = line.get('polygon_line_bbox')
                if polygon:
                    try:
                        _, ymin_line, _, _ = get_polygon_bounds(polygon)
                        y_min_table_end = ymin_line
                        
                        numbers = re.findall(r'[\d,]+\.?\d{1,2}', line_text_raw)
                        if numbers:
                            try:
                                grand_total_str = numbers[-1].replace(',', '')
                                document_grand_total = float(grand_total_str)
                            except (ValueError, IndexError):
                                pass
                        break
                    except Exception:
                        y_min_table_end = get_line_y_coordinate(line)
                        break
                else:
                    y_min_table_end = get_line_y_coordinate(line)
                    break
        
        # --- Extracción del cuerpo de tabla ---
        table_body_tesseract_lines = [line for line in cleaned_lines.get('tesseract_lines', []) if y_max_band < get_line_y_coordinate(line) < y_min_table_end]
        table_body_paddle_lines = [line for line in cleaned_lines.get('paddle_lines', []) if y_max_band < get_line_y_coordinate(line) < y_min_table_end]
        lines_for_structuring = table_body_paddle_lines or table_body_tesseract_lines
        tesseract_fallback_used = bool(not table_body_paddle_lines and table_body_tesseract_lines)

        # --- Estructuración geométrica ---
        self.geometric_structurer = GeometricTableStructurer(config=self.geometric_structurer_config)
        final_matrix = self.geometric_structurer.structure_table(
            lines_table_only=lines_for_structuring,
            main_header_line_elements=header_words
        )
        
        # --- Búsqueda de totales ---
        document_totals = self._search_document_totals_and_quantities(
            all_lines=cleaned_lines.get('paddle_lines', []),
            used_lines=lines_for_structuring,
            y_max_band=y_max_band,
            y_min_table_end=y_min_table_end,
            page_dimensions=page_dimensions
        )
        
        # --- Guardar datos intermedios ---
        self.json_output_handler.save(
            {
                "header_band_y_coordinates": [y_min_band, y_max_band],
                "table_end_y_coordinate": y_min_table_end,
                "tesseract_table_body_lines": table_body_tesseract_lines,
                "paddle_table_body_lines": table_body_paddle_lines,
            },
            output_dir,
            f"{base_name}_table_body_lines.json",
            output_type="table_body_lines"
        )
        
        final_payload = {
            "document_id": base_name,
            "status": "success_structured_binary_cuts",
            "message": "Tabla estructurada usando binarización directa y GeometricTableStructurer.",
            "outputs": {
                "table_matrix": final_matrix,
                "header_elements": header_words,
                "document_totals": document_totals
            }
        }
        
        if tesseract_fallback_used:
            final_payload.setdefault("summary", {})["warning"] = (
                "PaddleOCR no devolvió líneas; se usó Tesseract como fallback."
            )
        
        if document_grand_total:
            final_payload["outputs"]["document_grand_total"] = document_grand_total
        
        # Generar datos para entrenamiento de ML
        ml_training_data = prepare_header_ml_data(
            base_name=base_name,
            page_dimensions=page_dimensions,
            all_lines=lines_for_header_detection,
            header_words=header_words
        )
        if ml_training_data:
            final_payload.setdefault("outputs", {})["ml_training_data"] = ml_training_data
            # Guardar el JSON de entrenamiento de ML si está habilitado
            self.json_output_handler.save(
                data=ml_training_data,
                output_dir=output_dir,
                file_name_with_extension=f"{base_name}_ml_training_data.json",
                output_type="ml_training_data"
            )

        return final_payload

    def _search_document_totals_and_quantities(
        self,
        all_lines: List[Dict],
        used_lines: List[Dict],
        y_max_band: float,
        y_min_table_end: float,
        page_dimensions: Dict
    ) -> Dict[str, Any]:
        used_line_ids = {line.get('line_id') for line in used_lines}
        remaining_lines = []
        
        for line in all_lines:
            line_y = get_line_y_coordinate(line)
            line_id = line.get('line_id')
            
            if line_id not in used_line_ids:
                remaining_lines.append(line)
            elif line_y >= y_min_table_end:
                remaining_lines.append(line)
        
        total_keywords = self.header_detector_config.get('total_words', [])
        quantity_keywords = self.header_detector_config.get('items_qty', [])
        
        found_totals = []
        found_quantities = []
        
        for line in remaining_lines:
            line_text = line.get("text_raw", "")
            line_text_upper = line_text.upper()
            y_coord = get_line_y_coordinate(line)
            
            # Buscar totales monetarios
            for keyword in total_keywords:
                fuzzy_ratio = fuzz.partial_ratio(keyword.upper(), line_text_upper)
                if fuzzy_ratio >= 70:
                    numbers = re.findall(r'[\d,]+\.?\d{1,2}', line_text)
                    if numbers:
                        try:
                            amount = float(numbers[-1].replace(',', ''))
                            found_totals.append({
                                'type': 'monetary_total',
                                'keyword_found': keyword,
                                'fuzzy_match_score': fuzzy_ratio,
                                'amount': amount,
                                'line_text': line_text,
                                'line_y_coordinate': y_coord
                            })
                            break
                        except (ValueError, IndexError):
                            pass
            
            # Buscar cantidad de artículos
            for keyword in quantity_keywords:
                fuzzy_ratio = fuzz.partial_ratio(keyword.upper(), line_text_upper)
                if fuzzy_ratio >= 75:
                    numbers = re.findall(r'\d+', line_text)
                    if numbers:
                        try:
                            quantity = int(numbers[-1])
                            found_quantities.append({
                                'type': 'item_quantity',
                                'keyword_found': keyword,
                                'fuzzy_match_score': fuzzy_ratio,
                                'quantity': quantity,
                                'line_text': line_text,
                                'line_y_coordinate': y_coord
                            })
                            break
                        except (ValueError, IndexError):
                            pass
        
        return {
            'monetary_totals': found_totals,
            'item_quantities': found_quantities,
            'summary': {
                'total_lines_analyzed': len(remaining_lines),
                'monetary_totals_found': len(found_totals),
                'item_quantities_found': len(found_quantities)
            }
        }

    def _build_error_response(self, code: str, message: str) -> Dict[str, Any]:
        """
        Construye una respuesta de error estandarizada.
        """
        return {"status": code, "message": message, "outputs": {}}

    def _load_semantic_keywords(self) -> Dict[str, List[str]]:
        semantic_keywords_path = self.header_detector_config.get('table_header_keywords_list', {}).get('semantic_keywords_path')
        if not semantic_keywords_path:
            logger.error("No se encontró la ruta al archivo de palabras clave semánticas")
            return {}
        if not os.path.isabs(semantic_keywords_path):
            semantic_keywords_path = os.path.join(self.project_root, semantic_keywords_path)
        try:
            with open(semantic_keywords_path, 'r', encoding='utf-8') as f:
                semantic_keywords = yaml.safe_load(f)
            if not isinstance(semantic_keywords, dict):
                logger.error("El archivo de palabras clave semánticas no tiene el formato esperado")
                return {}
            
            processed_keywords = {}
            for category, keywords in semantic_keywords.items():
                if isinstance(keywords, list):
                    processed_keywords[category.lower()] = [str(kw).upper().strip() for kw in keywords if kw]
            
            return processed_keywords
        except Exception as e:
            logger.error(f"Error cargando palabras clave semánticas: {e}")
            return {}

    def _run_geometric_structuring(self, lines_for_structuring, header_words, page_dimensions):
        """
        Wrapper para estructuración geométrica, útil para pruebas o uso modular.
        """
        self.geometric_structurer = GeometricTableStructurer(config=self.geometric_structurer_config)
        result = self.geometric_structurer.structure_table(
            lines_table_only=lines_for_structuring,
            main_header_line_elements=header_words
        )
        return result


if __name__ == '__main__':
    pass
