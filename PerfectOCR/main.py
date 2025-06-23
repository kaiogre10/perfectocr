# PerfectOCR/main.py
import json
import yaml
import os
import sys
import argparse
import logging
import cv2
import numpy as np
import time
import shutil
from typing import Dict, Optional, Any, Tuple, List
from coordinators.input_validation_coordinator import InputValidationCoordinator
from coordinators.preprocessing_coordinator import PreprocessingCoordinator
from coordinators.spatial_coordinator import SpatialAnalyzerCoordinator
from coordinators.ocr_coordinator import OCREngineCoordinator
from coordinators.table_extractor_coordinator import TableExtractorCoordinator
from coordinators.postprocessing_coordinator import PostprocessingCoordinator
from coordinators.text_cleaning_coordinator import TextCleaningCoordinator
from utils.output_handlers import JsonOutputHandler, TextOutputHandler, ExcelOutputHandler
from utils.encoders import NumpyEncoder
from core.preprocessing import toolbox

# Configuración conservadora para estabilidad
os.environ.update({
    'OMP_NUM_THREADS': '2',        # Conservador para evitar contención
    'MKL_NUM_THREADS': '2',        # Conservador
    'FLAGS_use_mkldnn': '1',       # Mantener (es estable en main thread)
    'FLAGS_fraction_of_gpu_memory_to_use': '0'
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def setup_logging():
    """Configura el sistema de logging centralizado."""
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.INFO)
    if logger_root.hasHandlers():
        logger_root.handlers.clear()

    formatters = {
        'file': logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ),
        'console': logging.Formatter('%(levelname)s:%(name)s:%(lineno)d - %(message)s')
    }

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatters['file'])
    file_handler.setLevel(logging.DEBUG)
    logger_root.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatters['console'])
    console_handler.setLevel(logging.INFO)
    logger_root.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

class PerfectOCRWorkflow:
    """
    Orquesta el flujo de trabajo completo, con lógica híbrida para usar un 
    servicio potenciador o ejecutar todo localmente.
    """
    def __init__(self, master_config_path: str):
        self.config = self._load_config(master_config_path)
        self.project_root = PROJECT_ROOT
        self._extract_config_sections()
        self._shared_folder_available = None  # None = no verificado, True/False 
        self._input_validation_coordinator: Optional[InputValidationCoordinator] = None
        self._preprocessing_coordinator: Optional[PreprocessingCoordinator] = None
        self._spatial_analyzer_coordinator: Optional[SpatialAnalyzerCoordinator] = None
        self._ocr_coordinator: Optional[OCREngineCoordinator] = None
        self._table_extractor_coordinator: Optional[TableExtractorCoordinator] = None
        self._postprocessing_coordinator: Optional[PostprocessingCoordinator] = None
        self._text_cleaning_coordinator: Optional[TextCleaningCoordinator] = None
        self.json_output_handler = JsonOutputHandler(config=self.output_config)
        self.excel_output_handler = ExcelOutputHandler()
        logger.debug("PerfectOCRWorkflow listo para inicialización bajo demanda de coordinadores.")

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.critical(f"Error crítico cargando config maestra {config_path}: {e}", exc_info=True)
            raise

    def _extract_config_sections(self):
        self.workflow_config = self.config.get('workflow', {})
        self.image_preparation_config = self.config.get('image_preparation', {})
        self.spatial_analyzer_config = self.config.get('spatial_analyzer', {})
        self.ocr_config = self.config.get('ocr', {})
        self.output_config = self.config.get('output_config', {})
        self.table_extractor_config = self.config.get('table_extractor', {})
        self.postprocessing_config = self.config.get('postprocessing', {})
        self.text_cleaning_config = self.config.get('text_cleaning', {})

    @property
    def input_validation_coordinator(self) -> InputValidationCoordinator:
        if self._input_validation_coordinator is None:
            self._input_validation_coordinator = InputValidationCoordinator(config=self.image_preparation_config, project_root=self.project_root)
        return self._input_validation_coordinator

    @property
    def preprocessing_coordinator(self) -> PreprocessingCoordinator:
        if self._preprocessing_coordinator is None:
            self._preprocessing_coordinator = PreprocessingCoordinator(project_root=self.project_root)
        return self._preprocessing_coordinator

    @property
    def spatial_analyzer_coordinator(self) -> SpatialAnalyzerCoordinator:
        if self._spatial_analyzer_coordinator is None:
            self._spatial_analyzer_coordinator = SpatialAnalyzerCoordinator(config=self.spatial_analyzer_config, project_root=self.project_root)
        return self._spatial_analyzer_coordinator

    @property
    def ocr_coordinator(self) -> OCREngineCoordinator:
        if self._ocr_coordinator is None:
            self._ocr_coordinator = OCREngineCoordinator(config=self.ocr_config, project_root=self.project_root)
        return self._ocr_coordinator

    @property
    def table_extractor_coordinator(self) -> TableExtractorCoordinator:
        if self._table_extractor_coordinator is None:
            self._table_extractor_coordinator = TableExtractorCoordinator(
                config=self.table_extractor_config, 
                project_root=self.project_root
            )
            # CAMBIO: Pasar la configuración de output después de crear el coordinador
            self._table_extractor_coordinator.json_output_handler = self.json_output_handler
        return self._table_extractor_coordinator

    @property
    def postprocessing_coordinator(self) -> PostprocessingCoordinator:
        if self._postprocessing_coordinator is None:
            self._postprocessing_coordinator = PostprocessingCoordinator(config=self.postprocessing_config, project_root=self.project_root)
        return self._postprocessing_coordinator

    @property
    def text_cleaning_coordinator(self) -> TextCleaningCoordinator:
        if self._text_cleaning_coordinator is None:
            self._text_cleaning_coordinator = TextCleaningCoordinator(config=self.text_cleaning_config)
        return self._text_cleaning_coordinator

    def _check_shared_folder_availability(self) -> bool:
        """
        Verifica una sola vez si el directorio compartido está disponible.
        Cachea el resultado para evitar verificaciones repetidas.
        """
        if self._shared_folder_available is not None:
            return self._shared_folder_available
            
        config = self.config.get('shared_folder_enhancement', {})
        if not config.get('enabled', False):
            logger.info("El potenciamiento por directorio compartido está deshabilitado en configuración.")
            self._shared_folder_available = False
            return False

        input_dir = config.get('input_dir')
        processed_dir = config.get('processed_dir')

        if not all([input_dir, processed_dir]):
            logger.warning("Rutas de directorio compartido no configuradas completamente.")
            self._shared_folder_available = False
            return False
            
        # Verificar si la unidad base es accesible
        try:
            base_path = os.path.dirname(input_dir)
            if not os.path.exists(base_path):
                logger.warning(f"Directorio compartido no accesible: {base_path}. Modo local será usado para todo el procesamiento.")
                self._shared_folder_available = False
                return False
        except Exception as e:
            logger.warning(f"Error verificando directorio compartido: {e}. Modo local será usado para todo el procesamiento.")
            self._shared_folder_available = False
            return False

        logger.info("Directorio compartido disponible y configurado correctamente.")
        self._shared_folder_available = True
        return True

    def _get_enhanced_data_from_shared_folder(self, input_path: str) -> Optional[Dict[str, Any]]:
        """Orquesta el flujo a través del directorio compartido, esperando resultados del potenciador."""
        # Verificar disponibilidad solo una vez
        if not self._check_shared_folder_availability():
            return None
            
        config = self.config.get('shared_folder_enhancement', {})
        input_dir = config.get('input_dir')
        processed_dir = config.get('processed_dir')
        timeout = config.get('timeout_seconds', 120)

        original_filename = os.path.basename(input_path)
        base_name = os.path.splitext(original_filename)[0]
        
        tess_path = os.path.join(processed_dir, f"{base_name}_tess.png")
        paddle_path = os.path.join(processed_dir, f"{base_name}_paddle.png")
        meta_path = os.path.join(processed_dir, f"{base_name}_meta.json")

        try:
            shutil.copy(input_path, os.path.join(input_dir, original_filename))
            logger.info(f"Imagen '{original_filename}' colocada en '{input_dir}'. Esperando procesamiento por el potenciador...")

            start_wait = time.time()
            while time.time() - start_wait < timeout:
                if os.path.exists(tess_path) and os.path.exists(paddle_path) and os.path.exists(meta_path):
                    logger.info("¡Archivos procesados detectados!")
                    time.sleep(2)  # Pausa para asegurar que la escritura/sincronización haya finalizado
                    
                    tess_img = cv2.imread(tess_path, cv2.IMREAD_UNCHANGED)
                    paddle_img = cv2.imread(paddle_path, cv2.IMREAD_UNCHANGED)
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    if tess_img is None or paddle_img is None:
                        logger.error("Uno de los archivos de imagen procesados no se pudo leer. Volviendo a modo local.")
                        return None

                    os.remove(tess_path); os.remove(paddle_path); os.remove(meta_path)
                    
                    return {
                        "ocr_images": {"tesseract": tess_img, "paddleocr": paddle_img},
                        "noise_regions": metadata.get("noise_regions", []),
                    }
                time.sleep(3)

            logger.warning(f"Timeout de {timeout}s alcanzado. El servicio potenciador podría no estar activo. Volviendo a modo local.")
            return None

        except Exception as e:
            logger.error(f"Error durante el flujo de directorio compartido: {e}", exc_info=True)
            return None

    def process_document(self, input_path: str, output_dir_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        processing_times_summary: Dict[str, float] = {}
        original_file_name = os.path.basename(input_path)
        base_name = os.path.splitext(original_file_name)[0]
        logger.info(f"Iniciando procesamiento de documento: {original_file_name}")

        enhanced_results = self._get_enhanced_data_from_shared_folder(input_path)

        if enhanced_results:
            logger.info("Continuando flujo en modo Híbrido desde la Fase 4: OCR.")
            ocr_images_dict = enhanced_results.get("ocr_images", {})
            noise_regions = enhanced_results.get("noise_regions", [])
        else:
            if self._shared_folder_available is None:
                logger.info("Potenciador no utilizado. Procediendo en modo Standalone desde la Fase 1.")
            
            # FASE 1: VALIDACIÓN (Local)
            quality_observations, correction_plans, image_loaded, time_val = self.input_validation_coordinator.validate_and_assess_image(input_path)
            processing_times_summary["1_input_validation"] = round(time_val, 4)
            if image_loaded is None or not correction_plans:
                return self._build_error_response("error_input_validation", original_file_name, "Fallo en carga o planificación.", "input_validation")

            # FASE 2: PREPROCESAMIENTO (Local)
            preproc_results, time_prep = self.preprocessing_coordinator.apply_preprocessing_pipelines(image_loaded, correction_plans, input_path)
            processing_times_summary["2_preprocessing"] = round(time_prep, 4)
            if not preproc_results or "ocr_images" not in preproc_results:
                return self._build_error_response("error_preprocessing", original_file_name, "Fallo en la generación de imágenes para OCR.", "preprocessing")
            
            ocr_images_dict = preproc_results["ocr_images"]
            spatial_image = preproc_results.get("spatial_image")

            # FASE 3: ANÁLISIS ESPACIAL (Local)
            image_for_spatial_analysis = spatial_image
            if image_for_spatial_analysis is None:
                # Fallback por si la imagen espacial no se generó
                logger.warning("Imagen espacial dedicada no encontrada. Usando la de Tesseract como fallback.")
                tesseract_image = ocr_images_dict.get('tesseract')
                if tesseract_image is not None:
                    # La imagen de tesseract tiene texto negro (0), la invertimos para el análisis
                    image_for_spatial_analysis = cv2.bitwise_not(tesseract_image)

            if image_for_spatial_analysis is not None:
                self.spatial_analyzer_coordinator.analyze_image(image_for_spatial_analysis)
                noise_regions = self.spatial_analyzer_coordinator.detect_noise_regions()
                # NO limpiar datos aquí - se necesitan para la extracción de tabla
            else:
                noise_regions = []
                logger.error("No se pudo obtener ninguna imagen para el análisis espacial.")

        # --- EL FLUJO SE UNIFICA AQUÍ ---
        current_output_dir = output_dir_override if output_dir_override else self.workflow_config.get('output_folder')
        os.makedirs(current_output_dir, exist_ok=True)
        
        # FASE 4: OCR
        ocr_results_payload, time_ocr = self.ocr_coordinator.run_ocr_parallel(ocr_images_dict, noise_regions, original_file_name)
        processing_times_summary["4_ocr"] = round(time_ocr, 4)
        if not self._validate_ocr_results(ocr_results_payload, original_file_name):
            return self._build_error_response("error_ocr", original_file_name, "OCR no produjo texto utilizable.", "ocr_validation")
        ocr_results_json_path = self._save_ocr_results(ocr_results_payload, base_name, current_output_dir)

        # FASE 5: Reconstrucción de líneas (nuevo método en TableExtractorCoordinator)
        reconstructed_lines = self.table_extractor_coordinator.reconstruct_lines(
            ocr_results=ocr_results_payload,
            base_name=base_name,
            output_dir=current_output_dir
        )

        # FASE 5.5: Limpieza de texto
        output_dir = output_dir_override if output_dir_override else os.path.join(self.project_root, "output")
        cleaned_lines, corrections_count = self.text_cleaning_coordinator.clean_reconstructed_lines(
            reconstructed_lines_by_engine=reconstructed_lines,
            output_dir=output_dir,
            doc_id=base_name
        )
        if corrections_count > 0:
            logger.info(f"Limpieza de texto completada: {corrections_count} líneas contenían correcciones.")

        # FASE 6: Extracción de tabla usando líneas limpias
        table_extraction_payload = self.table_extractor_coordinator.extract_table_from_cleaned_lines(
            cleaned_lines=cleaned_lines,
            base_name=base_name,
            output_dir=current_output_dir
        )
        
        # Ahora sí, pásalo al postprocesamiento
        semantically_corrected_payload = self.postprocessing_coordinator.correct_table_structure(
            extraction_payload=table_extraction_payload
        )
        
        # Guardar la matriz corregida semánticamente para depuración
        if self.output_config.get('enabled_outputs', {}).get('debug_semantic_matrix', False):
            self._save_simplified_matrix(
                matrix_data=semantically_corrected_payload.get('outputs', {}).get('table_matrix', []),
                header_elements=semantically_corrected_payload.get('outputs', {}).get('header_elements', []),
                base_name=base_name,
                output_dir=current_output_dir,
                suffix="semantically_corrected_matrix",
                payload=semantically_corrected_payload
            )

        # Guardar la matriz después de la corrección de consistencia semántica
        if self.output_config.get('enabled_outputs', {}).get('semantic_consistency_matrix', False):
            self._save_simplified_matrix(
                matrix_data=semantically_corrected_payload.get('outputs', {}).get('semantic_consistency_matrix', []),
                header_elements=semantically_corrected_payload.get('outputs', {}).get('header_elements', []),
                base_name=base_name,
                output_dir=current_output_dir,
                suffix="semantic_consistency_matrix",
                payload=semantically_corrected_payload
            )

        # Guardar la matriz final después de math_max
        if self.output_config.get('enabled_outputs', {}).get('math_max_matrix', False):
            self._save_simplified_matrix(
                matrix_data=semantically_corrected_payload.get('outputs', {}).get('math_max_matrix', []),
                header_elements=semantically_corrected_payload.get('outputs', {}).get('header_elements', []),
                base_name=base_name,
                output_dir=current_output_dir,
                suffix="math_max_matrix",
                payload=semantically_corrected_payload
            )

        # Usar el payload corregido para la respuesta final
        final_response = self._build_final_response(
            original_file_name,
            ocr_results_json_path,
            semantically_corrected_payload
        )
        
        # Añadir tiempos de procesamiento
        if 'metadata' not in final_response: final_response['metadata'] = {}
        final_response['metadata']['processing_times_seconds'] = processing_times_summary

        return final_response
    
    # --- Métodos de ayuda (sin cambios) ---
    def _validate_ocr_results(self, ocr_results: Optional[dict], filename: str) -> bool:
        """
        Valida que los resultados de OCR contengan texto utilizable.
        Ahora adaptado para manejar motores habilitados/deshabilitados.
        """
        if not isinstance(ocr_results, dict): 
            return False
        
        ocr_raw_results = ocr_results.get("ocr_raw_results", {})
        enabled_engines = ocr_results.get("metadata", {}).get("enabled_engines", {})
        
        # Verificar si hay texto utilizable en al menos uno de los motores habilitados
        has_valid_text = False
        
        # Verificar Tesseract si está habilitado
        if enabled_engines.get("tesseract", False):
            tesseract_data = ocr_raw_results.get("tesseract", {})
            if "error" not in tesseract_data:
                has_tesseract_words = bool(tesseract_data.get("words", []))
                if has_tesseract_words:
                    has_valid_text = True
                    logger.debug(f"Tesseract produjo {len(tesseract_data.get('words', []))} palabras para {filename}")
            else:
                logger.warning(f"Tesseract tuvo error para {filename}: {tesseract_data.get('error')}")
        
        # Verificar PaddleOCR si está habilitado
        if enabled_engines.get("paddleocr", False):
            paddle_data = ocr_raw_results.get("paddleocr", {})
            if "error" not in paddle_data:
                has_paddle_lines = bool(paddle_data.get("lines", []))
                if has_paddle_lines:
                    has_valid_text = True
                    logger.debug(f"PaddleOCR produjo {len(paddle_data.get('lines', []))} líneas para {filename}")
            else:
                logger.warning(f"PaddleOCR tuvo error para {filename}: {paddle_data.get('error')}")
        
        if not has_valid_text:
            enabled_list = [engine for engine, enabled in enabled_engines.items() if enabled]
            logger.error(f"OCR no produjo texto utilizable para {filename}. "
                        f"Motores habilitados: {enabled_list}")
            return False
        
        return True

    def _save_ocr_results(self, ocr_results: dict, base_name: str, output_dir: str) -> Optional[str]:
        ocr_json_filename = f"{base_name}_ocr_raw_results.json"
        return self.json_output_handler.save(
            data=ocr_results, 
            output_dir=output_dir, 
            file_name_with_extension=ocr_json_filename,
            output_type="ocr_raw"
        )

    def _build_error_response(self, status: str, filename: str, message: str, stage: Optional[str] = None) -> dict:
        error_details = {"message": message}
        if stage: error_details["stage"] = stage
        return {"document_id": filename, "status_overall_workflow": status, "error_details": error_details }

    def _build_final_response(self, filename: str, ocr_path: Optional[str], table_payload: dict) -> dict:
        status = table_payload.get("status", "error_unknown")
        final_status = "success" if status.startswith("success") else status
        outputs = {"ocr_raw_json": ocr_path}
        
        # Solo intentar guardar la tabla estructurada si tenemos una ruta OCR válida
        if table_payload.get("outputs", {}).get("table_matrix") and ocr_path:
            # Usar el mismo directorio que el archivo OCR
            output_dir = os.path.dirname(ocr_path)
            if output_dir:  # Verificar que el directorio no esté vacío
                matrix_path = self.json_output_handler.save(
                    table_payload["outputs"], 
                    output_dir, 
                    f"{os.path.splitext(filename)[0]}_structured_table.json",
                    output_type="structured_table"
                )
                outputs["structured_table_json"] = matrix_path
            else:
                logger.warning(f"No se pudo determinar el directorio de salida para {filename}")
        elif table_payload.get("outputs", {}).get("table_matrix"):
            logger.warning(f"No se pudo guardar la tabla estructurada para {filename}: ocr_path es None")
            
        summary = {"table_extraction_status": status, "message": table_payload.get("message")}
        return {"document_id": filename, "status_overall_workflow": final_status, "outputs": outputs, "summary": summary}

    def _save_simplified_matrix(self, matrix_data: List[List[Dict]], header_elements: List[Dict], base_name: str, output_dir: str, suffix: str, payload: Optional[Dict] = None):
        """Guarda una versión simplificada de solo texto de una matriz estructurada."""
        if not matrix_data:
            return

        headers = [h.get("text_raw", "") for h in header_elements]
        
        # LÓGICA FUSIONADA: Usar semantic_math_types si está disponible, sino semantic_types básicos
        if suffix == "math_max_matrix" and payload:
            semantic_math_types = payload.get('outputs', {}).get('semantic_math_types')
            if semantic_math_types:
                semantic_types = semantic_math_types  # ← REEMPLAZAR directamente
            else:
                semantic_types = [h.get("semantic_type", "descriptivo") for h in header_elements]
        else:
            semantic_types = [h.get("semantic_type", "descriptivo") for h in header_elements]
        
        # Detectar si matrix_data contiene diccionarios o strings
        if matrix_data and matrix_data[0] and isinstance(matrix_data[0][0], dict):
            matrix_texts = [[cell.get("cell_text", "") for cell in row] for row in matrix_data]
        else:
            matrix_texts = matrix_data

        # Construir el diccionario en el orden deseado
        if suffix == "math_max_matrix":
            simplified_dict = {
                "headers": headers,
                "semantic_types": semantic_types,
                "semantic_format": "math_assigned",  # ← Aquí, en el orden correcto
                "matrix": matrix_texts
            }
        else:
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
            logger.info(f"Matriz simplificada de depuración ({suffix}) guardada en: {output_path}")
        except Exception as e:
            logger.error(f"Error guardando la matriz simplificada de depuración en {output_path}: {e}")

        if suffix == "semantically_corrected_matrix":
            if self.output_config.get('enabled_outputs', {}).get('ground_truth_excel', False):
                self._update_excel_with_matrix(
                    document_id=base_name,
                    headers=headers,
                    semantic_types=semantic_types,
                    matrix_rows=matrix_texts,
                    output_dir=output_dir
                )

    def _update_excel_with_matrix(self,
                                  document_id: str,
                                  headers: List[str],
                                  semantic_types: List[str],
                                  matrix_rows: List[List[str]],
                                  output_dir: str):
        """
        Crea (si no existe) o actualiza el archivo ground_truth_batch.xlsx
        con la matriz semánticamente corregida del documento actual.
        """
        matrix_dict = {
            "document_id": document_id,
            "headers": headers,
            "semantic_types": semantic_types,
            "matrix": matrix_rows
        }

        excel_path = os.path.join(output_dir, "ground_truth_batch.xlsx")
        if os.path.exists(excel_path):
            # Añadir al final del Excel existente
            ok = self.excel_output_handler.append_matrix_to_existing_excel(matrix_dict, excel_path)
            if not ok:
                logger.error("No se pudo añadir la matriz al Excel consolidado.")
        else:
            # Crear un nuevo archivo Excel con la primera matriz
            self.excel_output_handler.save_semantically_corrected_matrices(
                [matrix_dict], output_dir, file_name="ground_truth_batch.xlsx"
            )

    def run_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Método wrapper para compatibilidad con WorkflowManager.
        Procesa una sola imagen y retorna el resultado.
        """
        try:
            result = self.process_document(image_path)
            return result if result else {"error": "No result returned", "image": image_path}
        except Exception as e:
            logger.error(f"Error procesando imagen {image_path}: {e}")
            return {"error": str(e), "image": image_path}


def main():
    """
    Ejecución clásica: procesa un único archivo pasado por --input
    o una carpeta mediante argumentos de argparse.  NO usa la nueva CLI.
    """
    parser = argparse.ArgumentParser(description="PerfectOCR workflow (legacy)")
    parser.add_argument("--input", "-i", required=True, help="Ruta a imagen o carpeta")
    parser.add_argument("--output", "-o", default="./output", help="Directorio de salida")
    args = parser.parse_args()

    workflow = PerfectOCRWorkflow(MASTER_CONFIG_FILE)

    # Procesar 1 imagen o una carpeta (modo sencillo)
    input_path = args.input
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                workflow.process_document(os.path.join(input_path, f), args.output)
    else:
        workflow.process_document(input_path, args.output)

def simple_entrypoint(arg_path: str):
    """
    Llama a PerfectOCRWorkflow igual que antes (1 archivo o carpeta).
    """
    workflow = PerfectOCRWorkflow(MASTER_CONFIG_FILE)

    if os.path.isdir(arg_path):
        for f in os.listdir(arg_path):
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                workflow.process_document(os.path.join(arg_path, f))
    else:
        workflow.process_document(arg_path)

__all__ = ["PerfectOCRWorkflow"]

if __name__ == "__main__":
    # 1 argumento → modo clásico ultra-simple
    if len(sys.argv) == 2:
        simple_entrypoint(sys.argv[1])
    # ≥2 argumentos y el primero es sub-comando CLI → usa Typer
    elif len(sys.argv) > 1 and sys.argv[1] in ("run", "benchmark"):
        from cli.main import app
        app()                         # python main.py run …
    # todo lo demás → legacy argparse (‘--input …’)
    else:
        main()