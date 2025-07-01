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
import threading
import subprocess
import requests
from typing import Dict, Optional, Any, Tuple, List
from coordinators.input_validation_coordinator import InputValidationCoordinator
from coordinators.preprocessing_coordinator import PreprocessingCoordinator
from coordinators.reception_coordinator import ReceptionCoordinator
from coordinators.ocr_coordinator import OCREngineCoordinator
from coordinators.table_extractor_coordinator import TableExtractorCoordinator
from coordinators.postprocessing_coordinator import PostprocessingCoordinator
from coordinators.text_cleaning_coordinator import TextCleaningCoordinator
from utils.output_handlers import JsonOutputHandler, ExcelOutputHandler
from utils.encoders import NumpyEncoder
from utils.config_loader import ConfigLoader
from utils.batch_tools import get_optimal_workers
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        self._input_validation_coordinator: Optional[InputValidationCoordinator] = None
        self._preprocessing_coordinator: Optional[PreprocessingCoordinator] = None
        self._ocr_coordinator: Optional[OCREngineCoordinator] = None
        self._table_extractor_coordinator: Optional[TableExtractorCoordinator] = None
        self._postprocessing_coordinator: Optional[PostprocessingCoordinator] = None
        self._text_cleaning_coordinator: Optional[TextCleaningCoordinator] = None
        self.json_output_handler = JsonOutputHandler(config=self.output_config)
        self.excel_output_handler = ExcelOutputHandler()
        logger.debug("PerfectOCRWorkflow listo para inicialización bajo demanda de coordinadores.")
        config_loader = ConfigLoader(MASTER_CONFIG_FILE)
        table_extractor_config = config_loader.get_table_extractor_config()
        output_flags = config_loader.config.get('output_config', {}).get('enabled_outputs', {})
        self.output_flags = output_flags

        # Al crear el workflow o el coordinador:
        self._table_extractor_coordinator = TableExtractorCoordinator(
            config=table_extractor_config,
            project_root=self.project_root,
            output_flags=output_flags
        )

        self._postprocessing_coordinator = PostprocessingCoordinator(
            config=self.postprocessing_config,
            project_root=self.project_root,
            output_flags=output_flags
        )

        self._text_cleaning_coordinator = TextCleaningCoordinator(
            config=self.text_cleaning_config,
            output_flags=output_flags
        )

        self._ocr_coordinator = OCREngineCoordinator(
            config=self.ocr_config,
            project_root=self.project_root,
            output_flags=output_flags
        )

        # CONFIGURACIÓN DEL MÓDULO DE RECEPCIÓN
        self._reception_enabled = None
        self._reception_process = None
        self.reception_config = self.config.get('enhancement_service_api', {})  # Mantener clave por compatibilidad
        self._reception_coordinator = None

        self._ensure_enhanced_folders_exist()

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
    def ocr_coordinator(self) -> OCREngineCoordinator:
        return self._ocr_coordinator

    @property
    def table_extractor_coordinator(self) -> TableExtractorCoordinator:
        return self._table_extractor_coordinator

    @property
    def postprocessing_coordinator(self) -> PostprocessingCoordinator:
        return self._postprocessing_coordinator

    @property
    def text_cleaning_coordinator(self) -> TextCleaningCoordinator:
        return self._text_cleaning_coordinator

    @property
    def reception_coordinator(self):
        if self._reception_coordinator is None:
            # Leemos la ruta del script del servidor de recepción desde el YAML
            server_script = self.reception_config.get("server_script_path", "core/reception/reception_server.py")
            script_path = os.path.join(self.project_root, server_script)
            
            if not os.path.exists(script_path):
                logger.critical(f"No se encuentra el script del servidor de recepción en: {script_path}")
                raise FileNotFoundError(f"El script del servidor {server_script} no existe.")

            self._reception_coordinator = ReceptionCoordinator(script_path)
        return self._reception_coordinator

    def _check_reception_enabled(self) -> bool:
        """Verifica si el módulo de recepción está habilitado en configuración."""
        if self._reception_enabled is None:
            reception_config = self.config.get('enhancement_service_api', {})
            self._reception_enabled = reception_config.get('enabled', False)
        return self._reception_enabled

    def _check_for_received_images(self, original_file_name: str) -> Optional[Dict[str, Any]]:
        """Busca imágenes preprocesadas recibidas por el módulo de recepción."""
        temp_dir = self.reception_config.get("enhanced_input_folder", "./enhanced_input")
        base_name = os.path.splitext(original_file_name)[0]
        
        # Buscar archivos procesados
        tesseract_path = os.path.join(temp_dir, f"{base_name}_tesseract_preprocessed.png")
        paddleocr_path = os.path.join(temp_dir, f"{base_name}_paddleocr_preprocessed.png")
        
        if os.path.exists(tesseract_path) and os.path.exists(paddleocr_path):
            logger.info(f"📥 Encontradas imágenes preprocesadas para {base_name}")
            
            # Cargar las imágenes
            tess_img = cv2.imread(tesseract_path, cv2.IMREAD_UNCHANGED)
            paddle_img = cv2.imread(paddleocr_path, cv2.IMREAD_UNCHANGED)
            
            if tess_img is not None and paddle_img is not None:
                # Limpiar archivos temporales
                try:
                    os.remove(tesseract_path)
                    os.remove(paddleocr_path)
                except:
                    pass
                
                return {
                    "ocr_images": {"tesseract": tess_img, "paddleocr": paddle_img},
                    "noise_regions": []
                }
        
        return None

    def process_document(self, input_path: str, output_dir_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        workflow_start = time.perf_counter()
        processing_times_summary: Dict[str, float] = {}
        original_file_name = os.path.basename(input_path)
        base_name = os.path.splitext(original_file_name)[0]
        logger.info(f"Procesando: {original_file_name}")

        ocr_images_dict = None
        noise_regions = []
        ocr_results_payload = None

        use_reception_service = self._check_reception_enabled()

        if use_reception_service:
            # Asegurar que las carpetas existan
            self._ensure_enhanced_folders_exist()
            logger.info("📨 Modo de recepción de imágenes preprocesadas activado.")
            self.reception_coordinator.start_reception_server() # Lanza el servidor de RECEPCIÓN
            
            # Disparamos el procesamiento en el sistema externo (ej: Inspiron)
            external_trigger_url = self.reception_config.get("inspiron_trigger_url")
            if not external_trigger_url:
                logger.error("La URL 'inspiron_trigger_url' no está configurada en el YAML.")
            else:
                try:
                    logger.info(f"▶️  Solicitando procesamiento externo en {external_trigger_url} para '{base_name}'...")
                    connection_payload = {"action": "connect", "client": "main_system"}
                    requests.post(external_trigger_url, json=connection_payload, timeout=5)
                except requests.RequestException as e:
                    logger.error(f"❌ No se pudo contactar al sistema externo: {e}. ¿Está el servidor activo?")

            # Esperamos que las imágenes preprocesadas lleguen vía módulo de recepción
            received_images = self._check_for_received_images(original_file_name)
            if received_images:
                logger.info("✅ Imágenes preprocesadas recibidas correctamente!")
                ocr_images_dict = received_images["ocr_images"]
                noise_regions = received_images["noise_regions"]
            else:
                logger.warning("⏰ Timeout esperando imágenes. Continuando en modo local.")
                use_reception_service = False # Fallback
        
        # --- Si el modo de recepción se desactivó o falló, o si ya se obtuvieron resultados ---
        if ocr_results_payload:
             # Si tenemos resultados, saltamos directo a la reconstrucción
            logger.info("🔄 Usando imágenes recibidas, saltando a reconstrucción de líneas.")
        else:
            if use_reception_service: # Si llegamos aquí es porque hubo un fallback
                logger.info("🔄 Cambiando a flujo de trabajo local debido a error en recepción de imágenes.")

            # FLUJO NORMAL LOCAL
            if not use_reception_service:  # Condición faltante
                # FASE 1: VALIDACIÓN
                phase1_start = time.perf_counter()
                quality_observations, correction_plans, image_loaded, time_val = self.input_validation_coordinator.validate_and_assess_image(input_path)
                phase1_time = time.perf_counter() - phase1_start
                processing_times_summary["1_input_validation"] = round(time_val, 4)
                
                logger.info(f"Validación: {time_val:.3f}s")
                
                if image_loaded is None or not correction_plans:
                    return self._build_error_response("error_input_validation", original_file_name, "Fallo en validación", "input_validation")

                # FASE 2: PREPROCESAMIENTO
                phase2_start = time.perf_counter()
                preproc_results, time_prep = self.preprocessing_coordinator.apply_preprocessing_pipelines(image_loaded, correction_plans, input_path)
                phase2_time = time.perf_counter() - phase2_start
                processing_times_summary["2_preprocessing"] = round(time_prep, 4)
                
                logger.info(f"Preprocesamiento: {time_prep:.3f}s")
                
                if not preproc_results or "ocr_images" not in preproc_results:
                    return self._build_error_response("error_preprocessing", original_file_name, "Fallo en preprocesamiento", "preprocessing")
                
                ocr_images_dict = preproc_results["ocr_images"]
                noise_regions = []

        current_output_dir = output_dir_override if output_dir_override else self.workflow_config.get('output_folder')
        os.makedirs(current_output_dir, exist_ok=True)
        
        # FASE 3: OCR
        phase4_start = time.perf_counter()
        ocr_results_payload, time_ocr = self.ocr_coordinator.run_ocr_parallel(ocr_images_dict, noise_regions, original_file_name)
        phase4_time = time.perf_counter() - phase4_start
        processing_times_summary["4_ocr"] = round(time_ocr, 4)
        
        logger.info(f"OCR: {time_ocr:.3f}s")
        
        if not self.ocr_coordinator.validate_ocr_results(ocr_results_payload, original_file_name):
            return self._build_error_response("error_ocr", original_file_name, "OCR sin resultados", "ocr_validation")
        
        ocr_results_json_path = ocr_results_payload.get("ocr_raw_json_path")

        # FASE 4: Reconstrucción de líneas
        phase5_start = time.perf_counter()
        reconstructed_lines = self.table_extractor_coordinator.reconstruct_lines(
            ocr_results=ocr_results_payload,
            base_name=base_name,
            output_dir=current_output_dir
        )
        phase5_time = time.perf_counter() - phase5_start
        processing_times_summary["5_line_reconstruction"] = round(phase5_time, 4)
        logger.info(f"Reconstrucción: {phase5_time:.3f}s")

        # FASE 5: Limpieza de texto
        phase55_start = time.perf_counter()
        cleaned_lines, corrections_count = self.text_cleaning_coordinator.clean_reconstructed_lines(
            reconstructed_lines_by_engine=reconstructed_lines,
            output_dir=current_output_dir,
            doc_id=base_name
        )
        phase55_time = time.perf_counter() - phase55_start
        processing_times_summary["5.5_text_cleaning"] = round(phase55_time, 4)
        logger.info(f"Limpieza: {phase55_time:.3f}s")

        # FASE 6: Extracción de tabla
        phase6_start = time.perf_counter()
        table_extraction_payload = self.table_extractor_coordinator.extract_table_from_cleaned_lines(
            cleaned_lines=cleaned_lines,
            base_name=base_name,
            output_dir=current_output_dir
        )
        phase6_time = time.perf_counter() - phase6_start
        processing_times_summary["6_table_extraction"] = round(phase6_time, 4)
        logger.info(f"Extracción: {phase6_time:.3f}s")

        # FASE 7: Postprocesamiento semántico
        phase7_start = time.perf_counter()
        table_extraction_payload["output_dir"] = current_output_dir
        table_extraction_payload["doc_id"] = base_name

        semantically_corrected_payload = self.postprocessing_coordinator.correct_table_structure(
            extraction_payload=table_extraction_payload
        )
        phase7_time = time.perf_counter() - phase7_start
        processing_times_summary["7_semantic_correction"] = round(phase7_time, 4)
        logger.info(f"Corrección semántica: {phase7_time:.3f}s")

        # RESPUESTA FINAL
        final_response = self._build_final_response(
            original_file_name,
            ocr_results_json_path,
            semantically_corrected_payload
        )
        
        total_workflow_time = time.perf_counter() - workflow_start
        processing_times_summary["total_workflow"] = round(total_workflow_time, 4)
        
        if 'metadata' not in final_response: 
            final_response['metadata'] = {}
        final_response['metadata']['processing_times_seconds'] = processing_times_summary
        
        logger.info(f"Total: {total_workflow_time:.3f}s")

        if use_reception_service:
            self.reception_coordinator.stop_reception_server()

        return final_response
    
    def _build_error_response(self, status: str, filename: str, message: str, stage: Optional[str] = None) -> dict:
        error_details = {"message": message}
        if stage: error_details["stage"] = stage
        return {"document_id": filename, "status_overall_workflow": status, "error_details": error_details }

    def _build_final_response(self, filename: str, ocr_path: Optional[str], table_payload: dict) -> dict:
        status = table_payload.get("status", "error_unknown")
        final_status = "success" if status.startswith("success") else status
        outputs = {
            "ocr_raw_json": ocr_path,
            "structured_table_json": table_payload.get("outputs", {}).get("structured_table_json_path")
        }
        summary = {"table_extraction_status": status, "message": table_payload.get("message")}
        return {"document_id": filename, "status_overall_workflow": final_status, "outputs": outputs, "summary": summary}

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

    def _ensure_enhanced_folders_exist(self):
        """Asegura que las carpetas necesarias para el modo enhanced existan."""
        enhanced_input = self.reception_config.get("enhanced_input_folder")
        
        if enhanced_input:
            os.makedirs(enhanced_input, exist_ok=True)

    def handle_received_images(self, doc_id, images_dict, noise_regions):
        """Método especial para procesar imágenes recibidas"""
        if not hasattr(self, 'reception_results'):
            self.reception_results = {}
        ocr_result, time_ocr = self.ocr_coordinator.run_ocr_parallel(
            images_dict, noise_regions, doc_id
        )
        # Guardar resultado para consulta posterior
        self.reception_results[doc_id] = ocr_result

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
    # Cargar configuración desde YAML
    config_loader = ConfigLoader(MASTER_CONFIG_FILE)
    workflow_config = config_loader.get_workflow_config()
    input_folder = workflow_config.get('input_folder')
    output_folder = workflow_config.get('output_folder')
    batch_mode = workflow_config.get('batch_mode', True)  # Por defecto batch

    workflow = PerfectOCRWorkflow(MASTER_CONFIG_FILE)

    archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
    if not archivos:
        print("No se encontraron imágenes válidas en la carpeta de entrada.")
        sys.exit(1)

    if batch_mode:
        for f in archivos:
            workflow.process_document(os.path.join(input_folder, f), output_folder)
    else:
        # Solo procesa el primer archivo válido
        workflow.process_document(os.path.join(input_folder, archivos[0]), output_folder)