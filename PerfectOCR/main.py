# PerfectOCR/main.py
import json
import yaml
import os
import sys
import logging
import cv2
import numpy as np
import time
from typing import Dict, Optional, Any, Tuple, List
from coordinators.preprocessing_coordinator import PreprocessingCoordinator
from coordinators.ocr_coordinator import OCREngineCoordinator
from coordinators.geovectorization_coordinator import GeometricCosineCoordinator
#from coordinators.table_extractor_coordinator import TableExtractorCoordinator
#from coordinators.postprocessing_coordinator import PostprocessingCoordinator
#from coordinators.text_cleaning_coordinator import TextCleaningCoordinator
from utils.output_handlers import JsonOutputHandler, ExcelOutputHandler
from utils.encoders import NumpyEncoder
from utils.config_loader import ConfigLoader
from utils.batch_tools import get_optimal_workers
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuración conservadora para estabilidad
os.environ.update({
    'OMP_NUM_THREADS': '1',        # Conservador para evitar contención
    'MKL_NUM_THREADS': '2',        # Conservador
    'FLAGS_use_mkldnn': '1',       # Mantener (es estable en main thread)
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "Perfectocr.txt")
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def setup_logging():
    """Configura el sistema de logging centralizado."""
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
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
        # PRINCIPIO: Main inicializa el ConfigLoader pero NO distribuye configuraciones
        self.config_loader = ConfigLoader(master_config_path)
        self.config = self.config_loader.config
        self.project_root = PROJECT_ROOT
        
        self._preprocessing_coordinator: Optional[PreprocessingCoordinator] = None
        self._ocr_coordinator: Optional[OCREngineCoordinator] = None
        self._geovectorizator_coordinator: Optional[GeometricCosineCoordinator] = None
        #self._table_extractor_coordinator: Optional[TableExtractorCoordinator] = None
        #self._postprocessing_coordinator: Optional[PostprocessingCoordinator] = None
        #self._text_cleaning_coordinator: Optional[TextCleaningCoordinator] = None
        
        # Herramientas auxiliares para gestión de salida
        output_config = self.config.get('output_config', {})
        self.json_output_handler = JsonOutputHandler(config=output_config)
        self.excel_output_handler = ExcelOutputHandler()
        
        # Configuraciones específicas necesarias para el flujo principal
        self.workflow_config = self.config_loader.get_workflow_config()
        self.output_flags = self.config.get('output_config', {}).get('enabled_outputs', {})
        
        # Inicialización básica del coordinador OCR que se necesita siempre
        self._ocr_coordinator = OCREngineCoordinator(
            config=self.config_loader.get_ocr_config(),
            project_root=self.project_root,
            output_flags=self.output_flags,
            workflow_config=self.workflow_config
        )
        
    @property
    def preprocessing_coordinator(self) -> PreprocessingCoordinator:
        """Acceso lazy al coordinador de preprocesamiento que obtiene su configuración del ConfigLoader."""
        if self._preprocessing_coordinator is None:
            self._preprocessing_coordinator = PreprocessingCoordinator(
                config=self.config_loader.get_preprocessing_coordinator_config(),
                project_root=self.project_root
            )
        return self._preprocessing_coordinator

    @property
    def ocr_coordinator(self) -> OCREngineCoordinator:
        """Acceso al coordinador OCR ya inicializado."""
        return self._ocr_coordinator

    @property
    def geovectorization_coordinator(self) -> GeometricCosineCoordinator:
        """Instancia perezosa para el coordinador de vectorización."""
        if self._geovectorizator_coordinator is None:
            self._geovectorizator_coordinator = GeometricCosineCoordinator(
                config=self.config_loader.config,  # Pasar configuración completa
                project_root=self.project_root,
                output_flags=self.output_flags
            )
        return self._geovectorizator_coordinator
    
#    @property
#    def table_extractor_coordinator(self) -> TableExtractorCoordinator:
#        """Instancia perezosa para el coordinador extractor de tablas."""
#        if self._table_extractor_coordinator is None:
#            self._table_extractor_coordinator = TableExtractorCoordinator(
#                config=self.config_loader.get_table_extractor_config(),
#                project_root=self.project_root,
#                output_flags=self.output_flags
#            )
#        return self._table_extractor_coordinator

#    @property
#    def postprocessing_coordinator(self) -> PostprocessingCoordinator:
#        """Instancia perezosa para el coordinador de postprocesamiento."""
#        if self._postprocessing_coordinator is None:
#            self._postprocessing_coordinator = PostprocessingCoordinator(
#                config=self.config_loader.get_postprocessing_config(),
#                project_root=self.project_root,
#                output_flags=self.output_flags
#            )
#        return self._postprocessing_coordinator

#    @property
#    def text_cleaning_coordinator(self) -> TextCleaningCoordinator:
#        """Instancia perezosa para el coordinador de limpieza de texto."""
#        if self._text_cleaning_coordinator is None:
#            text_cleaning_config = self.config_loader.get_text_cleaning_config()
#            self._text_cleaning_coordinator = TextCleaningCoordinator(
#                config=text_cleaning_config['text_cleaning'],
#                output_flags=text_cleaning_config['output_flags']
#            )
#        return self._text_cleaning_coordinator


    def process_document(self, input_path: str, output_dir_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        workflow_start = time.perf_counter()
        processing_times_summary: Dict[str, float] = {}
        original_file_name = os.path.basename(input_path)
        base_name = os.path.splitext(original_file_name)[0]

        # CREAR ProcessingJob
        from domain.main_job import ProcessingJob
        job = ProcessingJob(source_uri=input_path)

        ocr_images_dict = None
        ocr_results_payload = None

        # --- Cargar imagen ---
        image_array = cv2.imread(input_path)
        if image_array is None:
            return self._build_error_response("error_loading_image", original_file_name, "No se pudo cargar la imagen", "load")

        job.image_data = image_array

        # FASE 1: PREPROCESAMIENTO (ya incluye evaluación interna)
        phase1_start = time.perf_counter()
        preproc_results, time_prep = self.preprocessing_coordinator.apply_preprocessing_pipelines(
            image_array,
            input_path
        )
        phase1_time = time.perf_counter() - phase1_start
        processing_times_summary["1_preprocessing"] = round(time_prep, 4)

        logger.info(f"Preprocesamiento: {phase1_time:.3f}s")
        
        if not preproc_results or "ocr_images" not in preproc_results:
            logger.critical("No hay imágenes pre-procesadas para OCR. Abortando.")
            return self._build_error_response("error_preprocessing", original_file_name,
                                                "No hay imágenes para OCR", "preprocessing")
        
        ocr_images_dict = preproc_results["ocr_images"]

        workflow_config = self.config_loader.get_workflow_config()
        current_output_dir = output_dir_override if output_dir_override else workflow_config.get('output_folder')
        os.makedirs(current_output_dir, exist_ok=True)
        
        # FASE 3: OCR
        phase4_start = time.perf_counter()
        ocr_results_payload, time_ocr = self.ocr_coordinator.run_ocr_parallel(ocr_images_dict, original_file_name)
        phase4_time = time.perf_counter() - phase4_start
        processing_times_summary["2_ocr"] = round(time_ocr, 4)
        
        logger.info(f"OCR: {time_ocr:.3f}s")
        
        if not self.ocr_coordinator.validate_ocr_results(ocr_results_payload, original_file_name):
            return self._build_error_response("error_ocr", original_file_name, "OCR sin resultados", "ocr_validation")

        ocr_results_json_path = ocr_results_payload.get("ocr_raw_json_path")

        # FASE 4: Vectorización y agrupación de líneas
        phase4_start = time.perf_counter()
        vectorization_payload = self.geovectorization_coordinator.orchestrate_vectorization_and_detection(
            ocr_results_payload=ocr_results_payload,
            doc_id=base_name
        )
        phase4_time = time.perf_counter() - phase4_start
        processing_times_summary["3_vectorization"] = round(phase4_time, 4)
        logger.info(f"Fase de vectorización tomó: {phase4_time:.3f}s.")

        # Main recibe alerta: "Vectorización completada"
        job.status = "COMPLETED"
        job.final_result = vectorization_payload

        final_payload = vectorization_payload

        # RESPUESTA FINAL
        final_response = self._build_final_response(
            original_file_name,
            ocr_results_json_path,
            final_payload
        )
        
        total_workflow_time = time.perf_counter() - workflow_start
        processing_times_summary["total_workflow"] = round(total_workflow_time, 4)
        
        if 'metadata' not in final_response: 
            final_response['metadata'] = {}
        final_response['metadata']['processing_times_seconds'] = processing_times_summary
        
        logger.info(f"Total: {total_workflow_time:.3f}s")
        return final_response
    
    def _build_error_response(self, status: str, filename: str, message: str, stage: Optional[str] = None) -> dict:
        error_details = {"message": message}
        if stage: error_details["stage"] = stage
        return {"document_id": filename, "status_overall_workflow": status, "error_details": error_details }

    def _build_final_response(self, filename: str, ocr_path: Optional[str], processing_payload: dict) -> dict:
        status = processing_payload.get("status", "error_unknown")
        final_status = "success" if status.startswith("success") else status
        
        # Guardar artefactos si se solicita
        output_config = self.config_loader.config.get('output_config', {})
        output_flags = output_config.get('enabled_outputs', {})
        
        if output_flags.get('line_grouping_results', False):
            output_dir = output_config.get('output_folder', './output')
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.json_output_handler.save(
                data=processing_payload,
                output_dir=output_dir,
                file_name_with_extension=f"{base_name}_line_grouping_results.json"
            )

        # La respuesta principal no incluirá la lista completa de líneas para ser más ligera.
        summary_payload = processing_payload.copy()
        if 'lines' in summary_payload:
            del summary_payload['lines']
            
        outputs = {
            "ocr_raw_json": ocr_path
            # "structured_table_json" ya no se genera en este punto
        }
        summary = {"processing_status": status, "details": summary_payload}
        return {"document_id": filename, "status_overall_workflow": final_status, "outputs": outputs, "summary": summary}

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

__all__ = ["PerfectOCRWorkflow"]

if __name__ == "__main__":
    # Cargar configuración desde YAML
    config_loader = ConfigLoader(MASTER_CONFIG_FILE)
    workflow_config = config_loader.get_workflow_config()
    input_folder = workflow_config.get('input_folder')
    output_folder = workflow_config.get('output_folder')
    batch_mode = workflow_config.get('batch_mode', True)  # Por defecto batch

    workflow = PerfectOCRWorkflow(MASTER_CONFIG_FILE)

    if not os.path.isdir(input_folder):
        logger.critical(f"La carpeta de entrada especificada no existe o no es un directorio: '{input_folder}'")
        sys.exit(1)

    logger.info(f"Buscando imágenes en la carpeta: '{input_folder}'")
    archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
    if not archivos:
        logger.critical(f"No se encontraron imágenes válidas con extensiones {VALID_IMAGE_EXTENSIONS} en la carpeta de entrada.")
        logger.critical("Por favor, asegúrate de que las imágenes estén en el directorio correcto y el programa terminará.")
        sys.exit(1)

    if batch_mode:
        for f in archivos:
            workflow.process_document(os.path.join(input_folder, f), output_folder)
    else:
        # Solo procesa el primer archivo válido
        workflow.process_document(os.path.join(input_folder, archivos[0]), output_folder)