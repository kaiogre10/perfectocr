# PerfectOCR/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple
from core.preprocessing.quality_validator import ImageQualityEvaluator
from utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
        
        # Un único worker que sabe evaluar Y corregir.
        self.quality_worker = ImageQualityEvaluator(
            config=config.get('quality_assessment_rules', {})
        )
        self.image_saver = ImageOutputHandler()
        
    def _save_preprocessed_image(self, image_to_save: np.ndarray, original_input_path: str):
        """Guarda la imagen preprocesada en el directorio de salida."""
        if not original_input_path:
            logger.warning("No se proporcionó 'original_input_path', no se puede guardar la imagen preprocesada con un nombre de archivo único.")
            return

        try:
            output_folder = self.workflow_config.get('output_folder')
            if not output_folder:
                logger.error("La 'output_folder' no está definida en la configuración del workflow.")
                return

            preprocessed_dir = os.path.join(output_folder, "preprocessed")
            
            base_name = os.path.basename(original_input_path)
            file_name, _ = os.path.splitext(base_name)
            output_filename = f"{file_name}_preprocessed.png"

            self.image_saver.save(image_to_save, preprocessed_dir, output_filename)
        except Exception as e:
            logger.error(f"Error al intentar guardar la imagen preprocesada para {original_input_path}: {e}", exc_info=True)

    def apply_preprocessing_pipelines(
        self,
        image_array: np.ndarray,
        input_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Inicia el pipeline de preprocesamiento y devuelve la imagen final.
        """
        start_time = time.perf_counter()

        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array

        try:
            # Delegar TODO el trabajo al worker.
            corrected_image = self.quality_worker.process_image(gray_image)
            
            if self.output_config.get('enabled_outputs', {}).get('preprocessed_image', False):
                self._save_preprocessed_image(corrected_image, input_path)

            # Empaquetar el resultado para el siguiente coordinador (OCR).
            ocr_images = {'paddleocr': corrected_image}
            logger.info("Preprocessing: Pipeline completado exitosamente por el worker.")

        except Exception as e:
            logger.error(f"Preprocessing: Falló el pipeline del worker: {e}", exc_info=True)
            ocr_images = {}

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Preprocessing: Proceso completado en {elapsed_time:.3f}s")
        
        return {"ocr_images": ocr_images}, elapsed_time