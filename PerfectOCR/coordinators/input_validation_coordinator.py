# PerfectOCR/coordinators/input_validation_coordinator.py
import os
import logging
import cv2
import time
from typing import Dict, Tuple, Optional, List, Any  # Se añade 'Any'
from core.input_validation.quality_evaluator import ImageQualityEvaluator

logger = logging.getLogger(__name__)

class InputValidationCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.quality_evaluator = ImageQualityEvaluator(config=config.get('quality_assessment_rules', {}))

    # --- INICIO DE LA MODIFICACIÓN ---
    # Se ha reemplazado 'cv2.typing.MatLike' por 'Any' para máxima compatibilidad
    def validate_and_assess_image(self, input_path: str) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]], Optional[Any], float]:
    # --- FIN DE LA MODIFICACIÓN ---
        """
        Carga la imagen y llama al evaluador para obtener un diccionario de planes de corrección.
        Devuelve: (observaciones, diccionario_de_planes, array_de_imagen, tiempo_de_ejecución)
        """
        start_time = time.perf_counter()
        
        try:
            image_array = cv2.imread(input_path)
            if image_array is None:
                return ["error_loading_image"], None, None, time.perf_counter() - start_time

            observations, correction_plans = self.quality_evaluator.evaluate_and_create_correction_plan(image_array)
            return observations, correction_plans, image_array, time.perf_counter() - start_time
            
        except Exception as e:
            logger.error(f"Error en validación de imagen: {e}")
            return ["error_validation"], None, None, time.perf_counter() - start_time