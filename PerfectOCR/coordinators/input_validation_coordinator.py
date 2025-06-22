# PerfectOCR/coordinators/input_validation_coordinator.py
import os
import logging
import cv2
import time
from typing import Dict, Tuple, Optional, List, Any
from core.input_validation.quality_evaluator import ImageQualityEvaluator

logger = logging.getLogger(__name__)

class InputValidationCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.quality_evaluator = ImageQualityEvaluator(config=config.get('quality_assessment_rules', {}))
        self.project_root = project_root
        
        # NUEVO: Obtener configuración de motores habilitados
        self.enabled_engines = self._get_enabled_engines()
    
    def _get_enabled_engines(self) -> Dict[str, bool]:
        """Obtiene la configuración de motores habilitados desde YAML."""
        try:
            # Importar aquí para evitar dependencias circulares
            from utils.config_loader import ConfigLoader
            config_path = os.path.join(self.project_root, "config", "master_config.yaml")
            
            if os.path.exists(config_path):
                config_loader = ConfigLoader(config_path)
                ocr_config = config_loader.get_ocr_config()
                enabled_engines = ocr_config.get('enabled_engines', {
                    'tesseract': True,
                    'paddleocr': True
                })
                return enabled_engines
        except Exception as e:
            logger.warning(f"Error obteniendo enabled_engines: {e}. Usando configuración por defecto.")
        
        return {'tesseract': True, 'paddleocr': True}

    def validate_and_assess_image(self, input_path: str) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]], Optional[Any], float]:
        """
        Carga la imagen y llama al evaluador para obtener un diccionario de planes de corrección.
        Solo genera planes para motores habilitados.
        """
        start_time = time.perf_counter()
        
        try:
            image_array = cv2.imread(input_path)
            if image_array is None:
                return ["error_loading_image"], None, None, time.perf_counter() - start_time

            observations, correction_plans = self.quality_evaluator.evaluate_and_create_correction_plan(image_array)
            
            # NUEVO: Filtrar planes solo para motores habilitados
            filtered_plans = {}
            for engine, plan in correction_plans.items():
                if engine in self.enabled_engines and self.enabled_engines[engine]:
                    filtered_plans[engine] = plan
                elif engine == 'spatial_analysis':
                    # El análisis espacial siempre se incluye si está en los planes
                    filtered_plans[engine] = plan
                else:
                    logger.debug(f"Motor {engine} deshabilitado, omitiendo plan de corrección")
            
            if not filtered_plans:
                logger.warning("Ningún motor habilitado encontrado en los planes de corrección")
                return ["no_enabled_engines"], None, None, time.perf_counter() - start_time
            
            return observations, filtered_plans, image_array, time.perf_counter() - start_time
            
        except Exception as e:
            logger.error(f"Error en validación de imagen: {e}")
            return ["error_validation"], None, None, time.perf_counter() - start_time