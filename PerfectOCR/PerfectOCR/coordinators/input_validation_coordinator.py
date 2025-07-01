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
        self.project_root = project_root
        
        # OPTIMIZACIÓN: Obtener motores habilitados PRIMERO
        self.enabled_engines = self._get_enabled_engines()
        
        # OPTIMIZACIÓN: Pasar motores habilitados al evaluador
        self.quality_evaluator = ImageQualityEvaluator(
            config=config.get('quality_assessment_rules', {}),
            enabled_engines=self.enabled_engines
        )
        
        logger.info(f"InputValidationCoordinator inicializado para motores: {[k for k, v in self.enabled_engines.items() if v]}")
    
    def _get_enabled_engines(self) -> Dict[str, bool]:
        """Obtiene la configuración de motores habilitados desde YAML."""
        try:
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
        """Carga imagen y genera planes SOLO para motores habilitados."""
        start_time = time.perf_counter()
        
        try:
            image_array = cv2.imread(input_path)
            if image_array is None:
                return ["error_loading_image"], None, None, time.perf_counter() - start_time

            # OPTIMIZACIÓN: El evaluador ya solo procesa motores habilitados
            observations, correction_plans = self.quality_evaluator.evaluate_and_create_correction_plan(image_array)
            
            # Verificar que tenemos planes para motores habilitados
            if not correction_plans:
                logger.warning("No se generaron planes de corrección para ningún motor habilitado")
                return ["no_enabled_engines"], None, None, time.perf_counter() - start_time
            
            validation_time = time.perf_counter() - start_time
            logger.debug(f"Validación completada: {len(correction_plans)} motores, {len(observations)} observaciones ({validation_time:.3f}s)")
            
            return observations, correction_plans, image_array, validation_time
            
        except Exception as e:
            logger.error(f"Error en validación de imagen: {e}")
            return ["error_validation"], None, None, time.perf_counter() - start_time