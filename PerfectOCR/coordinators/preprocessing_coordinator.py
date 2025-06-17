# PerfectOCR/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    def __init__(self, config: Dict = None, project_root: str = None):
        """
        Inicialización compatible con interfaz antigua y nueva.
        
        Args:
            config: Configuración legacy (se ignora, se usa YAML)
            project_root: Ruta del proyecto para encontrar master_config.yaml
        """
        self.project_root = project_root or "."
        
        # Importar aquí para evitar dependencias circulares
        try:
            from utils.config_loader import ConfigLoader, PreprocessingEngineConfig
            from core.preprocessing.image_corrector import ImageCorrector
            from core.preprocessing import toolbox
            
            # Cargar configuración desde master_config.yaml
            config_path = os.path.join(self.project_root, "config", "master_config.yaml")
            
            if os.path.exists(config_path):
                self.config_loader = ConfigLoader(config_path)
                self.engine_configs = self.config_loader.get_preprocessing_config()
                self.max_workers = self.config_loader.get_max_workers_for_cpu()
                self.corrector = ImageCorrector(self.config_loader)
                self._yaml_mode = True
                logger.info(f"PreprocessingCoordinator inicializado en modo YAML con {len(self.engine_configs)} motores")
            else:
                # Fallback al modo legacy si no existe master_config.yaml
                logger.warning(f"master_config.yaml no encontrado en {config_path}. Usando modo legacy.")
                self._init_legacy_mode(config)
                
        except ImportError as e:
            logger.warning(f"No se pudo importar utils.config_loader: {e}. Usando modo legacy.")
            self._init_legacy_mode(config)
    
    def _init_legacy_mode(self, config: Dict = None):
        """Inicialización en modo legacy para compatibilidad."""
        from core.preprocessing.image_corrector import ImageCorrector
        
        self._yaml_mode = False
        self.corrector = ImageCorrector()
        self.max_workers = min(os.cpu_count() - 2, 6) if os.cpu_count() else 4
        
        # Configuraciones hardcodeadas para modo legacy
        self.engine_configs = {
            'tesseract': {
                'needs_binarization': True,
                'invert_binary': False,
                'default_binarization': {'block_size': 31, 'c_value': 7}
            },
            'paddleocr': {
                'needs_binarization': False
            },
            'spatial_analysis': {
                'needs_binarization': True,
                'invert_binary': True,
                'default_binarization': {'block_size': 15, 'c_value': 5}
            }
        }
        
        logger.info("PreprocessingCoordinator inicializado en modo legacy")

    def apply_preprocessing_pipelines(
        self,
        image_array: np.ndarray,
        correction_plans: Dict[str, Any],
        image_path_for_log: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Genera imágenes optimizadas. Compatible con ambos modos.
        """
        start_time = time.perf_counter()
        
        if not correction_plans:
            logger.warning(f"Plan de corrección vacío para {image_path_for_log}")
            return None, time.perf_counter() - start_time

        # Conversión única a escala de grises
        gray_image = self._ensure_grayscale(image_array)
        
        # Procesamiento paralelo
        if self._yaml_mode:
            results = self._process_engines_parallel_yaml(gray_image, correction_plans, image_path_for_log)
        else:
            results = self._process_engines_parallel_legacy(gray_image, correction_plans, image_path_for_log)
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Preprocesamiento completado en {elapsed_time:.3f}s para {image_path_for_log}")
        
        return results, elapsed_time

    def _ensure_grayscale(self, image_array: np.ndarray) -> np.ndarray:
        """Convierte a escala de grises solo si es necesario."""
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array

    def _process_engines_parallel_yaml(
        self, 
        gray_image: np.ndarray, 
        correction_plans: Dict[str, Any],
        image_path_for_log: str
    ) -> Dict[str, Any]:
        """Procesamiento usando configuración YAML."""
        # Filtrar solo motores configurados en YAML
        valid_plans = {
            engine: plan for engine, plan in correction_plans.items() 
            if engine in self.engine_configs
        }
        
        if not valid_plans:
            logger.warning(f"Ningún motor válido encontrado en correction_plans para {image_path_for_log}")
            return {"ocr_images": {}, "spatial_image": None, "preprocessing_parameters_used": correction_plans}
        
        # Ajustar workers según planes válidos
        active_workers = min(len(valid_plans), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            futures = {
                engine: executor.submit(
                    self._process_single_engine_yaml, 
                    gray_image, 
                    plan, 
                    engine
                )
                for engine, plan in valid_plans.items()
            }
            
            # Recoger resultados
            ocr_images = {}
            spatial_image = None
            
            for engine, future in futures.items():
                try:
                    result = future.result()
                    if engine == 'spatial_analysis':
                        spatial_image = result
                    else:
                        ocr_images[engine] = result
                except Exception as e:
                    logger.error(f"Error procesando {engine} para {image_path_for_log}: {e}")
        
        return {
            "ocr_images": ocr_images,
            "spatial_image": spatial_image,
            "preprocessing_parameters_used": correction_plans
        }

    def _process_engines_parallel_legacy(
        self, 
        gray_image: np.ndarray, 
        correction_plans: Dict[str, Any],
        image_path_for_log: str
    ) -> Dict[str, Any]:
        """Procesamiento usando modo legacy."""
        from core.preprocessing import toolbox
        
        active_workers = min(len(correction_plans), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            futures = {}
            
            # Procesar cada plan
            for engine, plan in correction_plans.items():
                if engine == 'tesseract':
                    futures[engine] = executor.submit(self._process_for_tesseract_legacy, gray_image, plan)
                elif engine == 'paddleocr':
                    futures[engine] = executor.submit(self._process_for_paddleocr_legacy, gray_image, plan)
                elif engine == 'spatial_analysis':
                    futures['spatial'] = executor.submit(self._process_for_spatial_analysis_legacy, gray_image, plan)
            
            # Recoger resultados
            ocr_images = {}
            spatial_image = None
            
            for key, future in futures.items():
                try:
                    result = future.result()
                    if key == 'spatial':
                        spatial_image = result
                    else:
                        ocr_images[key] = result
                except Exception as e:
                    logger.error(f"Error procesando {key} para {image_path_for_log}: {e}")
        
        return {
            "ocr_images": ocr_images,
            "spatial_image": spatial_image,
            "preprocessing_parameters_used": correction_plans
        }

    def _process_single_engine_yaml(
        self, 
        gray_image: np.ndarray, 
        correction_plan: Dict[str, Any], 
        engine: str
    ) -> np.ndarray:
        """Procesamiento usando configuración YAML."""
        from core.preprocessing import toolbox
        
        engine_config = self.engine_configs[engine]
        
        # Combinar correction_plan con configuración YAML
        enhanced_plan = self._merge_plan_with_yaml_config(correction_plan, engine_config)
        
        # Aplicar correcciones estándar
        corrected_image = self.corrector.apply_grayscale_corrections(gray_image.copy(), enhanced_plan)
        
        # Aplicar binarización si el motor la necesita
        if engine_config.needs_binarization:
            binarization_params = enhanced_plan.get('binarization', engine_config.default_binarization)
            
            corrected_image = toolbox.apply_binarization(
                corrected_image,
                block_size=binarization_params['block_size'],
                c_value=binarization_params['c_value'],
                invert=engine_config.invert_binary
            )
        
        return corrected_image

    def _process_for_tesseract_legacy(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """Procesamiento legacy para Tesseract."""
        from core.preprocessing import toolbox
        
        corrected_image = self.corrector.apply_grayscale_corrections(gray_image.copy(), correction_plan)
        
        binarization_params = correction_plan.get('binarization', {'block_size': 31, 'c_value': 7})
        
        return toolbox.apply_binarization(
            corrected_image,
            block_size=binarization_params['block_size'],
            c_value=binarization_params['c_value'],
            invert=False  # Texto negro sobre blanco
        )

    def _process_for_paddleocr_legacy(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """Procesamiento legacy para PaddleOCR."""
        return self.corrector.apply_grayscale_corrections(gray_image.copy(), correction_plan)

    def _process_for_spatial_analysis_legacy(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """Procesamiento legacy para análisis espacial."""
        from core.preprocessing import toolbox
        
        corrected_image = self.corrector.apply_grayscale_corrections(gray_image.copy(), correction_plan)
        
        binarization_params = correction_plan.get('binarization', {'block_size': 15, 'c_value': 5})
        
        return toolbox.apply_binarization(
            corrected_image,
            block_size=binarization_params['block_size'],
            c_value=binarization_params['c_value'],
            invert=True  # Texto blanco sobre negro
        )

    def _merge_plan_with_yaml_config(
        self,
        correction_plan: Dict[str, Any],
        engine_config  # ← ahora puede ser dataclass o dict legacy
    ) -> Dict[str, Any]:
        """
        Combina el correction_plan con la configuración proveniente de YAML.
        Es compatible tanto con dataclass (modo YAML) como con dict (modo legacy).
        """
        enhanced_plan = correction_plan.copy()

        # --- Extraer valores de forma segura, según el tipo ---
        if isinstance(engine_config, dict):             # modo legacy
            needs_bin      = engine_config.get('needs_binarization', False)
            default_bin    = engine_config.get('default_binarization')
            denoise_conf   = engine_config.get('denoise_config')  or engine_config.get('denoise')
            contrast_conf  = engine_config.get('contrast_config') or engine_config.get('contrast')
        else:                                           # modo YAML → dataclass
            needs_bin      = getattr(engine_config, 'needs_binarization', False)
            default_bin    = getattr(engine_config, 'default_binarization', None)
            denoise_conf   = getattr(engine_config, 'denoise_config', None)
            contrast_conf  = getattr(engine_config, 'contrast_config', None)

        # --- Completar el plan solo si faltan llaves ---
        if 'denoise' not in enhanced_plan and denoise_conf:
            enhanced_plan['denoise'] = denoise_conf

        if 'contrast' not in enhanced_plan and contrast_conf:
            enhanced_plan['contrast'] = contrast_conf

        if 'binarization' not in enhanced_plan and needs_bin and default_bin:
            enhanced_plan['binarization'] = default_bin

        return enhanced_plan

    def get_supported_engines(self) -> list:
        """Devuelve motores soportados."""
        return list(self.engine_configs.keys())

    def get_engine_config(self, engine: str) -> Optional[Dict[str, Any]]:
        """Obtiene configuración específica de un motor."""
        return self.engine_configs.get(engine)