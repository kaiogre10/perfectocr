# PerfectOCR/core/preprocessing/image_corrector.py
import numpy as np
import logging
from core.preprocessing import toolbox
from typing import Dict, Any, Optional
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class ImageCorrector:
    def __init__(self, config_loader=None):
        """
        Inicialización compatible con modo YAML y legacy.
        
        Args:
            config_loader: ConfigLoader para modo YAML, None para modo legacy
        """
        self.config_loader = config_loader
        
        if config_loader:
            # Modo YAML: cargar desde configuración
            self.default_corrections = self._load_default_corrections_yaml()
            self._yaml_mode = True
        else:
            # Modo legacy: valores hardcodeados
            self.default_corrections = self._load_default_corrections_legacy()
            self._yaml_mode = False
        
    def _load_default_corrections_yaml(self) -> Dict[str, Any]:
        """Carga valores por defecto desde configuración YAML."""
        return {
            'deskew': {'angle': 0.0},
            'denoise': {'strength': 10},
            'contrast': {'clahe_params': {'clip_limit': 2.0, 'grid_size': (8, 8)}}
        }
    
    def _load_default_corrections_legacy(self) -> Dict[str, Any]:
        """Valores por defecto para modo legacy."""
        return {
            'deskew': {'angle': 0.0},
            'denoise': {'strength': 10},
            'contrast': {'clahe_params': {'clip_limit': 2.0, 'grid_size': (8, 8)}}
        }
    
    def apply_grayscale_corrections(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """
        Aplica correcciones en escala de grises.
        Compatible con ambos modos.
        """
        if not isinstance(correction_plan, dict):
            logger.warning(f"Plan de corrección inválido: {type(correction_plan)}")
            return gray_image

        corrected_image = gray_image
        
        # Pipeline de correcciones
        if correction_plan.get('deskew', {}).get('angle', 0.0) != 0.0:
            corrected_image = self._apply_deskew(corrected_image, correction_plan)

        if correction_plan.get('denoise', {}).get('strength', 0) > 0:
            corrected_image = self._apply_denoise(corrected_image, correction_plan)

        corrected_image = self._apply_contrast(corrected_image, correction_plan)

        return corrected_image
    
    def _apply_deskew(self, image: np.ndarray, plan: Dict[str, Any]) -> np.ndarray:
        """Aplica corrección de inclinación."""
        deskew_config = plan.get('deskew', self.default_corrections['deskew'])
        angle = deskew_config.get('angle', 0.0)
        
        return toolbox.apply_deskew(image, angle) if angle != 0.0 else image
    
    def _apply_denoise(self, image: np.ndarray, plan: Dict[str, Any]) -> np.ndarray:
        """Aplica eliminación de ruido."""
        denoise_config = plan.get('denoise')
        if denoise_config and 'strength' in denoise_config:
            return toolbox.apply_denoise(image, denoise_config['strength'])
        return image
    
    def _apply_contrast(self, image: np.ndarray, plan: Dict[str, Any]) -> np.ndarray:
        """Aplica mejora de contraste."""
        contrast_config = plan.get('contrast')
        if contrast_config and 'clahe_params' in contrast_config:
            return toolbox.apply_clahe_contrast(image, **contrast_config['clahe_params'])
        return image