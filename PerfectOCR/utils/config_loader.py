# PerfectOCR/utils/config_loader.py
import yaml
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingEngineConfig:
    """Configuración específica por motor de OCR desde YAML."""
    needs_binarization: bool = False
    invert_binary: bool = False
    default_binarization: Dict[str, int] = None
    denoise_config: Dict[str, Any] = None
    contrast_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_binarization is None:
            self.default_binarization = {'block_size': 31, 'c_value': 7}
        if self.denoise_config is None:
            self.denoise_config = {}
        if self.contrast_config is None:
            self.contrast_config = {}

class ConfigLoader:
    """Cargador centralizado de configuración que conecta YAML con el código."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml_config()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Carga el archivo YAML principal."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Configuración cargada desde {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error cargando configuración desde {self.config_path}: {e}")
            raise
    
    def get_preprocessing_config(self) -> Dict[str, PreprocessingEngineConfig]:
        """
        Extrae configuración de preprocesamiento desde image_preparation.quality_assessment_rules
        y la convierte en objetos PreprocessingEngineConfig.
        """
        image_prep = self.config.get('image_preparation', {})
        quality_rules = image_prep.get('quality_assessment_rules', {})
        
        engine_configs = {}
        
        # Configuración para Tesseract
        tesseract_rules = quality_rules.get('tesseract', {})
        engine_configs['tesseract'] = PreprocessingEngineConfig(
            needs_binarization=True,
            invert_binary=False,  # Texto negro sobre blanco
            default_binarization={
                'block_size': tesseract_rules.get('binarization', {}).get('block_sizes_map', [31])[0],
                'c_value': tesseract_rules.get('binarization', {}).get('adaptive_c_value', 7)
            },
            denoise_config=tesseract_rules.get('denoise', {}),
            contrast_config=tesseract_rules.get('contrast_enhancement', {})
        )
        
        # Configuración para PaddleOCR
        paddle_rules = quality_rules.get('paddleocr', {})
        engine_configs['paddleocr'] = PreprocessingEngineConfig(
            needs_binarization=False,  # Solo escala de grises
            denoise_config=paddle_rules.get('denoise', {}),
            contrast_config=paddle_rules.get('contrast_enhancement', {})
        )
        
        # Configuración para Spatial Analysis
        spatial_rules = quality_rules.get('spatial_analysis', {})
        engine_configs['spatial_analysis'] = PreprocessingEngineConfig(
            needs_binarization=True,
            invert_binary=True,  # Texto blanco sobre negro
            default_binarization={
                'block_size': spatial_rules.get('binarization', {}).get('block_sizes_map', [15])[0],
                'c_value': spatial_rules.get('binarization', {}).get('adaptive_c_value', 5)
            },
            denoise_config=spatial_rules.get('denoise', {}),
            contrast_config=spatial_rules.get('contrast_enhancement', {})
        )
        
        return engine_configs
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Obtiene configuración del flujo de trabajo."""
        return self.config.get('workflow', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Obtiene configuración de motores OCR."""
        return self.config.get('ocr', {})
    
    def get_spatial_analyzer_config(self) -> Dict[str, Any]:
        """Obtiene configuración del analizador espacial."""
        return self.config.get('spatial_analyzer', {})
    
    def get_table_extractor_config(self) -> Dict[str, Any]:
        """Obtiene configuración del extractor de tablas."""
        return self.config.get('table_extractor', {})
    
    def get_postprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de postprocesamiento."""
        return self.config.get('postprocessing', {})
    
    def get_max_workers_for_cpu(self) -> int:
        """
        Calcula workers óptimos basado en configuración o hardware.
        Para tu i5-8400H: 8 hilos - 2 reservados = 6 máximo.
        """
        workflow_config = self.get_workflow_config()
        configured_workers = workflow_config.get('max_workers')
        
        if configured_workers:
            return int(configured_workers)
        
        # Auto-detección para tu CPU específico
        import os
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(cpu_count - 2, 6)  # Nunca más de 6 para tu i5-8400H
        
        logger.info(f"Auto-detectados {optimal_workers} workers para CPU con {cpu_count} hilos")
        return optimal_workers