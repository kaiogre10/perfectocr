# PerfectOCR/core/ocr/engine_manager.py
import threading
import logging
from typing import Optional, Dict, Any
from core.ocr.paddle_wrapper import PaddleOCRWrapper
from core.ocr.tesseract_wrapper import TesseractOCR

logger = logging.getLogger(__name__)

class OCREngineManager:
    """
    Gestor singleton thread-safe para instancias de motores OCR.
    Evita la recreación costosa de PaddleOCR y Tesseract en cada documento.
    """
    _lock = threading.Lock()
    _paddle_instance: Optional[PaddleOCRWrapper] = None
    _tesseract_instance: Optional[TesseractOCR] = None
    
    @classmethod
    def get_paddle_engine(cls, config_dict: Dict[str, Any], project_root: str) -> PaddleOCRWrapper:
        """
        Obtiene la instancia singleton de PaddleOCR.
        Thread-safe para uso con ThreadPoolExecutor.
        """
        if cls._paddle_instance is None:
            with cls._lock:
                if cls._paddle_instance is None:  # Double-check locking
                    logger.info("Inicializando PaddleOCR singleton...")
                    cls._paddle_instance = PaddleOCRWrapper(config_dict, project_root)
                    logger.info("PaddleOCR singleton inicializado exitosamente")
        return cls._paddle_instance
    
    @classmethod
    def get_tesseract_engine(cls, full_ocr_config: Dict[str, Any]) -> TesseractOCR:
        """
        Obtiene la instancia singleton de Tesseract.
        Thread-safe para uso con ThreadPoolExecutor.
        """
        if cls._tesseract_instance is None:
            with cls._lock:
                if cls._tesseract_instance is None:  # Double-check locking
                    logger.info("Inicializando Tesseract singleton...")
                    cls._tesseract_instance = TesseractOCR(full_ocr_config)
                    logger.info("Tesseract singleton inicializado exitosamente")
        return cls._tesseract_instance
    
    @classmethod
    def clear_engines(cls):
        """
        Limpia las instancias singleton (útil para testing o reinicio).
        """
        with cls._lock:
            cls._paddle_instance = None
            cls._tesseract_instance = None
            logger.info("Instancias singleton de OCR limpiadas")