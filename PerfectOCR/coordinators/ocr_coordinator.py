# PerfectOCR/coordinators/ocr_coordinator.py
import os
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from utils.geometric import calculate_iou
from core.ocr.tesseract_wrapper import TesseractOCR
from core.ocr.paddle_wrapper import PaddleOCRWrapper
import time
from utils.output_handlers import JsonOutputHandler
from core.ocr.engine_manager import OCREngineManager

logger = logging.getLogger(__name__)

class OCREngineCoordinator:
    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool]):
        self.ocr_config_from_workflow = config
        self.project_root = project_root
        self.output_flags = output_flags
        self.json_output_handler = JsonOutputHandler()
        
        # OPTIMIZACIÓN: Workers adaptativos
        enabled_engines = config.get('enabled_engines', {'tesseract': True, 'paddleocr': True})
        enabled_count = sum(enabled_engines.values())
        
        if enabled_count == 2:
            self.num_workers = 4  # Ambos motores
        elif enabled_count == 1:
            self.num_workers = 6  # Solo un motor
        else:
            self.num_workers = 2  # Fallback
            
        logger.info(f"OCR workers: {self.num_workers} para {enabled_count} motores")

        # USAR SINGLETONS
        if enabled_engines.get('tesseract', True):
            tesseract_specific_config = self.ocr_config_from_workflow.get('tesseract')
            if tesseract_specific_config: 
                singleton_start = time.perf_counter()
                self.tesseract = OCREngineManager.get_tesseract_engine(self.ocr_config_from_workflow)
                singleton_time = time.perf_counter() - singleton_start
                logger.info(f"Tesseract singleton: {singleton_time:.3f}s")
            else:
                self.tesseract = None
        else:
            self.tesseract = None

        if enabled_engines.get('paddleocr', True):
            paddle_specific_config = self.ocr_config_from_workflow.get('paddleocr')
            if paddle_specific_config:
                singleton_start = time.perf_counter()
                self.paddle = OCREngineManager.get_paddle_engine(paddle_specific_config, self.project_root)
                singleton_time = time.perf_counter() - singleton_start
                logger.info(f"PaddleOCR singleton: {singleton_time:.3f}s")
            else:
                self.paddle = None
        else:
            self.paddle = None
        
        if not (self.tesseract or self.paddle):
            raise ValueError("No OCR engines enabled")

    def _run_tesseract_task(self, binary_image_for_ocr: np.ndarray, image_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Tarea para ejecutar Tesseract OCR y manejar su resultado."""
        if self.tesseract:
            try:
                if binary_image_for_ocr is None:
                    raise ValueError("Imagen de entrada para Tesseract es None")
                
                # Verificar formato de imagen
                if not isinstance(binary_image_for_ocr, np.ndarray):
                    raise ValueError(f"Formato de imagen inválido para Tesseract: {type(binary_image_for_ocr)}")
                
                tess_result_dict = self.tesseract.extract_detailed_word_data(binary_image_for_ocr, image_file_name)
                return tess_result_dict
            except Exception as e:
                logger.error(f"Excepción crítica en Tesseract OCR para {image_file_name}: {e}", exc_info=True)
                return {
                    "ocr_engine": "tesseract",
                    "processing_time_seconds": 0.0,
                    "recognized_text": {"text_layout": [], "words": []},
                    "error": str(e)
                }
        return {
            "ocr_engine": "tesseract",
            "processing_time_seconds": 0.0,
            "recognized_text": {"text_layout": [], "words": []},
            "error": "Tesseract not enabled"
        }

    def _run_paddle_task(self, img, fname=None):
        if self.paddle:
            return self.paddle.extract_detailed_line_data(img, fname)
        else:
            return {
                "ocr_engine": "paddleocr",
                "processing_time_seconds": 0.0,
                "recognized_text": {"lines": [], "words": []},
                "error": "PaddleOCR not enabled"
            }

    def _filter_words_by_noise_regions(self, words: List[Dict], noise_regions: List[List[int]]) -> List[Dict]:
        """
        Versión mejorada del filtrado de palabras con mejor detección de solapamiento.
        """
        if not noise_regions:
            return words

        filtered_words = []
        filtered_count = 0
        
        logger.debug(f"Iniciando filtrado de {len(words)} palabras contra {len(noise_regions)} regiones de ruido.")
        
        for word in words:
            word_bbox = word.get('bbox')
            if not word_bbox or len(word_bbox) != 4:
                filtered_words.append(word)  # Mantener si no tiene bbox válido
                continue

            # Verificar coordenadas válidas
            try:
                word_xmin, word_ymin, word_xmax, word_ymax = [float(coord) for coord in word_bbox]
            except (ValueError, TypeError):
                filtered_words.append(word)  # Mantener si bbox no es válido
                continue
            
            if word_ymax <= word_ymin or word_xmax <= word_xmin:
                logger.debug(f"Palabra '{word.get('text', 'N/A')}' descartada por bbox degenerado: {word_bbox}")
                continue  # Descartar bbox degenerado

            word_center_y = (word_ymin + word_ymax) / 2
            word_height = word_ymax - word_ymin
            
            is_noise = False
            
            for noise_box in noise_regions:
                if len(noise_box) != 4:
                    continue
                
                try:
                    noise_xmin, noise_ymin, noise_xmax, noise_ymax = [float(coord) for coord in noise_box]
                except (ValueError, TypeError):
                    continue
                
                # Método 1: Verificar si el centro vertical de la palabra está dentro de la región de ruido
                if noise_ymin <= word_center_y <= noise_ymax:
                    is_noise = True
                    logger.debug(f"Palabra '{word.get('text', 'N/A')}' filtrada: centro Y={word_center_y:.1f} "
                               f"está en región de ruido Y:[{noise_ymin:.1f}, {noise_ymax:.1f}]")
                    break
                
                # Método 2: Verificar solapamiento vertical significativo
                vertical_overlap = max(0, min(word_ymax, noise_ymax) - max(word_ymin, noise_ymin))
                
                if word_height > 0:
                    overlap_ratio = vertical_overlap / word_height
                    if overlap_ratio > 0.3:  # 30% de solapamiento vertical
                        is_noise = True
                        logger.debug(f"Palabra '{word.get('text', 'N/A')}' filtrada: solapamiento "
                                   f"{overlap_ratio:.1%} con región Y:[{noise_ymin:.1f}, {noise_ymax:.1f}]")
                        break
                
                # Método 3: Verificar proximidad para palabras muy pequeñas
                if word_height <= 10:  # Palabras muy pequeñas
                    distance_to_noise = min(abs(word_center_y - noise_ymin), abs(word_center_y - noise_ymax))
                    if distance_to_noise <= 3:  # Muy cerca de la línea de ruido
                        is_noise = True
                        logger.debug(f"Palabra pequeña '{word.get('text', 'N/A')}' filtrada por proximidad "
                                   f"({distance_to_noise:.1f}px) a región de ruido")
                        break
            
            if not is_noise:
                filtered_words.append(word)
            else:
                filtered_count += 1
        
        logger.info(f"Filtrado mejorado: {filtered_count} palabras eliminadas de {len(words)} total. "
                   f"Quedan {len(filtered_words)} palabras.")
        return filtered_words

    def run_ocr_parallel(self, preprocessed_images: Dict[str, np.ndarray], noise_regions: List[List[int]], 
                        image_file_name: Optional[str] = None, folder_origin: Optional[str] = "unknown_folder", 
                        image_pil_mode: Optional[str] = "unknown_mode") -> Tuple[Dict[str, Any], float]:
        
        start_time = time.perf_counter()
        
        folder_origin = self.ocr_config_from_workflow.get('default_folder_origin', "unknown_folder")
        image_pil_mode = self.ocr_config_from_workflow.get('default_image_pil_mode', "unknown_mode")
        
        tesseract_image = preprocessed_images.get('tesseract') if self.tesseract else None
        paddle_image = preprocessed_images.get('paddleocr') if self.paddle else None
        
        if self.tesseract and tesseract_image is None:
            return {"error": "Missing Tesseract preprocessed image"}, time.perf_counter() - start_time
            
        if self.paddle and paddle_image is None:
            return {"error": "Missing PaddleOCR preprocessed image"}, time.perf_counter() - start_time

        if tesseract_image is not None:
            page_dims = {"width": tesseract_image.shape[1], "height": tesseract_image.shape[0]}
        elif paddle_image is not None:
            page_dims = {"width": paddle_image.shape[1], "height": paddle_image.shape[0]}
        else:
            page_dims = {"width": 0, "height": 0}

        # Ejecutar motores
        execution_start = time.perf_counter()
        tess_result_payload = None
        padd_result_payload = None
        
        if self.tesseract and self.paddle:
            # Paralelo
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_tesseract = executor.submit(self._run_tesseract_task, tesseract_image, image_file_name)
                padd_result_payload = self._run_paddle_task(paddle_image, image_file_name)
                tess_result_payload = future_tesseract.result()
        elif self.tesseract:
            tess_result_payload = self._run_tesseract_task(tesseract_image, image_file_name)
        elif self.paddle:
            padd_result_payload = self._run_paddle_task(paddle_image, image_file_name)

        execution_time = time.perf_counter() - execution_start

        # Consolidar resultados
        output_data = self._consolidate_ocr_results(
            tess_result_payload, padd_result_payload, 
            image_file_name, folder_origin, image_pil_mode, page_dims, noise_regions
        )

        total_time = time.perf_counter() - start_time
        
        return output_data, total_time

    def _consolidate_ocr_results(self, tess_result_payload: Optional[Dict], padd_result_payload: Optional[Dict],
                               image_file_name: str, folder_origin: str, image_pil_mode: str, 
                               page_dims: Dict, noise_regions: List) -> Dict[str, Any]:
        
        output_data = {
            "metadata": {
                "image": image_file_name,
                "folder_origin": folder_origin,
                "image_pil_mode": image_pil_mode,
                "timestamp": datetime.now().isoformat(),
                "page_dimensions": page_dims,
                "enabled_engines": {
                    "tesseract": self.tesseract is not None,
                    "paddleocr": self.paddle is not None
                },
                "processing_time_seconds": {
                    "tesseract": tess_result_payload.get("processing_time_seconds", 0.0) if tess_result_payload else 0.0,
                    "paddleocr": padd_result_payload.get("processing_time_seconds", 0.0) if padd_result_payload else 0.0
                },
                "overall_confidence": {}
            },
            "ocr_raw_results": {},
            "visual_output": {"tesseract_text": "", "paddleocr_text": ""}
        }
        
        # Tesseract
        if tess_result_payload and "error" not in tess_result_payload:
            raw_tess_words = tess_result_payload.get("recognized_text", {}).get("words", [])
            filtered_tess_words = self._filter_words_by_noise_regions(raw_tess_words, noise_regions)
            
            output_data["ocr_raw_results"]["tesseract"] = {
                "words": filtered_tess_words,
                "text_layout": tess_result_payload.get("recognized_text", {}).get("text_layout", [])
            }
            output_data["metadata"]["overall_confidence"]["tesseract_words_avg"] = tess_result_payload.get("overall_confidence_words", 0.0)
            
            if output_data["ocr_raw_results"]["tesseract"]["text_layout"]:
                output_data["visual_output"]["tesseract_text"] = "\n".join(
                    [line.get('line_text', ' ') for line in output_data["ocr_raw_results"]["tesseract"]["text_layout"]]
                )
            elif output_data["ocr_raw_results"]["tesseract"]["words"]:
                output_data["visual_output"]["tesseract_text"] = " ".join(
                    [word.get('text', '') for word in output_data["ocr_raw_results"]["tesseract"]["words"]]
                )
        elif tess_result_payload and "error" in tess_result_payload:
            output_data["ocr_raw_results"]["tesseract"] = {"error": tess_result_payload.get("error")}

        # PaddleOCR
        if padd_result_payload and "error" not in padd_result_payload:
            output_data["ocr_raw_results"]["paddleocr"] = {
                "lines": padd_result_payload.get("recognized_text", {}).get("lines", []),
                "full_text": padd_result_payload.get("recognized_text", {}).get("full_text", ""),
                "words": padd_result_payload.get("recognized_text", {}).get("words", [])
            }
            output_data["metadata"]["overall_confidence"]["paddleocr_lines_avg"] = padd_result_payload.get("overall_confidence_avg_lines")
            output_data["visual_output"]["paddleocr_text"] = output_data["ocr_raw_results"]["paddleocr"]["full_text"]
        elif padd_result_payload and "error" in padd_result_payload:
            output_data["ocr_raw_results"]["paddleocr"] = {"error": padd_result_payload.get("error")}
        
        # Guardar JSON
        ocr_json_path = None
        if self.output_flags.get('ocr_raw', False) and image_file_name:
            output_dir = self.ocr_config_from_workflow.get('output_dir', './output')
            base_name = os.path.splitext(os.path.basename(image_file_name))[0]
            ocr_json_path = self.json_output_handler.save(
                data=output_data,
                output_dir=output_dir,
                file_name_with_extension=f"{base_name}_ocr_raw_results.json"
            )
            
        output_data["ocr_raw_json_path"] = ocr_json_path
        return output_data

    def validate_ocr_results(self, ocr_results: Optional[dict], filename: str) -> bool:
        """MOVIDO DESDE MAIN.PY - Valida resultados OCR"""
        if not isinstance(ocr_results, dict): 
            return False
        
        ocr_raw_results = ocr_results.get("ocr_raw_results", {})
        enabled_engines = ocr_results.get("metadata", {}).get("enabled_engines", {})
        
        has_valid_text = False
        
        if enabled_engines.get("tesseract", False):
            tesseract_data = ocr_raw_results.get("tesseract", {})
            if "error" not in tesseract_data and len(tesseract_data.get("words", [])) > 0:
                has_valid_text = True
        
        if enabled_engines.get("paddleocr", False):
            paddle_data = ocr_raw_results.get("paddleocr", {})
            if "error" not in paddle_data and len(paddle_data.get("lines", [])) > 0:
                has_valid_text = True
        
        return has_valid_text