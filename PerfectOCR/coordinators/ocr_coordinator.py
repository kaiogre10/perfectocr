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

logger = logging.getLogger(__name__)

class OCREngineCoordinator:
    def __init__(self, config: Dict, project_root: str): 
        self.ocr_config_from_workflow = config
        self.project_root = project_root
        self.num_workers = 2  # Conservador pero estable

        tesseract_specific_config = self.ocr_config_from_workflow.get('tesseract')
        if tesseract_specific_config: 
            self.tesseract = TesseractOCR(full_ocr_config=self.ocr_config_from_workflow) 
            logger.debug(f"TesseractOCR inicializado para OCREngineCoordinator con {self.num_workers} workers.")
        else:
            self.tesseract = None
            logger.warning("Configuración para Tesseract no encontrada o vacía en la sección OCR.")

        paddle_specific_config = self.ocr_config_from_workflow.get('paddleocr')
        if paddle_specific_config:
            self.paddle = PaddleOCRWrapper(config_dict=paddle_specific_config, project_root=self.project_root)
            logger.debug(f"PaddleOCRWrapper inicializado para OCREngineCoordinator con {self.num_workers} workers.")
        else:
            self.paddle = None
            logger.warning("Configuración para PaddleOCR no encontrada o vacía en la sección OCR.")
        
        if not (self.tesseract or self.paddle):
            logger.error("Ningún motor OCR (Tesseract o PaddleOCR) pudo ser inicializado.")

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
            "error": "Tesseract not configured"
        }

    def _run_paddle_task(self, img, fname=None):
        # en vez de reutilizar self.paddle, se crea una nueva
        wrapper = PaddleOCRWrapper(self.ocr_config_from_workflow['paddleocr'],
                                   self.project_root)
        return wrapper.extract_detailed_line_data(img, fname)

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

    def run_ocr_parallel(
        self,
        preprocessed_images: Dict[str, np.ndarray],
        noise_regions: List[List[int]], 
        image_file_name: Optional[str] = None,
        folder_origin: Optional[str] = "unknown_folder", 
        image_pil_mode: Optional[str] = "unknown_mode" 
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ejecuta OCR en paralelo usando las imágenes preprocesadas específicas para cada motor.
        """
        start_time = time.perf_counter()
        
        folder_origin = self.ocr_config_from_workflow.get('default_folder_origin', "unknown_folder")
        image_pil_mode = self.ocr_config_from_workflow.get('default_image_pil_mode', "unknown_mode")
        
        # Obtener las imágenes específicas para cada motor
        tesseract_image = preprocessed_images.get('tesseract')
        paddle_image = preprocessed_images.get('paddleocr')
        
        if tesseract_image is None or paddle_image is None:
            logger.error("Faltan imágenes preprocesadas para uno o ambos motores OCR")
            return {
                "error": "Missing preprocessed images",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "image": image_file_name
                }
            }, time.perf_counter() - start_time

        page_dims = {"width": tesseract_image.shape[1], "height": tesseract_image.shape[0]}
        
        # Usar ThreadPoolExecutor con número óptimo de workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_tesseract = executor.submit(self._run_tesseract_task, tesseract_image, image_file_name)
            
            # PaddleOCR se ejecuta en el hilo PRINCIPAL (thread-safe)
            padd_result_payload = self._run_paddle_task(paddle_image, image_file_name)
            
            # Esperar resultado de Tesseract
            tess_result_payload = future_tesseract.result()

        # Consolidar resultados
        output_data = {
            "metadata": {
            "image": image_file_name,
            "folder_origin": folder_origin,
            "timestamp": datetime.now().isoformat(),
            "dimensions": page_dims,
            "image_mode": image_pil_mode,
            "processing_time_seconds": {
                "tesseract": tess_result_payload.get("processing_time_seconds", 0.0),
                "paddleocr": padd_result_payload.get("processing_time_seconds", 0.0)
            },
            "overall_confidence": {} 
        },
        "ocr_raw_results": {
            "tesseract": {"words": [], "text_layout": []},
            "paddleocr": {"lines": [], "full_text": ""}
        },
        "visual_output": {
            "tesseract_text": "",
            "paddleocr_text": ""
        }
    }

        # Poblar resultados de Tesseract
        if "error" not in tess_result_payload:
            raw_tess_words = tess_result_payload.get("recognized_text", {}).get("words", [])
            filtered_tess_words = self._filter_words_by_noise_regions(raw_tess_words, noise_regions)
            logger.info(f"Tesseract: {len(raw_tess_words) - len(filtered_tess_words)} palabras de ruido filtradas espacialmente.")
            output_data["ocr_raw_results"]["tesseract"]["words"] = filtered_tess_words
            output_data["metadata"]["overall_confidence"]["tesseract_words_avg"] = tess_result_payload.get("overall_confidence_words", 0.0)
            if output_data["ocr_raw_results"]["tesseract"]["text_layout"]:                output_data["visual_output"]["tesseract_text"] = "\n".join(
                    [line.get('line_text',' ') for line in output_data["ocr_raw_results"]["tesseract"]["text_layout"]]
                )
            elif output_data["ocr_raw_results"]["tesseract"]["words"]:
                output_data["visual_output"]["tesseract_text"] = " ".join(
                    [word.get('text','') for word in output_data["ocr_raw_results"]["tesseract"]["words"]]
                )
        else:
            output_data["ocr_raw_results"]["tesseract"]["error"] = tess_result_payload.get("error", "Error desconocido en Tesseract")

        # Poblar resultados de PaddleOCR
        if "error" not in padd_result_payload:
            output_data["ocr_raw_results"]["paddleocr"]["lines"] = padd_result_payload.get("recognized_text", {}).get("lines", [])
            output_data["ocr_raw_results"]["paddleocr"]["full_text"] = padd_result_payload.get("recognized_text", {}).get("full_text", "")
            output_data["ocr_raw_results"]["paddleocr"]["words"] = padd_result_payload.get("recognized_text", {}).get("words", [])
            output_data["metadata"]["overall_confidence"]["paddleocr_lines_avg"] = padd_result_payload.get("overall_confidence_avg_lines")
            output_data["visual_output"]["paddleocr_text"] = output_data["ocr_raw_results"]["paddleocr"]["full_text"]
        else:
            output_data["ocr_raw_results"]["paddleocr"]["error"] = padd_result_payload.get("error", "Error desconocido en PaddleOCR")
            
        # Actualizar dimensiones si no estaban (como fallback)
        if not output_data["metadata"]["dimensions"] and tesseract_image is not None:
            output_data["metadata"]["dimensions"] = {"width": tesseract_image.shape[1], "height": tesseract_image.shape[0]}
        
        return output_data, time.perf_counter() - start_time