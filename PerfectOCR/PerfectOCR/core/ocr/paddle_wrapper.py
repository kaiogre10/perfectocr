# PerfectOCR/core/ocr/paddle_wrapper.py
import os
import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class PaddleOCRWrapper:
    def __init__(self, config_dict: Dict, project_root: str):
        self.paddle_config = config_dict
        self.project_root = project_root
        logger.debug(f"PaddleOCRWrapper inicializando con config: {self.paddle_config}")

        def resolve_model_path(model_key: str) -> Optional[str]:
            path = self.paddle_config.get(model_key)
            if not path: return None
            if os.path.isabs(path): return path
            return os.path.abspath(os.path.join(self.project_root, path))

        try:
            self.engine = PaddleOCR(
                use_angle_cls=self.paddle_config.get('use_angle_cls', True),
                lang=self.paddle_config.get('lang', 'es'),
                det_model_dir=resolve_model_path('det_model_dir'),
                rec_model_dir=resolve_model_path('rec_model_dir'),
                cls_model_dir=resolve_model_path('cls_model_dir'),
                use_gpu=self.paddle_config.get('use_gpu', False),
                show_log=self.paddle_config.get('show_log', False)
            )
            logger.info(f"PaddleOCR engine inicializado exitosamente desde rutas locales.")
        except Exception as e:
            logger.error(f"Error crítico al inicializar PaddleOCR engine: {e}", exc_info=True)
            self.engine = None

    def _parse_paddle_result_to_spec(self, paddle_ocr_result_raw: Optional[List[Any]]) -> List[Dict[str, Any]]:
        output_lines: List[Dict[str, Any]] = []
        if not paddle_ocr_result_raw or not paddle_ocr_result_raw[0]:
            return output_lines
        items_for_first_image = paddle_ocr_result_raw[0]
        if items_for_first_image is None: return output_lines

        for line_counter, item_tuple in enumerate(items_for_first_image):
            if not (isinstance(item_tuple, (list, tuple)) and len(item_tuple) == 2): continue
            bbox_polygon_raw, text_and_confidence = item_tuple[0], item_tuple[1]
            if not (isinstance(text_and_confidence, (list, tuple)) and len(text_and_confidence) == 2): continue
            text, confidence_raw = text_and_confidence
            try:
                polygon_coords_formatted = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                if len(polygon_coords_formatted) < 3: continue
            except (TypeError, ValueError, IndexError): continue
            output_lines.append({
                "line_number": line_counter + 1, "text": str(text).strip(),
                "polygon_coords": polygon_coords_formatted,
                "confidence": round(float(confidence_raw) * 100.0, 2) if isinstance(confidence_raw, (float, int)) else 0.0
            })
        return output_lines

    def _segment_line_into_words(self, line_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        text, polygon_coords = line_data.get('text', ''), line_data.get('polygon_coords', [])
        if not text or not polygon_coords: return []
        words = [word.strip() for word in text.split() if word.strip()]
        if not words: return []
        x_coords = [p[0] for p in polygon_coords]; y_coords = [p[1] for p in polygon_coords]
        min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
        line_width, total_chars = max_x - min_x, sum(len(w) for w in words)
        word_results, current_x = [], min_x
        for i, word in enumerate(words):
            char_ratio = len(word) / total_chars if total_chars > 0 else 1.0 / len(words)
            word_width = line_width * char_ratio
            word_results.append({
                "word_number": i + 1, "text": word,
                "polygon_coords": [[current_x, min_y], [current_x + word_width, min_y], [current_x + word_width, max_y], [current_x, max_y]],
                "confidence": line_data.get('confidence', 0.0), "source_line_number": line_data.get('line_number', 0)
            })
            current_x += word_width
        return word_results

    def extract_detailed_line_data(self, image: np.ndarray, image_file_name: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if image is None: return {"error": "Invalid image input", "ocr_engine": "paddleocr"}
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.engine is None:
            logger.error("Motor PaddleOCR no inicializado. No se puede extraer texto.")
            return {"error": "PaddleOCR engine not initialized", "ocr_engine": "paddleocr"}

        img_h, img_w = image.shape[:2]
        try:
            raw_paddle_result = self.engine.ocr(image, cls=self.paddle_config.get('use_angle_cls', True))
            parsed_lines = self._parse_paddle_result_to_spec(raw_paddle_result)
            full_text = "\n".join([line['text'] for line in parsed_lines])
            avg_conf = np.mean([line['confidence'] for line in parsed_lines]) if parsed_lines else 0.0
            all_words = [word for line in parsed_lines for word in self._segment_line_into_words(line)]
        except RuntimeError as e:
            if "could not execute a primitive" in str(e):
                logger.error(
                    f"PaddleOCR falló con un error de bajo nivel para '{image_file_name}'. "
                    f"Esto usualmente indica un problema con los modelos o el entorno, no con el código. "
                    f"Posibles causas: 1) Modelos corruptos en 'models/paddle/', 2) Incompatibilidad de librerías, "
                    f"3) Un problema con la imagen de entrada. Intente re-descargar los modelos o reinstalar 'paddlepaddle-cpu'.",
                    exc_info=True
                )
                return {"error": "PaddleOCR low-level runtime error", "ocr_engine": "paddleocr"}
            else:
                logger.error(f"Error inesperado de Runtime en PaddleOCR para {image_file_name}: {e}", exc_info=True)
                return {"error": str(e), "ocr_engine": "paddleocr"}
        except Exception as e:
            logger.error(f"Error en PaddleOCR extract_detailed_line_data para {image_file_name}: {e}", exc_info=True)
            return {"error": str(e), "ocr_engine": "paddleocr"}
            
        return {
            "ocr_engine": "paddleocr", "processing_time_seconds": round(time.perf_counter() - start_time, 3),
            "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
            "recognized_text": {"full_text": full_text, "lines": parsed_lines, "words": all_words},
            "overall_confidence_avg_lines": round(float(avg_conf), 2)
        }