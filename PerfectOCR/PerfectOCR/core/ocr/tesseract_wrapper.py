import re
import pytesseract
import numpy as np
import os
import time
import logging
import cv2
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TesseractOCR:
    def __init__(self, full_ocr_config: Dict):
        if 'tesseract' not in full_ocr_config or not isinstance(full_ocr_config['tesseract'], dict):
            msg = "La sección 'tesseract' es inválida o no se encuentra en la configuración OCR."
            logger.error(msg)
            raise ValueError(msg)
        
        self.config = full_ocr_config['tesseract']
        self.lang = self.config.get('lang', 'spa+eng')
        self.default_psm = self.config.get('psm', {})
        self.default_oem = self.config.get('oem', {})
        self.char_whitelist = self.config.get('tessedit_char_whitelist', {})
        
        tesseract_cmd_path_from_config = self.config.get('cmd_path')
        if tesseract_cmd_path_from_config:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path_from_config
        elif not pytesseract.pytesseract.tesseract_cmd:
            logger.info("cmd_path de Tesseract no especificado... pytesseract intentará encontrarlo.")
        # logger.debug(f"Tesseract cmd_path: {pytesseract.pytesseract.tesseract_cmd}")

    def _limpiar(self, texto: str) -> str:
        if not isinstance(texto, str): return " "
        texto = texto.replace('\r', ' ').replace('\n', ' ')
        texto = re.sub(r'\|--', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def _build_config_str(self) -> str:
        # Simplificado para tomar siempre desde la configuración de la instancia
        config_parts = [
            f"--psm {self.default_psm}",
            f"--oem {self.default_oem}",
            f"-l {self.lang}"
        ]
        
        if self.config.get('dpi'): 
            config_parts.append(f"--dpi {self.config['dpi']}")
        
        if self.char_whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={self.char_whitelist}")
        
        if self.config.get("preserve_interword_spaces") is not None:
            config_parts.append(f"-c preserve_interword_spaces={self.config.get('preserve_interword_spaces')}")

        for key, value in self.config.items():
            if key.startswith("tessedit_") and key not in ["tessedit_char_whitelist"]:
                config_parts.append(f"-c {key}={value}")
        
        return " ".join(config_parts)

    def _rescale_ocr_data(self, ocr_data: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
        """Re-escala las coordenadas geométricas en los datos de Tesseract."""
        if scale_factor == 1.0:
            return ocr_data
        
        num_items = len(ocr_data['text'])
        for i in range(num_items):
            for key in ['left', 'top', 'width', 'height']:
                ocr_data[key][i] = int(round(int(ocr_data[key][i]) / scale_factor))
        
        return ocr_data

    # MODIFICAMOS EL MÉTODO PRINCIPAL DE EXTRACCIÓN
    def extract_detailed_word_data(self, image: np.ndarray, image_file_name: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if image is None or not isinstance(image, np.ndarray):
            logger.error(f"Imagen inválida para Tesseract OCR: {type(image)}")
            return {"error": "Invalid image input"}
        
        # El preprocesamiento ahora garantiza una imagen adecuada (binaria, 8-bit)
        if len(image.shape) == 3:
             logger.warning(f"Tesseract recibió una imagen con canales de color, convirtiendo a escala de grises. "
                           f"Se esperaba una imagen preprocesada binaria. Shape: {image.shape}")
             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        config_str = self._build_config_str()
        confidence_threshold = self.config.get('confidence_threshold', 30.0)
        output_words: List[Dict[str, Any]] = []
        text_layout_lines: List[Dict[str, Any]] = []
        img_h, img_w = image.shape[:2]
        total_confidence = 0.0
        word_count_for_avg = 0
        
        lines_data_temp: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {} 

        # --- EL ESCALADO DE IMAGEN (cv2.resize) SE ELIMINA DE AQUÍ ---
        # Esta es la modificación clave que garantiza dimensiones consistentes.
        # El preprocesamiento ahora se encarga de entregar una imagen optimizada.
        
        try:
            data = pytesseract.image_to_data(image, config=config_str, output_type=pytesseract.Output.DICT)
            num_items = len(data['text'])
            word_internal_counter = 0

            for i in range(num_items):
                confidence_val_str = data['conf'][i]
                text_val = data['text'][i]

                if text_val and text_val.strip():
                    try:
                        confidence_val = float(confidence_val_str)
                    except ValueError:
                        continue

                    if confidence_val < confidence_threshold:
                        continue

                    txt = self._limpiar(text_val)
                    if not txt: continue

                    word_internal_counter += 1
                    x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])

                    if w <= 0 or h <= 0: continue

                    output_words.append({
                        "word_number": word_internal_counter, "text": txt,
                        "bbox": [x, y, x + w, y + h], "confidence": round(confidence_val, 2)
                    })

                    if confidence_val >= 0:
                        total_confidence += confidence_val
                        word_count_for_avg += 1

                    line_key = (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i]))
                    lines_data_temp.setdefault(line_key, []).append({"text": txt, "confidence": confidence_val, "x_coord": x})

            inferred_line_num = 0
            for key in sorted(lines_data_temp.keys()):
                words_in_line = sorted(lines_data_temp[key], key=lambda item: item['x_coord']) 
                line_text = " ".join([word['text'] for word in words_in_line]).strip()
                
                if line_text: 
                    inferred_line_num += 1
                    line_confidences = [w['confidence'] for w in words_in_line if w['confidence'] >= 0]
                    avg_conf = round(sum(line_confidences) / len(line_confidences), 2) if line_confidences else 0.0
                    
                    text_layout_lines.append({
                        "line_number_inferred": inferred_line_num,
                        "line_text": line_text,
                        "line_avg_word_confidence": avg_conf
                    })

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract ejecutable no encontrado o no configurado correctamente.")
            return {"error": "TesseractNotFoundError"}
        except Exception as e:
            logger.error(f"Error en Tesseract para '{image_file_name}': {e}", exc_info=True)
            return {"error": str(e)}

        processing_time = round(time.perf_counter() - start_time, 3)
        overall_avg_confidence = round(total_confidence / word_count_for_avg, 2) if word_count_for_avg > 0 else 0.0
        
        logger.info(f"Tesseract para '{image_file_name}' OK. Palabras: {len(output_words)}, Conf. Prom: {overall_avg_confidence:.2f}%, Tiempo: {processing_time}s")
        return {
            "ocr_engine": "tesseract",
            "processing_time_seconds": processing_time,
            "overall_confidence_words": overall_avg_confidence,
            "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
            "recognized_text": {"text_layout": text_layout_lines, "words": output_words}
        }
