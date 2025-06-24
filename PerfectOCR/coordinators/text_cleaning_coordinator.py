# PerfectOCR/coordinators/text_cleaning_coordinator.py
import logging
from typing import Dict, List, Any, Optional, Tuple
from core.text_cleaning.text_cleaner import TextCleaner
import os
from utils.output_handlers import JsonOutputHandler

logger = logging.getLogger(__name__)

class TextCleaningCoordinator:
    """
    Coordinador para limpieza de texto OCR.
    Maneja la limpieza automática y la interfaz batch para correcciones manuales.
    """
    
    def __init__(self, config: Dict, output_flags: Dict[str, bool]):
        self.config = config
        self.output_flags = output_flags
        self.text_cleaner = None
        self.json_handler = JsonOutputHandler()  # Sin config
        
        # Inicializar TextCleaner si está habilitado
        if self.config.get('enabled', True):
            self.text_cleaner = TextCleaner(config=self.config)
            logger.info("TextCleaningCoordinator inicializado con TextCleaner")
        else:
            logger.info("TextCleaningCoordinator inicializado sin TextCleaner (deshabilitado)")
    
    def clean_reconstructed_lines(self, reconstructed_lines_by_engine: Dict[str, List[Dict]], output_dir: str, doc_id: str) -> Tuple[Dict[str, List[Dict]], int]:
        """
        Limpia todas las líneas reconstruidas y sus elementos constituyentes.
        """
        if not self.text_cleaner:
            logger.warning("TextCleaner no disponible. Devolviendo líneas originales.")
            return reconstructed_lines_by_engine, 0
        
        logger.info("Iniciando limpieza de líneas reconstruidas...")
        
        page_dimensions = reconstructed_lines_by_engine.get('page_dimensions', {})
        
        cleaned_lines_by_engine = {'page_dimensions': page_dimensions} if page_dimensions else {}
        
        total_lines_processed = 0
        total_lines_cleaned = 0
        
        for engine_name, lines in reconstructed_lines_by_engine.items():
            if engine_name == 'page_dimensions' or not isinstance(lines, list):
                continue
                
            cleaned_lines = []
            
            for line in lines:
                total_lines_processed += 1
                cleaned_line = line.copy()
                original_line_text = line.get('text_raw', '')
                
                # 1. Limpiar cada palabra/segmento individualmente
                if 'constituent_elements_ocr_data' in cleaned_line:
                    for word_element in cleaned_line['constituent_elements_ocr_data']:
                        word_text_raw = word_element.get('text_raw', '')
                        word_confidence = word_element.get('confidence', 100.0)
                        
                        # Aplicar limpieza a la palabra
                        cleaned_word_text = self.text_cleaner.clean_text(word_text_raw, word_confidence)
                        if cleaned_word_text != word_text_raw:
                            word_element['text_original'] = word_text_raw
                        word_element['text_raw'] = cleaned_word_text

                # 2. Reconstruir el texto de la línea a partir de los constituyentes ya limpios
                if 'constituent_elements_ocr_data' in cleaned_line:
                    cleaned_line_text = " ".join(
                        w.get('text_raw', '') for w in cleaned_line.get('constituent_elements_ocr_data', [])
                    ).strip()
                    cleaned_line['text_raw'] = cleaned_line_text
                
                # Marcar la línea como corregida si el texto final es diferente al original
                if cleaned_line.get('text_raw') != original_line_text:
                    total_lines_cleaned += 1
                    cleaned_line['text_original'] = original_line_text
                
                cleaned_lines.append(cleaned_line)
            
            cleaned_lines_by_engine[engine_name] = cleaned_lines
        
        # Guardar las líneas corregidas en un archivo intermedio
        cleaned_lines_path = os.path.join(output_dir, f"{doc_id}_cleaned_lines.json")
        if self.output_flags.get('cleaned_lines', False):
            self.json_handler.save(
                data=cleaned_lines_by_engine,
                output_dir=output_dir,
                file_name_with_extension=f"{doc_id}_cleaned_lines.json"
            )
            logger.info(f"Líneas corregidas guardadas en: {cleaned_lines_path}")
        else:
            logger.info("El guardado de cleaned_lines está desactivado por configuración.")
        
        logger.info(f"Limpieza completada: {total_lines_cleaned}/{total_lines_processed} líneas corregidas")
        return cleaned_lines_by_engine, total_lines_cleaned
    
    # ... (El resto de los métodos de la clase permanecen sin cambios) ...
    def get_batch_corrections(self, reconstructed_lines_by_engine: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Obtiene todas las correcciones propuestas para modo batch.
        """
        if not self.text_cleaner:
            return []
        
        corrections = []
        
        for engine_name, lines in reconstructed_lines_by_engine.items():
            for line_idx, line in enumerate(lines):
                text_raw = line.get('text_raw', '')
                avg_confidence = line.get('avg_constituent_confidence', 100.0)
                
                # Solo procesar si confianza es baja
                if avg_confidence < self.text_cleaner.low_confidence_threshold:
                    # Aplicar corrección
                    corrected_text = self.text_cleaner.clean_text(text_raw, avg_confidence)
                    
                    if corrected_text != text_raw:
                        corrections.append({
                            'engine': engine_name,
                            'line_number': line_idx + 1,
                            'original_text': text_raw,
                            'corrected_text': corrected_text,
                            'confidence': avg_confidence,
                            'similarity': self.text_cleaner._calculate_similarity(text_raw, corrected_text)
                        })
        
        return corrections

    def apply_batch_corrections(self, reconstructed_lines_by_engine: Dict[str, List[Dict]], confirmed_corrections: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Aplica las correcciones confirmadas del modo batch.
        """
        if not confirmed_corrections:
            return reconstructed_lines_by_engine
        
        # Crear copia para no modificar el original
        corrected_lines_by_engine = {}
        
        for engine_name, lines in reconstructed_lines_by_engine.items():
            corrected_lines = lines.copy()
            
            # Aplicar correcciones para este motor
            engine_corrections = [c for c in confirmed_corrections if c['engine'] == engine_name]
            
            for correction in engine_corrections:
                line_idx = correction['line_number'] - 1
                
                if 0 <= line_idx < len(corrected_lines):
                    corrected_lines[line_idx]['text_raw'] = correction['corrected_text']
                    corrected_lines[line_idx]['text_original'] = correction['original_text']
                    logger.info(f"Aplicada corrección batch: {engine_name} línea {line_idx+1}")
            
            corrected_lines_by_engine[engine_name] = corrected_lines
        
        return corrected_lines_by_engine
    
    def get_cleaning_summary(self, original_lines: Dict[str, List[Dict]], cleaned_lines: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Genera un resumen de la limpieza aplicada.
        """
        summary = {
            'total_lines_processed': 0,
            'total_lines_cleaned': 0,
            'engines_processed': list(original_lines.keys()),
            'cleaning_stats_by_engine': {}
        }
        
        for engine_name in original_lines.keys():
            original_count = len(original_lines.get(engine_name, []))
            cleaned_count = len(cleaned_lines.get(engine_name, []))
            
            # Contar líneas que cambiaron
            changed_count = 0
            if engine_name in original_lines and engine_name in cleaned_lines:
                for orig_line, clean_line in zip(original_lines[engine_name], cleaned_lines[engine_name]):
                    if orig_line.get('text_raw') != clean_line.get('text_raw'):
                        changed_count += 1
            
            summary['total_lines_processed'] += original_count
            summary['total_lines_cleaned'] += changed_count
            
            summary['cleaning_stats_by_engine'][engine_name] = {
                'original_lines': original_count,
                'cleaned_lines': cleaned_count,
                'changed_lines': changed_count,
                'change_percentage': (changed_count / original_count * 100) if original_count > 0 else 0
            }
        
        return summary

    def _apply_text_cleaning_to_lines(self, reconstructed_lines_by_engine: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Aplica limpieza de texto SOLO a líneas con confianza < 90%.
        """
        if not self.text_cleaner:
            return reconstructed_lines_by_engine
        
        cleaned_lines_by_engine = {}
        lines_processed = 0
        lines_cleaned = 0
        
        for engine_name, lines in reconstructed_lines_by_engine.items():
            cleaned_lines = []
            
            for line in lines:
                text_raw = line.get('text_raw', '')
                avg_confidence = line.get('avg_constituent_confidence', 100.0)
                
                # NUEVO: Solo procesar si confianza < 90%
                if avg_confidence < self.text_cleaner.low_confidence_threshold:
                    lines_processed += 1
                    cleaned_text = self.text_cleaner.clean_text(text_raw, avg_confidence)
                    
                    if cleaned_text != text_raw:
                        lines_cleaned += 1
                    
                    # Crear línea limpia
                    cleaned_line = line.copy()
                    cleaned_line['text_raw'] = cleaned_text
                    cleaned_line['text_original'] = text_raw
                    cleaned_lines.append(cleaned_line)
                else:
                    # Confianza alta, no procesar
                    cleaned_lines.append(line)
            
            cleaned_lines_by_engine[engine_name] = cleaned_lines
        
        if lines_processed > 0:
            logger.info(f"Limpieza condicional: {lines_cleaned}/{lines_processed} líneas corregidas (confianza < 90%)")
        else:
            logger.info("No se encontraron líneas con confianza < 90%. TextCleaner no activado.")
        
        return cleaned_lines_by_engine