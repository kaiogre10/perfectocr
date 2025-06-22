# PerfectOCR/core/geo_matrix/header_detector.py
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from utils.geometric import get_polygon_y_center, get_polygon_bounds
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class HeaderDetector:
    def __init__(self,
                config: Dict,
                header_keywords_list: List[str],
                page_dimensions: Optional[Dict[str, Any]] = None):
        self.config = config
        
        self.page_dimensions: Dict[str, Any] = {}
        self.page_height: float = 0.0
        self.page_width: float = 0.0

        if page_dimensions:
            self.set_page_dimensions(page_dimensions)
        
        self.header_keywords_list = [str(kw).upper().strip() for kw in header_keywords_list if kw]
        self.table_end_keywords = [str(kw).upper().strip() for kw in config.get('table_end_keywords', [])]
        
        self.header_fuzzy_min_ratio = float(self.config.get('header_detection_fuzzy_min_ratio', 85.0))
        self.min_y_ratio = float(self.config.get('header_min_y_ratio', 0.05))
        self.max_y_ratio = float(self.config.get('header_max_y_ratio', 0.75))
        self.min_keywords_in_line = int(self.config.get('min_header_keywords_in_line', 2))
        self.max_keywords_in_line = int(self.config.get('max_header_keywords_in_line', 5))
        self.min_line_confidence = float(self.config.get('min_line_confidence_for_header', 70.0))
        self.max_header_line_gap_factor = float(self.config.get('max_header_line_gap_factor', 2.5))
        self.default_line_height_for_gap = float(self.config.get('default_line_height_for_gap', 20.0))

    def set_page_dimensions(self, page_dimensions_input: Dict[str, Any]):
        if page_dimensions_input and page_dimensions_input.get('width') and page_dimensions_input.get('height'):
            self.page_dimensions = page_dimensions_input.copy() 
            self.page_height = float(page_dimensions_input['height'])
            self.page_width = float(page_dimensions_input['width'])
            logger.info(f"HeaderDetector.set_page_dimensions: Successfully set W:{self.page_width}, H:{self.page_height}")
        else:
            self.page_dimensions = {}
            self.page_height = 0.0
            self.page_width = 0.0

    def _expand_header_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        Divide los elementos de texto que contienen múltiples palabras clave.
        """
        if not self.header_keywords_list:
            return elements

        expanded_elements = []
        for elem in elements:
            elem_text = elem.get('text', elem.get('text_raw', '')).strip()
            keywords_found_in_elem = [kw for kw in self.header_keywords_list if kw in elem_text.upper()]
            
            if len(keywords_found_in_elem) > 1 and ' ' in elem_text:
                logger.debug(f"Element '{elem_text}' contains multiple keywords {keywords_found_in_elem}. Attempting to split.")
                words_split = elem_text.split()
                if len(words_split) > 1 and elem.get('xmin') is not None and elem.get('width', 0) > 0:
                    total_chars = sum(len(w) for w in words_split)
                    if total_chars > 0:
                        current_x = float(elem['xmin'])
                        elem_width = float(elem['width'])
                        for i, word_text in enumerate(words_split):
                            word_char_ratio = len(word_text) / total_chars
                            word_width = elem_width * word_char_ratio
                            
                            pseudo_elem = elem.copy()
                            pseudo_elem['text'] = word_text
                            pseudo_elem['text_raw'] = word_text
                            pseudo_elem['xmin'] = current_x
                            pseudo_elem['xmax'] = current_x + word_width
                            pseudo_elem['cx'] = current_x + (word_width / 2)
                            pseudo_elem['width'] = word_width
                            expanded_elements.append(pseudo_elem)
                            current_x += word_width
                        continue
            expanded_elements.append(elem)
        
        return expanded_elements

    def _get_words_from_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_words: List[Dict[str, Any]] = []
        for line in lines:
            words_in_line = line.get('constituent_elements_ocr_data', [])
            if isinstance(words_in_line, list):
                for word in words_in_line:
                    if isinstance(word, dict): 
                        all_words.append(word)
            else:
                logger.warning(f"Line {line.get('line_id', 'N/A')} has 'constituent_elements_ocr_data' not as a list.")
        return all_words

    def _is_line_a_header_candidate(self, line_obj: Dict[str, Any]) -> bool:
        line_text_preview_for_debug = str(line_obj.get('text_raw', 'N/A'))[:70]

        if not isinstance(line_obj, dict):
            logger.error(f"HeaderDetector._is_line_a_header_candidate: 'line_obj' is not a dictionary. Received: {type(line_obj)}")
            return False

        current_page_height = self.page_height 
        if not isinstance(current_page_height, (int, float)) or current_page_height <= 0:
            logger.warning(
                f"Line '{line_text_preview_for_debug}': self.page_height is not valid ({current_page_height}). "
                f"Cannot validate Y-zone."
            )
            return False
            
        line_polygon = line_obj.get('polygon_line_bbox')
        if not line_polygon:
            logger.debug(f"Line '{line_text_preview_for_debug}' has no 'polygon_line_bbox'. Skipping as header candidate.")
            return False
            
        try:
            line_y_center = get_polygon_y_center(line_polygon)
        except Exception as e_ycenter: 
            logger.warning(f"Line '{line_text_preview_for_debug}': Exception in get_polygon_y_center: {e_ycenter}. Coords: {line_polygon}. Skipping.")
            return False
        
        min_y_allowed = current_page_height * self.min_y_ratio
        max_y_allowed = current_page_height * self.max_y_ratio

        if not (min_y_allowed <= line_y_center <= max_y_allowed):
            logger.debug(f"Line '{line_text_preview_for_debug}' (Y-center: {line_y_center:.2f}) is outside allowed Y-range ({min_y_allowed:.2f} - {max_y_allowed:.2f}). Skipping.")
            return False

        constituent_elements = line_obj.get('constituent_elements_ocr_data', [])
        if not constituent_elements:
            logger.debug(f"Line '{line_text_preview_for_debug}' has no 'constituent_elements_ocr_data'. Skipping.")
            return False

        if not self.header_keywords_list: 
            logger.warning(f"Line '{line_text_preview_for_debug}': Header keywords list is empty. Cannot find keywords.")
            return False

        elements_for_processing = self._expand_header_elements(constituent_elements)
        if len(elements_for_processing) > len(constituent_elements):
            logger.debug(f"Line '{line_text_preview_for_debug}' had elements expanded from {len(constituent_elements)} to {len(elements_for_processing)}. Updating line object.")
            line_obj['constituent_elements_ocr_data'] = elements_for_processing

        keyword_count = 0
        for elem_dict in elements_for_processing:
            if not isinstance(elem_dict, dict): continue 
            
            word_text_raw = elem_dict.get('text', elem_dict.get('text_raw', '')) 
            text_upper = str(word_text_raw).upper().strip()
            if not text_upper:
                continue
            
            if text_upper in self.header_keywords_list:
                keyword_count += 1
            else: 
                match_result_fuzzy = process.extractOne(
                    text_upper, 
                    self.header_keywords_list, 
                    scorer=fuzz.WRatio, 
                    score_cutoff=self.header_fuzzy_min_ratio
                )
                if match_result_fuzzy:
                    keyword_count += 1
        
        passes_min_keywords = keyword_count >= self.min_keywords_in_line
        passes_max_keywords = keyword_count <= self.max_keywords_in_line
        
        current_line_avg_conf = float(line_obj.get('avg_constituent_confidence', 0.0))
        passes_confidence = current_line_avg_conf >= self.min_line_confidence
        
        final_decision = passes_min_keywords and passes_max_keywords and passes_confidence
        
        return final_decision

    def _assign_semantic_type(self, header_text: str, semantic_keywords: Dict[str, list]) -> str:
        text_upper = header_text.upper().strip()

        # 1) Buscar coincidencia exacta en las listas de palabras clave por categoría
        for sem_type, keywords in semantic_keywords.items():
            for kw in keywords:
                kw_upper = str(kw).upper().strip()
                if kw_upper == text_upper or kw_upper in text_upper or text_upper in kw_upper:
                    return sem_type  # Devolver la CATEGORÍA, no la palabra clave

        # 2) Usar fuzzy matching como fallback
        all_keywords_flat = []
        keyword_to_category = {}
        for sem_type, keywords in semantic_keywords.items():
            for kw in keywords:
                kw_upper = str(kw).upper().strip()
                all_keywords_flat.append(kw_upper)
                keyword_to_category[kw_upper] = sem_type
        
        if all_keywords_flat:
            match_result = process.extractOne(
                text_upper, 
                all_keywords_flat, 
                scorer=fuzz.WRatio, 
                score_cutoff=self.header_fuzzy_min_ratio
            )
            if match_result:
                matched_keyword = match_result[0]
                return keyword_to_category.get(matched_keyword, 'descriptivo')

        # 3) Heurísticas: números ⇒ cuantitativo; tokens de código ⇒ identificador
        if any(ch.isdigit() for ch in text_upper):
            return 'cuantitativo'
        if any(tok in text_upper for tok in ('COD', 'CLAVE', 'ID')):
            return 'identificador'

        # 4) Por defecto
        return 'descriptivo'

    def identify_header_band_and_words(self, formed_lines: List[Dict[str, Any]], semantic_keywords: Optional[Dict[str, list]] = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[float], Optional[float]]:
        if not isinstance(self.page_height, (int, float)) or self.page_height <= 0:
            logger.error(
                f"HeaderDetector.identify_header_band_and_words: Cannot identify headers without "
                f"valid self.page_height. Current self.page_height: {self.page_height} (type: {type(self.page_height)}). "
                f"Associated self.page_dimensions dict was: {self.page_dimensions}. Aborting."
            )
            return None, None, None
        
        if not self.header_keywords_list:
            logger.error("HeaderDetector.identify_header_band_and_words: Header keywords list is empty. Cannot proceed.")
            return None, None, None
        
        potential_header_lines: List[Dict[str, Any]] = []
        for line_to_check in formed_lines:
            if self._is_line_a_header_candidate(line_to_check):
                potential_header_lines.append(line_to_check)
        
        if not potential_header_lines:
            logger.warning("No potential header lines found after filtering with _is_line_a_header_candidate.")
            return None, None, None

        potential_header_lines.sort(key=lambda l: get_polygon_y_center(l.get('polygon_line_bbox', [])))
        
        final_header_lines_block: List[Dict[str, Any]] = []
        if potential_header_lines:
            final_header_lines_block.append(potential_header_lines[0]) 
            
            if len(potential_header_lines) > 1:
                last_line_in_block = potential_header_lines[0]
                
                candidate_heights = []
                for cand_line in potential_header_lines:
                    poly = cand_line.get('polygon_line_bbox')
                    if poly:
                        try:
                            _, ymin_cand, _, ymax_cand = get_polygon_bounds(poly)
                            if ymax_cand > ymin_cand:
                                candidate_heights.append(ymax_cand - ymin_cand)
                        except Exception: 
                            pass 
                
                avg_h = np.median(candidate_heights) if candidate_heights else self.default_line_height_for_gap
                if avg_h <= 0: avg_h = self.default_line_height_for_gap
                logger.debug(f"Grouping header lines. Median line height for gap: {avg_h:.2f}")

                for next_line in potential_header_lines[1:]:
                    next_line_poly = next_line.get('polygon_line_bbox')
                    last_line_poly = last_line_in_block.get('polygon_line_bbox')
                    
                    if not next_line_poly or not last_line_poly:
                        continue 
                    try:
                        _, _, _, last_y_max = get_polygon_bounds(last_line_poly)
                        _, next_y_min, _, _ = get_polygon_bounds(next_line_poly)
                        gap = next_y_min - last_y_max
                        
                        if gap < (avg_h * self.max_header_line_gap_factor):
                            final_header_lines_block.append(next_line)
                            last_line_in_block = next_line
                        else:
                            logger.debug(f"Gap ({gap:.2f}) between '{last_line_in_block.get('text_raw')}' and '{next_line.get('text_raw')}' exceeds threshold ({avg_h * self.max_header_line_gap_factor:.2f}). Ending header block.")
                            break 
                    except Exception as e_gap:
                        logger.warning(f"Exception calculating header line gap: {e_gap}. Ending header block.")
                        break

        if not final_header_lines_block:
            logger.warning("Final header lines block is empty after grouping.")
            return None, None, None
        
        header_words = self._get_words_from_lines(final_header_lines_block)
        if not header_words:
            logger.warning("No words extracted from the final header lines block.")
            return None, None, None
        
        expanded_header_words = self._expand_header_elements(header_words)
        if len(expanded_header_words) > len(header_words):
            logger.info(f"Header words expanded from {len(header_words)} to {len(expanded_header_words)}.")
            header_words = expanded_header_words
        
        # Asignar semantic_type a cada header_word
        if semantic_keywords is not None:
            for hw in header_words:
                hw['semantic_type'] = self._assign_semantic_type(hw.get('text_raw', ''), semantic_keywords)
        
        all_ymins: List[float] = []
        all_ymaxs: List[float] = []
        for line in final_header_lines_block:
            line_poly_for_bounds = line.get('polygon_line_bbox')
            if line_poly_for_bounds:
                try:
                    _, ymin_b, _, ymax_b = get_polygon_bounds(line_poly_for_bounds)
                    all_ymins.append(ymin_b)
                    all_ymaxs.append(ymax_b)
                except Exception as e_bounds:
                    logger.warning(f"Could not get bounds for header line: {e_bounds}")
        
        y_min_band = min(all_ymins) if all_ymins else None
        y_max_band = max(all_ymaxs) if all_ymins else None
        
        if y_min_band is None or y_max_band is None:
            logger.warning("Could not determine Y-bounds of the header band.")
            return header_words, None, None
            
        logger.info(f"Header band identified between Y={y_min_band:.2f} and Y={y_max_band:.2f} with {len(header_words)} words.")
        return header_words, y_min_band, y_max_band
