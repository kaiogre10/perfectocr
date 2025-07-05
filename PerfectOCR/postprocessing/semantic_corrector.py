# PerfectOCR/core/postprocessing/correctors.py
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class SemanticTableCorrector:
    """
    Reconstruye las filas de una tabla desde sus componentes atómicos (palabras)
    utilizando un algoritmo de anclaje bidireccional que respeta el orden
    lineal del texto, la compatibilidad semántica y la afinidad geométrica.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el corrector.
        
        Args:
            config: El diccionario de configuración del post-procesamiento.
        """
        self.config = config if config is not None else {}
        self.quantitative_whitelist = set(self.config.get('quantitative_char_whitelist', 
            ['.', ',', '$', '€', 'KG', 'LB', 'OZ']))

    def correct_matrix(self, matrix: List[List[Dict[str, Any]]], header_elements: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Punto de entrada principal. Orquesta la corrección de la tabla completa,
        procesando cada fila de manera individual e independiente.
        """
        if not header_elements:
            return [[cell.get('cell_text', '') for cell in row] for row in matrix]

        header_info = self._prepare_header_info(header_elements)
        
        reconstructed_matrix = []
        for i, row_cells in enumerate(matrix):
            try:
                refined_row = self._refine_single_row(row_cells, header_info)
                reconstructed_matrix.append(refined_row)
            except Exception as e:
                original_row_text = [cell.get('cell_text', '') for cell in row_cells]
                reconstructed_matrix.append(original_row_text)
        
        return reconstructed_matrix

    def _prepare_header_info(self, header_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pre-procesa la información de los encabezados para un acceso eficiente."""
        header_info = [{
            "text": h.get("text_raw", ""),
            "centroid": (h.get("cx", 0.0), h.get("cy", 0.0)),
            "semantic_type": h.get("semantic_type", "descriptivo")
        } for h in sorted(header_elements, key=lambda x: x.get('xmin', 0))]
        
        return header_info

    def _refine_single_row(self, row_cells: List[Dict[str, Any]], header_info: List[Dict[str, Any]]) -> List[str]:
        """Orquesta el pipeline de 3 fases para reconstruir una única fila."""
        atomic_words = self._phase1_fragment_and_classify(row_cells)
        if not atomic_words:
            return [""] * len(header_info)

        assigned_cells = self._phase2_bidirectional_reassignment(atomic_words, header_info)
        recomposed_row = self._phase3_recompose_cells(assigned_cells)
        return recomposed_row

    def _phase1_fragment_and_classify(self, row_cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fragmenta, fusiona y clasifica palabras, mostrando un resumen final."""
        # --- A. fragmentación por espacios ---
        words_raw = [w for c in row_cells for w in c.get('words', [])]
        fragments: List[Dict[str, Any]] = []
        for w in words_raw:
            txt = w.get('text', '').strip()
            if ' ' in txt:
                subs = txt.split()
                if not subs: continue
                tot = len("".join(subs))
                cur_x = w.get('xmin', 0)
                for sub in subs:
                    ratio = len(sub) / tot if tot else 0
                    width = w.get('width', 0) * ratio
                    fragments.append({**w, 'text': sub, 'xmin': cur_x, 'xmax': cur_x + width, 'width': width, 'cx': cur_x + width / 2})
                    cur_x += width
            elif txt:
                fragments.append(w)
        fragments.sort(key=lambda f: f.get('xmin', 0))
        if not fragments: return []

        # --- B. fusión símbolo-número ---
        merged: List[Dict[str, Any]] = []
        i = 0
        while i < len(fragments):
            cur = fragments[i]
            if (i + 1 < len(fragments) and
                cur.get('text', '').upper() in self.quantitative_whitelist and
                re.fullmatch(r'[\d.,]+', fragments[i + 1].get('text', ''))):
                nxt = fragments[i + 1]
                merged.append({**cur, 'text': f"{cur['text']} {nxt['text']}", 'xmax': nxt['xmax'], 'width': nxt['xmax'] - cur['xmin'], 'cx': (cur['xmin'] + nxt['xmax']) / 2})
                i += 2
            else:
                merged.append(cur)
                i += 1

        # --- C. clasificación ---
        numeric_extra = set(self.quantitative_whitelist) | {'.', ',', ' '}
        allowed_mixed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for w in merged:
            t = w.get('text', '')
            if any(ch.isdigit() for ch in t):
                residue = {ch.upper() for ch in t} - set("0123456789") - numeric_extra
                if not residue:
                    w['word_type'] = 'numeric'
                else:
                    residue2 = {ch.upper() for ch in t} - allowed_mixed
                    w['word_type'] = 'mixed' if not residue2 else 'textual'
            else:
                w['word_type'] = 'textual'

        return merged

    def _phase2_bidirectional_reassignment(self, atomic_words: List[Dict[str, Any]], header_info: List[Dict[str, Any]]) -> List[List[Dict]]:
        """Asigna palabras desde los extremos con bloqueo condicional."""
        num_cols = len(header_info)
        assigned_cells = [[] for _ in range(num_cols)]
        unassigned_words = list(atomic_words)
        
        col_boundaries = {'left': 0, 'right': num_cols - 1}
        anchored_quants = set()
        anchored_ids = set()

        while unassigned_words and col_boundaries['left'] <= col_boundaries['right']:
            best_left = self._find_best_col_for_word(unassigned_words[0], header_info, col_boundaries)
            best_right = self._find_best_col_for_word(unassigned_words[-1], header_info, col_boundaries) if len(unassigned_words) > 1 else None

            # Regla de anclaje: no permitir un segundo NÚMERO o ID en una columna ya anclada
            if best_left:
                header = header_info[best_left['col_idx']]
                word_type = unassigned_words[0].get('word_type')
                if (header['semantic_type'] == 'cuantitativo' and word_type == 'numeric' and best_left['col_idx'] in anchored_quants) or \
                   (header['semantic_type'] == 'identificador' and word_type == 'mixed' and best_left['col_idx'] in anchored_ids):
                    best_left = None
            
            if best_right:
                header = header_info[best_right['col_idx']]
                word_type = unassigned_words[-1].get('word_type')
                if (header['semantic_type'] == 'cuantitativo' and word_type == 'numeric' and best_right['col_idx'] in anchored_quants) or \
                   (header['semantic_type'] == 'identificador' and word_type == 'mixed' and best_right['col_idx'] in anchored_ids):
                    best_right = None

            if best_left and (not best_right or best_left['score'] >= best_right['score']):
                word_to_assign = unassigned_words.pop(0)
                col_idx = best_left['col_idx']
                header_type = header_info[col_idx]['semantic_type']
                assigned_cells[col_idx].append(word_to_assign)
                
                # Bloqueo estricto para quant/id, flexible para descriptivo
                if header_type in ['cuantitativo', 'identificador']:
                    col_boundaries['left'] = col_idx + 1
                else:
                    col_boundaries['left'] = col_idx

                if header_type == 'cuantitativo' and word_to_assign.get('word_type') == 'numeric':
                    anchored_quants.add(col_idx)
                if header_type == 'identificador' and word_to_assign.get('word_type') == 'mixed':
                    anchored_ids.add(col_idx)
            
            elif best_right:
                word_to_assign = unassigned_words.pop(-1)
                col_idx = best_right['col_idx']
                header_type = header_info[col_idx]['semantic_type']
                assigned_cells[col_idx].append(word_to_assign)

                # Bloqueo estricto para quant/id, flexible para descriptivo
                if header_type in ['cuantitativo', 'identificador']:
                    col_boundaries['right'] = col_idx - 1
                else:
                    col_boundaries['right'] = col_idx

                if header_type == 'cuantitativo' and word_to_assign.get('word_type') == 'numeric':
                    anchored_quants.add(col_idx)
                if header_type == 'identificador' and word_to_assign.get('word_type') == 'mixed':
                    anchored_ids.add(col_idx)
            
            else:
                break
        
        return assigned_cells

    def _find_best_col_for_word(self, word: Dict[str, Any], header_info: List[Dict[str, Any]], boundaries: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Encuentra la mejor columna para una palabra, considerando semántica como filtro y geometría como desempate."""
        best_option = {'score': -1, 'col_idx': -1}
        word_centroid = (word.get('cx', 0), word.get('cy', 0))

        for col_idx in range(boundaries['left'], boundaries['right'] + 1):
            header = header_info[col_idx]
            
            # Filtro 1: Compatibilidad Semántica
            if not self._is_semantically_compatible(word, header['semantic_type']):
                continue
            
            # Desempate 2: Afinidad Geométrica
            geom_affinity = self._calculate_cosine_similarity(word_centroid, header['centroid'])

            if geom_affinity > best_option['score']:
                best_option['score'] = geom_affinity
                best_option['col_idx'] = col_idx

        result = best_option if best_option['col_idx'] != -1 else None
        return result

    def _phase3_recompose_cells(self, assigned_cells: List[List[Dict[str, Any]]]) -> List[str]:
        """Convierte las celdas con palabras asignadas en una fila de strings finales."""
        final_row = []
        for i, cell_words in enumerate(assigned_cells):
            cell_words.sort(key=lambda w: w.get('xmin', 0))
            cell_text = " ".join([word.get('text', '') for word in cell_words])
            final_row.append(cell_text)
        
        return final_row

    def _is_semantically_compatible(self, word: Dict[str, Any], col_type: str) -> bool:
        """Verifica si una palabra es compatible con el tipo semántico de una columna, aplicando la whitelist."""
        word_type = word.get('word_type', 'textual')
        text = word.get('text', '')

        if col_type == 'cuantitativo':
            return word_type == 'numeric' or text.upper() in self.quantitative_whitelist
        elif col_type == 'descriptivo':
            return True
        elif col_type == 'identificador':
            # mixed válido = letras+numeros SIN símbolos especiales
            return word_type == 'mixed' and re.fullmatch(r'[A-Za-z0-9]+', text.replace(" ", ""))
        return False

    def _calculate_cosine_similarity(self, word_centroid: Tuple[float, float], header_centroid: Tuple[float, float]) -> float:
        """
        Calcula la afinidad geométrica entre la palabra y el encabezado, dando más peso
        a la alineación horizontal (x) que a la vertical (y).
        """
        word_cx, word_cy = word_centroid
        header_cx, header_cy = header_centroid

        # Distancia horizontal (más importante)
        dx = abs(word_cx - header_cx)
        
        # Normalizar para que valores más pequeños sean mejores
        # Convertimos la distancia en una medida de similitud
        # Usamos una función exponencial negativa para que distancias pequeñas den valores cercanos a 1
        # y distancias grandes den valores cercanos a 0
        similarity = np.exp(-dx / 100)  # El divisor 100 controla la "sensibilidad" de la función
        
        return similarity
