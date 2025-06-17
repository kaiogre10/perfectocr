import logging
import re
import os
import yaml
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class TextCorrector:
    def __init__(self, postprocessing_config_path: str, project_root_path: str):
        with open(postprocessing_config_path, 'r', encoding='utf-8') as f:
            pp_config = yaml.safe_load(f)

        self.vocab = []
        self.min_confidence_spell = 85.0

        # Obtener configuración de corrección
        correction_config = pp_config.get('correction', {})
        relative_vocab_path = correction_config.get('vocab_path')

        if relative_vocab_path:
            absolute_vocab_path = os.path.join(project_root_path, relative_vocab_path)
            if os.path.exists(absolute_vocab_path):
                try:
                    self.vocab = self._load_vocab(absolute_vocab_path)
                    logger.info(f"Vocabulario cargado desde {absolute_vocab_path} con {len(self.vocab)} palabras.")
                except Exception as e:
                    logger.error(f"No se pudo cargar el vocabulario desde {absolute_vocab_path}: {e}")
            else:
                logger.warning(f"Archivo de vocabulario no encontrado en: {absolute_vocab_path}. La corrección ortográfica podría no funcionar.")
        else:
            logger.info("No se especificó vocab_path en la configuración. La corrección ortográfica basada en vocabulario estará limitada o desactivada.")
        
        self.min_confidence_spell = float(correction_config.get('min_confidence', 85.0))

        # Definir errores comunes
        self.common_errors = correction_config.get('common_errors', {})
        self.vocab = self.load_vocabulary(relative_vocab_path) if relative_vocab_path else set()

    def _load_vocab(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def load_vocabulary(self, vocab_path: str) -> set:
        if vocab_path:
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    return set(line.strip().lower() for line in f if line.strip())
            except Exception as e:
                logger.error(f"Error cargando el vocabulario desde {vocab_path}: {e}")
                return set()
        else:
            logger.info("No se especificó vocab_path en la configuración. La corrección ortográfica basada en vocabulario estará limitada o desactivada.")
            return set()

    def fix_common_errors(self, text: str) -> str:
        """Corrige errores recurrentes del OCR"""
        for wrong, right in self.common_errors.items():
            text = text.replace(wrong, right)
        return text

    def correct_spelling(self, text: str) -> str:
        """Corrección ortográfica basada en vocabulario"""
        if not self.vocab:
            return text
            
        words = re.findall(r'\b\w+\b', text)
        corrected_words = []
        
        for word in words:
            if word.lower() in self.vocab:
                corrected_words.append(word)
                continue
                
            # Búsqueda aproximada con RapidFuzz
            matches = process.extract(word.lower(), self.vocab, scorer=fuzz.WRatio, limit=1)
            if matches:
                best_match, score, _ = matches[0]
                if score >= self.min_confidence_spell:
                    # Lógica para preservar capitalización
                    if word.isupper():
                        corrected_words.append(best_match.upper())
                    elif word.istitle():
                        corrected_words.append(best_match.capitalize())
                    else:
                        corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        # Reconstruir el texto manteniendo la estructura original
        if words and len(words) == len(corrected_words):
            new_text = text
            for original, corrected in zip(words, corrected_words):
                if original != corrected:
                    new_text = re.sub(r'\b' + re.escape(original) + r'\b', corrected, new_text, count=1)
            return new_text
        
        return text

    def contextual_correction(self, text: str, context_rules: Dict[str, List[str]] = None) -> str:
        """Corrección basada en reglas contextuales"""
        if not context_rules:
            context_rules = {
                r'\b(cliente|proveedor)\b': ['nombre', 'dirección', 'teléfono'],
                r'\b(total|subtotal)\b': ['$', '€', 'USD']
            }
        
        for pattern, expected_context in context_rules.items():
            matches = re.finditer(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                context_window = text[max(0, match.start()-50):match.end()+50]
                if not any(ctx in context_window for ctx in expected_context):
                    text = text.replace(match.group(), f"[VERIFICAR:{match.group()}]")
        
        return text

class SemanticTableCorrector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.money_pattern = re.compile(r'(\$?\s*[\d,]+\.?\d{1,2})')
        self.number_pattern = re.compile(r'^\d+(\.\d+)?$')
        self.tolerance = self.config.get('arithmetic_tolerance', 0.10)
        logger.info(f"SemanticTableCorrector inicializado con tolerancia aritmética de {self.tolerance}")

    def _str_to_float(self, s: str) -> Optional[float]:
        if not isinstance(s, str): return None
        # Limpiar el string de cualquier caracter no numérico (excepto punto decimal)
        cleaned_s = re.sub(r'[^\d.]', '', s)
        try:
            return float(cleaned_s)
        except (ValueError, TypeError):
            return None

    def _find_col_indices(self, semantic_types: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Encuentra los índices de las columnas clave."""
        try:
            desc_idx = semantic_types.index("descriptivo")
        except ValueError:
            desc_idx = -1 # No es fatal, pero se logueará más adelante.

        # Heurística para PRECIO vs IMPORTE: PRECIO suele estar antes de IMPORTE
        cuant_indices = [i for i, stype in enumerate(semantic_types) if stype == "cuantitativo"]
        
        if len(cuant_indices) < 2:
            return None, -1, -1, desc_idx if desc_idx !=-1 else None

        # Asumir que Cantidad es el primer cuantitativo, Precio el segundo e Importe el tercero.
        # Esto es una heurística fuerte pero común.
        qty_idx = cuant_indices[0] if len(cuant_indices) > 0 else None
        price_idx = cuant_indices[1] if len(cuant_indices) > 1 else None
        total_idx = cuant_indices[2] if len(cuant_indices) > 2 else None

        # Si solo hay dos cuantitativos, el segundo es probablemente el importe.
        if not total_idx and price_idx:
            total_idx = price_idx
            price_idx = None # No podemos inferir el precio unitario

        return qty_idx, price_idx, total_idx, desc_idx

    def _basic_cleanup(self, matrix, sem_types):
        money = re.compile(r'\$\d[\d,]*\.?\d{0,2}|\d+\.\d{2}')
        int_re = re.compile(r'^\d{1,2}$')
        cleaned = []
        for row in matrix:
            new = [{'cell_text':'','words':[]} for _ in sem_types]
            full = ' '.join(c['cell_text'] for c in row).strip()

            # extraer importes de derecha a izquierda
            monies = money.findall(full)[-2:]         # máx. 2 (PU y MTL)
            for idx, val in enumerate(reversed(monies)):
                col = len(sem_types)-1-idx            # derecha (IMPORTE, luego PRECIO)
                new[col]['cell_text'] = val
                full = full.replace(val, '', 1)

            # extraer cantidad (1-2 dígitos) al inicio
            m = re.match(r'(\d{1,2})\s+', full)
            if m:
                qty = m.group(1)
                new[0]['cell_text'] = qty            # columna CANT.
                full = full[m.end():]

            # lo demás es descripción
            try:
                desc_idx = sem_types.index('descriptivo')
                new[desc_idx]['cell_text'] = re.sub(r'\s+', ' ', full).strip()
            except ValueError:
                # Si no hay columna descriptiva, usar la primera columna que no tenga contenido
                for i, cell in enumerate(new):
                    if not cell['cell_text']:
                        new[i]['cell_text'] = re.sub(r'\s+', ' ', full).strip()
                        break
            cleaned.append(new)
        return cleaned

    def _arithmetic_validate(self, matrix, sem_types):
        """
        Recorre cada fila, valida c * pu ≈ mtl y rellena la celda faltante
        cuando haya al menos dos de los tres valores.
        """
        qty_idx, price_idx, total_idx, desc_idx = self._find_col_indices(sem_types)
        if qty_idx is None or total_idx is None or desc_idx is None:
            logger.warning("No se pudieron determinar las columnas clave. Se omite validación aritmética.")
            return matrix

        corrected = []
        for row in matrix:
            # Copia profunda para no modificar la entrada
            new_row = [dict(cell) for cell in row]
            qty_val   = self._str_to_float(new_row[qty_idx]['cell_text'])
            price_val = self._str_to_float(new_row[price_idx]['cell_text']) if price_idx is not None else None
            total_val = self._str_to_float(new_row[total_idx]['cell_text'])

            # ----- Calcular o corregir -----
            if qty_val and price_val and not total_val:
                new_row[total_idx]['cell_text'] = f"{qty_val*price_val:.2f}"

            elif qty_val and total_val and not price_val and qty_val != 0:
                calc = total_val / qty_val
                new_row[price_idx]['cell_text'] = f"{calc:.2f}"

            elif price_val and total_val and not qty_val and price_val != 0:
                calc_qty = round(total_val / price_val)
                if abs(calc_qty*price_val - total_val) <= self.tolerance*total_val:
                    new_row[qty_idx]['cell_text'] = str(calc_qty)

            corrected.append(new_row)
        return corrected

    def correct_matrix(self, matrix, header_elements):
        sem_types = [h.get('semantic_type','descriptivo') for h in header_elements]
        step1 = self._basic_cleanup(matrix, sem_types)
        step2 = self._arithmetic_validate(step1, sem_types)
        return step2