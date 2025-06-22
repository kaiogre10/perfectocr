# PerfectOCR/core/text_cleaning/text_cleaner.py
import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Limpiador de texto específico para ruido OCR.
    Corrección puramente semántica y contextual usando embeddings.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Configuración de embeddings
        self.use_embeddings = self.config.get('use_embeddings', True)
        self.embedding_model = self.config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.4)
        self.correct_low_confidence = self.config.get('correct_low_confidence', True)
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 90.0)
        
        # Configuración de modo interactivo
        self.interactive_mode = self.config.get('interactive_mode', False)
        self.batch_confirmation = self.config.get('batch_confirmation', True)
        self.show_similarity_scores = self.config.get('show_similarity_scores', True)
        self.min_confidence_for_interaction = self.config.get('min_confidence_for_interaction', 70.0)
        
        # Cargar modelo si se requiere embeddings
        if self.use_embeddings:
            embedding_manager.load_model(self.embedding_model)
        
        logger.info(f"TextCleaner inicializado - Activación condicional por confianza")

    def _detect_and_correct_anomalies(self, text: str, confidence: float) -> str:
        """
        Detecta y corrige anomalías usando el gestor centralizado de embeddings.
        """
        if not embedding_manager.is_available():
            return text  # Sin embeddings, devolver texto original
        
        if confidence >= self.low_confidence_threshold:
            return text
        
        try:
            # Dividir texto en palabras
            words = re.findall(r'\b\w+\b', text.lower())
            
            if len(words) < 2:
                return text
            
            corrected_text = text
            
            for i, word in enumerate(words):
                if len(word) < 3:
                    continue
                
                # Crear contexto con palabras vecinas
                context_words = self._get_context_words(words, i, window_size=2)
                if not context_words:
                    continue
                
                # Generar variaciones de la palabra actual
                variations = self._generate_semantic_variations(word)
                
                if not variations:
                    continue
                
                # Usar embedding_manager para evaluar semánticamente
                best_variation = self._evaluate_semantic_fit(word, variations, context_words)
                
                if best_variation and best_variation != word:
                    corrected_text = self._replace_word_preserving_case(corrected_text, word, best_variation)
            
            return corrected_text
            
        except Exception as e:
            return text

    def _get_context_words(self, words: List[str], current_idx: int, window_size: int = 2) -> List[str]:
        """Obtiene palabras de contexto alrededor de la palabra actual."""
        context_words = []
        
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(words), current_idx + window_size + 1)
        
        for i in range(start_idx, end_idx):
            if i != current_idx and len(words[i]) >= 3:
                context_words.append(words[i])
        
        return context_words

    def _generate_semantic_variations(self, word: str) -> List[str]:
        """
        Genera variaciones semánticas de la palabra usando patrones comunes de OCR.
        """
        variations = []
        
        # Patrones de confusión típicos del OCR
        char_variations = {
            'a': ['o', 'e'], 'o': ['a', 'e'], 'e': ['a', 'o'],
            'i': ['l', '1'], 'l': ['i', '1'], '1': ['i', 'l'],
            'n': ['m', 'h'], 'm': ['n', 'h'], 'h': ['n', 'm'],
            'r': ['n', 'm'], 'u': ['v', 'w'], 'v': ['u', 'w'],
            's': ['5', 'f'], '5': ['s', 'f'], 'f': ['s', '5'],
            't': ['7', 'f'], '7': ['t', 'f'],
            'g': ['9', 'q'], '9': ['g', 'q'], 'q': ['g', '9'],
            'z': ['2', 's'], '2': ['z', 's'],
            'b': ['8', 'h'], '8': ['b', 'h'],
            'c': ['e', 'o'], 'd': ['b', 'p'], 'p': ['b', 'd']
        }
        
        # Generar variaciones cambiando un carácter a la vez
        for i, char in enumerate(word):
            if char in char_variations:
                for replacement in char_variations[char]:
                    variation = word[:i] + replacement + word[i+1:]
                    variations.append(variation)
        
        return variations

    def _evaluate_semantic_fit(self, original_word: str, variations: List[str], context_words: List[str]) -> Optional[str]:
        """
        Evalúa cuál variación se ajusta mejor semánticamente al contexto.
        """
        if not variations or not context_words:
            return None
        
        try:
            # Crear frases de contexto con cada variación
            test_phrases = []
            for variation in variations:
                test_phrase = f"{' '.join(context_words[:len(context_words)//2])} {variation} {' '.join(context_words[len(context_words)//2:])}"
                test_phrases.append(test_phrase)
            
            # Frase original para comparar
            original_phrase = f"{' '.join(context_words[:len(context_words)//2])} {original_word} {' '.join(context_words[len(context_words)//2:])}"
            
            # Usar embedding_manager para calcular similitudes
            similarities = embedding_manager.calculate_batch_similarity(original_phrase, test_phrases)
            
            # Encontrar la mejor variación
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # Solo aceptar si la similitud es significativamente mejor
            if best_similarity > self.anomaly_threshold:
                return variations[best_idx]
            
            return None
            
        except Exception as e:
            return None

    def _replace_word_preserving_case(self, text: str, old_word: str, new_word: str) -> str:
        """
        Reemplaza una palabra preservando el case original.
        """
        pattern = re.compile(re.escape(old_word), re.IGNORECASE)
        return pattern.sub(new_word, text, count=1)

    def clean_text(self, text: str, confidence: Optional[float] = None) -> str:
        """
        Limpia el texto SOLO si confianza < 90%.
        PRESERVA espaciado original para mantener propiedades geométricas.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Verificar confianza ANTES de cualquier procesamiento
        if confidence is None or confidence >= self.low_confidence_threshold:
            return text  # No hacer nada si confianza es alta

        # Paso 0: Normalizar errores numéricos de OCR
        texto_normalizado = self._normalize_numeric_ocr_errors(text)
        if texto_normalizado != text:
            logger.info(f"Normalizado valor numérico: '{text}' → '{texto_normalizado}'")
        text = texto_normalizado

        # Paso 0.5: Eliminar ruido semántico
        text = self._remove_semantic_noise(text)

        original_text = text
        cleaned_text = text

        # Paso 1: Identificar y proteger números
        protected_numbers = self._protect_numeric_values(cleaned_text)
        
        # Paso 2: Restaurar números protegidos
        cleaned_text = self._restore_numeric_values(cleaned_text, protected_numbers)
        
        # Paso 3: Corrección semántica (solo si embeddings están disponibles)
        if self.correct_low_confidence:
            cleaned_text = self._detect_and_correct_anomalies(cleaned_text, confidence)
        
        if cleaned_text != original_text:
            logger.debug(f"Texto corregido: '{original_text[:50]}...' → '{cleaned_text[:50]}...'")
        
        return cleaned_text

    def get_proposed_corrections(self, matrix: List[List[Any]], confidence_data: Optional[Dict] = None) -> List[Dict]:
        """
        Obtiene todas las correcciones propuestas para modo batch.
        """
        corrections = []
        
        for row_idx, row in enumerate(matrix):
            for col_idx, cell in enumerate(row):
                if isinstance(cell, dict):
                    cell_text = cell.get('cell_text', '')
                    cell_confidence = cell.get('confidence', None)
                elif isinstance(cell, str):
                    cell_text = cell
                    cell_confidence = None
                else:
                    cell_text = str(cell) if cell is not None else ""
                    cell_confidence = None
                
                if cell_confidence is None and confidence_data:
                    cell_confidence = self._get_confidence_for_cell(row_idx, col_idx, confidence_data)
                
                # Solo procesar si confianza es baja
                if cell_confidence and cell_confidence < self.low_confidence_threshold:
                    # Aplicar corrección
                    corrected_text = self.clean_text(cell_text, cell_confidence)
                    
                    if corrected_text != cell_text:
                        corrections.append({
                            'line_number': row_idx + 1,
                            'column_number': col_idx + 1,
                            'original_text': cell_text,
                            'corrected_text': corrected_text,
                            'confidence': cell_confidence,
                            'similarity': self._calculate_similarity(cell_text, corrected_text)
                        })
        
        return corrections

    def _calculate_similarity(self, original: str, corrected: str) -> float:
        """Calcula similitud entre texto original y corregido."""
        return embedding_manager.calculate_similarity(original, corrected)

    def apply_confirmed_corrections(self, matrix: List[List[Any]], confirmed_corrections: List[Dict]) -> List[List[str]]:
        """
        Aplica las correcciones confirmadas a la matriz.
        """
        # Convertir matriz a formato simple si es necesario
        simple_matrix = []
        for row in matrix:
            simple_row = []
            for cell in row:
                if isinstance(cell, dict):
                    simple_row.append(cell.get('cell_text', ''))
                else:
                    simple_row.append(str(cell) if cell is not None else "")
            simple_matrix.append(simple_row)
        
        # Aplicar correcciones confirmadas
        for correction in confirmed_corrections:
            line_idx = correction['line_number'] - 1
            col_idx = correction['column_number'] - 1
            
            if 0 <= line_idx < len(simple_matrix) and 0 <= col_idx < len(simple_matrix[line_idx]):
                simple_matrix[line_idx][col_idx] = correction['corrected_text']
                logger.info(f"Aplicada corrección: línea {line_idx+1}, columna {col_idx+1}")
        
        return simple_matrix

    def clean_matrix(self, matrix: List[List[Any]], confidence_data: Optional[Dict] = None) -> List[List[str]]:
        """Limpia toda una matriz aplicando corrección semántica pura."""
        cleaned_matrix = []
        
        for row_idx, row in enumerate(matrix):
            cleaned_row = []
            for col_idx, cell in enumerate(row):
                if isinstance(cell, dict):
                    cell_text = cell.get('cell_text', '')
                    cell_confidence = cell.get('confidence', None)
                elif isinstance(cell, str):
                    cell_text = cell
                    cell_confidence = None
                else:
                    cell_text = str(cell) if cell is not None else ""
                    cell_confidence = None
                
                if cell_confidence is None and confidence_data:
                    cell_confidence = self._get_confidence_for_cell(row_idx, col_idx, confidence_data)
                
                cleaned_cell = self.clean_text(cell_text, cell_confidence)
                cleaned_row.append(cleaned_cell)
            
            cleaned_matrix.append(cleaned_row)
        
        logger.info(f"Matriz corregida semánticamente: {len(matrix)} filas procesadas")
        return cleaned_matrix

    # Métodos de protección de números y utilidades
    def _protect_numeric_values(self, text: str) -> Dict[str, str]:
        """Identifica y protege valores numéricos con placeholders únicos."""
        protected_numbers = {}
        counter = 0
        
        numeric_pattern = r'\b\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?\b'
        
        def replace_number(match):
            nonlocal counter
            placeholder = f"__NUM_{counter:04d}__"
            protected_numbers[placeholder] = match.group(0)
            counter += 1
            return placeholder
        
        protected_text = re.sub(numeric_pattern, replace_number, text)
        logger.debug(f"Protegidos {len(protected_numbers)} valores numéricos")
        return protected_numbers

    def _restore_numeric_values(self, text: str, protected_numbers: Dict[str, str]) -> str:
        """Restaura los valores numéricos originales."""
        for placeholder, original_number in protected_numbers.items():
            text = text.replace(placeholder, original_number)
        return text

    def _get_confidence_for_cell(self, row_idx: int, col_idx: int, confidence_data: Dict) -> Optional[float]:
        """Obtiene la confianza para una celda específica."""
        try:
            if 'cell_confidence' in confidence_data:
                return confidence_data['cell_confidence'].get(f"{row_idx}_{col_idx}")
            elif 'row_confidence' in confidence_data:
                return confidence_data['row_confidence'].get(row_idx)
            elif 'overall_confidence' in confidence_data:
                return confidence_data['overall_confidence']
        except Exception as e:
            logger.debug(f"Error obteniendo confianza para celda [{row_idx},{col_idx}]: {e}")
        
        return None

    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Obtiene estadísticas de la limpieza aplicada."""
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'text_changed': original_text != cleaned_text,
            'numeric_values_preserved': True,
            'semantic_correction_applied': self.use_embeddings,
            'confidence_threshold': self.low_confidence_threshold
        }

    def _normalize_numeric_ocr_errors(self, text: str) -> str:
        """
        Corrige errores comunes de OCR en valores numéricos (comillas, tildes, etc. por punto decimal).
        Solo reemplaza si el símbolo está entre dígitos.
        """
        simbolos = r"`'´‘’‚‛ʻʼʽʾʿˈˊˋˌˍˎˏˑ˒˓˔˕˖˗˘˙˚˛˜˝˞˟ˠˡˢˣˤ˥˦˧˨˩˪˫ˬ˭ˮ˯˰˱˲˳˴˵˶˷˸˹˺˻˼˽˾˿·•,"
        return re.sub(rf"(?<=\d)[{simbolos}](?=\d)", ".", text)

    def _remove_semantic_noise(self, text: str) -> str:
        """
        Elimina palabras/frases que no encajan semánticamente en el contexto de la línea.
        """
        if not embedding_manager.is_available():
            return text

        words = text.split()
        if len(words) < 2:
            return text

        # Embedding de toda la línea (contexto)
        context_embedding = embedding_manager.encode(" ".join(words), convert_to_tensor=True)

        cleaned_words = []
        for word in words:
            word_embedding = embedding_manager.encode(word, convert_to_tensor=True)
            from sentence_transformers.util import cos_sim
            similarity = cos_sim(context_embedding, word_embedding)[0][0].item()
            # Si la similitud es muy baja, consideramos que es ruido
            if similarity > 0.2:  # Puedes ajustar el umbral
                cleaned_words.append(word)
            else:
                logger.info(f"Eliminado por ruido semántico: '{word}' en '{text}' (similitud={similarity:.2f})")

        return " ".join(cleaned_words)