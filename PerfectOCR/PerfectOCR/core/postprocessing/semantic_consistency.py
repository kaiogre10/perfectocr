import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)

class UnifiedSemanticConsistencyCorrector:
    """
    Analiza y corrige la consistencia semántica vertical de CADA columna en una tabla.
    NO ELIMINA DATOS. En su lugar, los extrae a una lista de cuarentena para
    su posterior reutilización por otros módulos como MathMax.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        # Umbrales de configuración
        self.magnitude_z_score_threshold = 3.0
        self.descriptive_noise_similarity_threshold = 0.5

    def correct_matrix(self, matrix: List[List[str]], semantic_types: List[str]) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """
        Punto de entrada principal.
        
        Returns:
            Tuple[List[List[str]], List[Dict[str, Any]]]: 
                - La matriz corregida.
                - Una lista de datos puestos en cuarentena con su contexto original.
        """
        if not matrix or not semantic_types or len(matrix[0]) != len(semantic_types):
            logger.warning("Matriz o tipos semánticos no válidos. Saltando corrección de consistencia.")
            return matrix, []

        corrected_matrix = [row[:] for row in matrix]
        quarantined_data = []  # Lista de cuarentena con seguimiento de origen

        for col_idx, s_type in enumerate(semantic_types):
            #logger.info(f"--- Analizando Columna {col_idx} (Tipo: {s_type}) ---")
            column_values = [row[col_idx] for row in corrected_matrix]
            
            if s_type == 'cuantitativo':
                profile = self._profile_quantitative_column(column_values)
                if profile.get('is_valid'):
                    for row_idx, cell in enumerate(column_values):
                        corrected_cell = self._correct_quantitative_cell(cell, profile, row_idx, col_idx)
                        corrected_matrix[row_idx][col_idx] = corrected_cell

            elif s_type == 'descriptivo':
                profile = self._profile_descriptive_column(column_values)
                if profile.get('is_valid'):
                    for row_idx, cell in enumerate(column_values):
                        cleaned_cell, extracted_item = self._extract_from_descriptive_cell(cell, profile, row_idx, col_idx)
                        if extracted_item and self._is_embedding_score_low(extracted_item):  # Nuevo chequeo
                            quarantined_data.append({
                                "value": extracted_item,
                                "original_row": row_idx,
                                "original_col": col_idx,
                                "reason": "ruido_grave",
                                "status": "eliminado"  # Solo eliminar si embedding es bajísimo
                            })
                        elif extracted_item:
                            quarantined_data.append({
                                "value": extracted_item,
                                "original_row": row_idx,
                                "original_col": col_idx,
                                "reason": "potencial_ruido",
                                "status": "en_cuarentena"  # Devolver si no se usa
                            })
                            corrected_matrix[row_idx][col_idx] = cleaned_cell if extracted_item else cell  # Devolver si no se usa
                            
        # Fase final: Devolver datos no usados a su origen
        for item in quarantined_data[:]:  # Copia para no modificar durante iteración
            if item["status"] == "en_cuarentena":  # Verificar si se usó en otro lugar
                corrected_matrix[item["original_row"]][item["original_col"]] = item["value"]  # Devolver
                quarantined_data.remove(item)  # Limpiar de la lista
        
        logger.info(f"Corrección de consistencia completada. {len(quarantined_data)} datos en cuarentena final.")
        return corrected_matrix, quarantined_data

    # --- Métodos para Columnas Cuantitativas ---
    def _profile_quantitative_column(self, column: List[str]) -> Dict:
        values = []
        for val_str in column:
            # Limpieza más robusta para encontrar el número principal en una celda
            match = re.search(r'(\d[\d,.]*\d|\d)', val_str)
            if match:
                num_str = match.group(1).replace(',', '')
                try:
                    values.append(float(num_str))
                except ValueError:
                    continue
        
        if not values: return {'is_valid': False}

        profile = {
            'is_valid': True,
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0
        }
        return profile

    def _correct_quantitative_cell(self, cell: str, profile: Dict, row_idx: int, col_idx: int) -> str:
        # Corrige un formato anómalo como "00.96" o "00'96" a "96.00"
        match = re.search(r'^[0.,\s]*([1-9]\d*[\.,`\']?\d*)$', cell.strip())
        if match and profile['std'] > 0:
            potential_number_str = match.group(1).replace('`', '.').replace("'", "")
            try:
                potential_value = float(potential_number_str)
                original_value = float(cell.replace('`', '.').replace("'", "."))

                z_score_potential = abs(potential_value - profile['mean']) / profile['std']
                z_score_original = abs(original_value - profile['mean']) / profile['std']
                
                if z_score_potential < z_score_original and z_score_potential < self.magnitude_z_score_threshold:
                    corrected_value = f"{potential_value:.2f}"
                    logger.info(f"    Corrección Cuantitativa en [{row_idx},{col_idx}]: '{cell}' -> '{corrected_value}' (Z-score mejoró de {z_score_original:.2f} a {z_score_potential:.2f})")
                    return corrected_value
            except (ValueError, TypeError):
                pass
        return cell

    # --- Métodos para Columnas Descriptivas ---
    def _profile_descriptive_column(self, column: List[str]) -> Dict:
        # Extrae solo el texto no numérico para crear el perfil semántico
        text_parts = [re.sub(r'(\d+\.?\d*)', '', cell).strip() for cell in column]
        valid_texts = [text for text in text_parts if len(text) > 3]
        if not valid_texts: return {'is_valid': False}

        # Calcula el "significado promedio" de la columna usando embedding_manager
        embeddings = embedding_manager.encode(valid_texts, convert_to_tensor=False)
        centroid_embedding = np.mean(embeddings, axis=0)
        profile = {'is_valid': True, 'centroid': centroid_embedding}
        return profile

    def _extract_from_descriptive_cell(self, cell: str, profile: Dict, row_idx: int, col_idx: int) -> Tuple[str, Optional[str]]:
        """
        Extrae un número que parece ruido de una celda descriptiva.
        Devuelve la celda limpia y el número extraído.
        """
        match = re.search(r'^(.*?)(\s+(\d{1,}[\.,`\']\d{2}))$', cell)
        if match:
            text_part = match.group(1).strip()
            potential_noise = match.group(2).strip()
            
            # Añadir verificación de estructura: Comparar con perfil de columna
            if self._has_anomalous_structure(potential_noise, profile):
                return text_part, potential_noise  # Marcar para cuarentena
        
        return cell, None

    def _has_anomalous_structure(self, number_str: str, profile: Dict) -> bool:
        # Verificar si el número tiene una estructura diferente al perfil de la columna
        try:
            num_value = float(re.sub(r'[^\d.]', '', number_str))  # Limpiar y convertir
            column_mean = profile.get('mean', 0)
            column_std = profile.get('std', 1)
            z_score = abs(num_value - column_mean) / column_std if column_std > 0 else 0
            return z_score > 3.0  # Umbral alto para anomalías, como en tu punto 4
        except ValueError:
            return False  # No es un número válido, no es anómalo aquí

    def _is_embedding_score_low(self, text: str) -> bool:
        from utils.embedding_manager import embedding_manager  # Asegurar import
        if not embedding_manager.is_available():
            return False  # No eliminar si embeddings no disponibles
        
        # Calcular similitud con contexto general (umbral bajo, e.g., < 0.1)
        context_sample = "texto descriptivo común"  # Puedes ajustar a un contexto real
        similarity = embedding_manager.calculate_similarity(text, context_sample)
        return similarity < 0.1  # Umbral bajísimo como discutimos
