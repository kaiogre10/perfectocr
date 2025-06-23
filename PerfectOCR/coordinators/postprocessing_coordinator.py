# PerfectOCR/coordinators/postprocessing_coordinator.py
import logging
import os
from typing import Dict, Any, List
from core.postprocessing.semantic_corrector import SemanticTableCorrector
from core.postprocessing.semantic_consistency import UnifiedSemanticConsistencyCorrector
from core.postprocessing.math_max import MatrixSolver
import re 
from utils.output_handlers import JsonOutputHandler

logger = logging.getLogger(__name__)

class PostprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str):
        """
        Inicializa el coordinador de post-procesamiento.
        Orquesta el pipeline completo de 3 fases:
        1. Corrección Estructural
        2. Corrección de Consistencia Semántica
        3. Resolución Matricial Aritmética
        """
        self.config = config
        self.project_root = project_root

        postprocessing_cfg = self.config.get('postprocessing', {})
        
        self.structural_corrector = SemanticTableCorrector(config=postprocessing_cfg)
        self.consistency_corrector = UnifiedSemanticConsistencyCorrector(config=postprocessing_cfg)
        self.matrix_solver = MatrixSolver(config=postprocessing_cfg)

        # --- INICIALIZA EL HANDLER DE OUTPUTS JSON ---
        output_config = self.config.get('output_config', {})
        self.json_handler = JsonOutputHandler(config=output_config)

    def correct_table_structure(self, extraction_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica el pipeline de corrección de 3 fases a la matriz de la tabla.
        """
        outputs = extraction_payload.get('outputs')
        if not all(k in outputs for k in ['table_matrix', 'header_elements']):
            logger.error("Payload incompleto. Faltan 'table_matrix' o 'header_elements'.")
            return extraction_payload

        matrix_raw = outputs['table_matrix']
        headers = outputs['header_elements']
        semantic_types = [h.get('semantic_type', 'descriptivo') for h in headers]
        document_totals = outputs.get('document_totals') # opcional
        document_grand_total = outputs.get('document_grand_total')  # El total del documento

        # --- FASE 0: Adaptar Matriz de Entrada ---
        adapted_matrix = self._adapt_matrix_for_corrector(matrix_raw)
        
        # --- FASE 1: Corrección Estructural ---
        logger.info("Fase 1: Ejecutando corrección estructural...")
        structured_matrix = self.structural_corrector.correct_matrix(
            matrix=adapted_matrix,
            header_elements=headers
        )
        
        # --- FASE 2: Corrección de Consistencia Semántica ---
        logger.info("Fase 2: Ejecutando corrección de consistencia semántica...")
        consistency_corrected_matrix, quarantined_data = self.consistency_corrector.correct_matrix(
            matrix=structured_matrix,
            semantic_types=semantic_types
        )
        logger.info(f"Datos en cuarentena después de Fase 2: {quarantined_data}")
        
        # Guardar la matriz después de la corrección de consistencia semántica
        outputs['semantic_consistency_matrix'] = consistency_corrected_matrix

        # --- FASE 3: Resolución Matricial Aritmética ---
        logger.info("Fase 3: Ejecutando resolución matricial (MatrixSolver)...")
        
        # Preparar el formato correcto para document_totals que espera MatrixSolver
        formatted_totals = None
        if document_grand_total is not None:
            formatted_totals = {'total_mtl': document_grand_total}
            logger.info(f"Usando total del documento para validación aritmética: {document_grand_total}")
        
        result = self.matrix_solver.solve(
            matrix=consistency_corrected_matrix,
            semantic_types=semantic_types,
            quarantined_data=quarantined_data,
            document_totals=formatted_totals
        )
        final_matrix = result["matrix"]
        semantic_math_types = result["semantic_math_types"]
        
        # Extraer headers como texto plano
        headers_text = [h.get('text_raw', '') for h in headers]

        math_max_output = {
            "headers": headers_text,
            "semantic_types": semantic_math_types,
            "matrix": final_matrix,
            "semantic_format": "math_assigned"
        }

        output_dir = extraction_payload.get("output_dir")
        doc_id = extraction_payload.get("doc_id")  # O 'base_name'

        self.json_handler.save(
            data=math_max_output,
            output_dir=output_dir,
            file_name_with_extension=f"{doc_id}_math_max_matrix.json",
            output_type="math_max_matrix"
        )
        
        # --- Finalizar y Actualizar Payload ---
        corrected_payload = extraction_payload.copy()
        corrected_payload['outputs'] = outputs.copy()
        corrected_payload['outputs']['table_matrix'] = final_matrix
        corrected_payload['outputs']['math_max_matrix'] = final_matrix
        corrected_payload['outputs']['quarantined_data'] = quarantined_data
        corrected_payload['outputs']['semantic_math_types'] = semantic_math_types
        
        corrected_payload['status'] = 'success_arithmetically_solved'
        corrected_payload['message'] = 'Tabla reconstruida y resuelta aritméticamente.'
        
        return corrected_payload

    def _adapt_matrix_for_corrector(self, matrix: List[List[Any]]) -> List[List[Dict[str, Any]]]:
        """
        Adapta la matriz del formato del TableExtractor al formato esperado por el corrector estructural.
        Asegura que cada 'word' tenga la clave 'text'.
        """
        adapted_matrix = []
        
        for row_idx, row in enumerate(matrix):
            adapted_row = []
            
            for col_idx, cell in enumerate(row):
                if isinstance(cell, dict):
                    working_cell = cell
                else:
                    working_cell = {'cell_text': str(cell) if cell is not None else '', 'words': []}

                adapted_cell = {'cell_text': working_cell.get('cell_text', ''), 'words': []}
                original_words = working_cell.get('words', [])

                if original_words:
                    for word in original_words:
                        if isinstance(word, dict):
                            adapted_word = word.copy()
                            adapted_word['text'] = adapted_cell['cell_text']
                            adapted_cell['words'].append(adapted_word)

                elif adapted_cell['cell_text'].strip():
                    adapted_word = {
                        'text': adapted_cell['cell_text'].strip(),
                        'xmin': col_idx * 100, 'xmax': (col_idx + 1) * 100,
                        'cx': col_idx * 100 + 50, 'cy': row_idx * 100 + 50,
                        'width': 100, 'height': 50
                    }
                    adapted_cell['words'].append(adapted_word)
                
                adapted_row.append(adapted_cell)
            
            adapted_matrix.append(adapted_row)
        
        return adapted_matrix

    # --- DESACTIVADO: Método completo para la validación del total ---
    # def _validate_grand_total(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    #     """Compara la suma de la columna de importes con el total extraído del documento."""
    #     outputs = payload.get('outputs', {})
    #     document_total = outputs.get('document_grand_total')
    #     matrix = outputs.get('table_matrix', [])
    #     headers = outputs.get('header_elements', [])
    #     
    #     if document_total is None:
    #         return {"status": "skipped", "message": "No se extrajo un total del documento para validar."}
    #     if not matrix or not headers:
    #         return {"status": "skipped", "message": "Matriz o cabeceras no disponibles para validación."}
    #
    #     # Identificar la columna de importe (la última cuantitativa antes de la última columna)
    #     total_col_idx = -1
    #     try:
    #         # La heurística más robusta: la columna cuyo header contenga "IMPORTE"
    #         total_col_idx = next(i for i, h in enumerate(headers) if "IMPORTE" in h.get('text_raw','').upper())
    #     except StopIteration:
    #         logger.warning("No se encontró cabecera 'IMPORTE'. Se usará una heurística de posición.")
    #         cuant_indices = [i for i, h in enumerate(headers) if h.get('semantic_type') == 'cuantitativo']
    #         if len(cuant_indices) >= 2:
    #             total_col_idx = cuant_indices[-1] # El último cuantitativo suele ser el total de línea
    #
    #     if total_col_idx == -1:
    #          return {"status": "error", "message": "No se pudo identificar la columna de importes."}
    #
    #     # Sumar los valores de la columna de importe
    #     calculated_sum = 0.0
    #     for row in matrix:
    #         cell = row[total_col_idx]
    #         if isinstance(cell, dict):
    #             cell_text = cell.get('cell_text', '')
    #         else:
    #             cell_text = str(cell) if cell is not None else ''
    #         cleaned_s = re.sub(r'[^\d.]', '', cell_text)
    #         try:
    #             calculated_sum += float(cleaned_s)
    #         except (ValueError, TypeError):
    #             pass # Ignorar celdas no numéricas o con errores
    #
    #     # Comparar con una tolerancia (ej. 5.0, como la validación de fila)
    #     tolerance = self.config.get('semantic_table_correction', {}).get('arithmetic_tolerance', 5.0)
    #     difference = abs(calculated_sum - document_total)
    #     
    #     if difference <= tolerance:
    #         status = "success"
    #         message = f"La suma de importes ({calculated_sum:.2f}) coincide con el total del documento ({document_total:.2f})."
    #     else:
    #         status = "failed"
    #         message = f"La suma de importes ({calculated_sum:.2f}) NO coincide con el total del documento ({document_total:.2f}). Diferencia: {difference:.2f}"
    #
    #     return {
    #         "status": status,
    #         "message": message,
    #         "document_total": document_total,
    #         "calculated_sum": round(calculated_sum, 2),
    #         "difference": round(difference, 2)
    #     }