# PerfectOCR/coordinators/postprocessing_coordinator.py
import logging
import os
from typing import Dict, Any, List
from core.postprocessing.semantic_corrector import SemanticTableCorrector
from core.postprocessing.semantic_consistency import UnifiedSemanticConsistencyCorrector
from core.postprocessing.math_max import MatrixSolver
import re 
from utils.output_handlers import JsonOutputHandler, ExcelOutputHandler
import glob
import json

logger = logging.getLogger(__name__)

class PostprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool]):
        """
        Inicializa el coordinador de post-procesamiento.
        Orquesta el pipeline completo de 3 fases:
        1. Corrección Estructural
        2. Corrección de Consistencia Semántica
        3. Resolución Matricial Aritmética
        """
        self.config = config
        self.project_root = project_root
        self.output_flags = output_flags

        postprocessing_cfg = self.config.get('postprocessing', {})
        
        self.structural_corrector = SemanticTableCorrector(config=postprocessing_cfg)
        self.consistency_corrector = UnifiedSemanticConsistencyCorrector(config=postprocessing_cfg)
        self.matrix_solver = MatrixSolver(config=postprocessing_cfg.get('math_solver', {}))

        # El handler ya no recibe config, solo es herramienta
        self.json_handler = JsonOutputHandler()
        self.excel_handler = ExcelOutputHandler()

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
        document_totals = outputs.get('document_totals')

        adapted_matrix = self._adapt_matrix_for_corrector(matrix_raw)
        
        logger.info("Fase 1: Ejecutando corrección estructural...")
        structured_matrix = self.structural_corrector.correct_matrix(
            matrix=adapted_matrix,
            header_elements=headers
        )
        
        logger.info("Fase 2: Ejecutando corrección de consistencia semántica...")
        consistency_corrected_matrix, quarantined_data = self.consistency_corrector.correct_matrix(
            matrix=structured_matrix,
            semantic_types=semantic_types
        )
        logger.info(f"Datos en cuarentena después de Fase 2: {quarantined_data}")
        
        outputs['semantic_consistency_matrix'] = consistency_corrected_matrix

        logger.info("Fase 3: Ejecutando resolución matricial (MatrixSolver)...")
        
        formatted_totals = None
        if document_totals and document_totals.get('total_mtl') is not None:
            formatted_totals = {'total_mtl': document_totals['total_mtl']}
            logger.info(f"Usando total del documento para validación aritmética: {document_totals['total_mtl']}")
        
        result = self.matrix_solver.solve(
            matrix=consistency_corrected_matrix,
            semantic_types=semantic_types,
            quarantined_data=quarantined_data,
            document_totals=formatted_totals
        )
        final_matrix = result["matrix"]
        semantic_math_types = result["semantic_math_types"]
        
        headers_text = [h.get('text_raw', '') for h in headers]

        math_max_output = {
            "headers": headers_text,
            "semantic_types": semantic_math_types,
            "matrix": final_matrix,
            "semantic_format": "math_assigned"
        }

        output_dir = extraction_payload.get("output_dir")
        doc_id = extraction_payload.get("doc_id")

        # Guardado de outputs según flags explícitos
        if output_dir and doc_id:
            if self.output_flags.get('debug_semantic_matrix', False):
                self._save_simplified_matrix(
                    matrix_data=structured_matrix,
                    header_elements=headers,
                    base_name=doc_id,
                    output_dir=output_dir,
                    suffix="semantically_corrected_matrix",
                    payload=None
                )
            if self.output_flags.get('semantic_consistency_matrix', False):
                self._save_simplified_matrix(
                    matrix_data=consistency_corrected_matrix,
                    header_elements=headers,
                    base_name=doc_id,
                    output_dir=output_dir,
                    suffix="semantic_consistency_matrix",
                    payload=None
                )
            if self.output_flags.get('math_max_matrix', False):
                self._save_simplified_matrix(
                    matrix_data=final_matrix,
                    header_elements=headers,
                    base_name=doc_id,
                    output_dir=output_dir,
                    suffix="math_max_matrix",
                    payload={"outputs": {"semantic_math_types": semantic_math_types}}
                )
            # Guardar el JSON completo de math_max_matrix (única vez)
            if self.output_flags.get('math_max_matrix', False):
                self.json_handler.save(
                    data=math_max_output,
                    output_dir=output_dir,
                    file_name_with_extension=f"{doc_id}_math_max_matrix.json"
                )
                
                # AGREGAR: Generar Excel consolidado después de guardar el JSON
                self._generate_consolidated_excel(output_dir)
        
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
        """Adapta la matriz para el corrector estructural."""
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
                    # Crear una palabra placeholder si no hay palabras reales pero sí texto
                    adapted_word = {
                        'text': adapted_cell['cell_text'].strip(),
                        'xmin': col_idx * 100, 'xmax': (col_idx + 1) * 100,
                        'cx': col_idx * 100 + 50, 'cy': row_idx * 100 + 50
                    }
                    adapted_cell['words'].append(adapted_word)
                adapted_row.append(adapted_cell)
            adapted_matrix.append(adapted_row)
        return adapted_matrix

    def _save_simplified_matrix(self, matrix_data, header_elements, base_name, output_dir, suffix, payload=None):
        if not matrix_data:
            return

        headers = [h.get("text_raw", "") for h in header_elements]
        if suffix == "math_max_matrix" and payload:
            semantic_math_types = payload.get('outputs', {}).get('semantic_math_types')
            if semantic_math_types:
                semantic_types = semantic_math_types
            else:
                semantic_types = [h.get("semantic_type", "descriptivo") for h in header_elements]
        else:
            semantic_types = [h.get("semantic_type", "descriptivo") for h in header_elements]

        if matrix_data and matrix_data[0] and isinstance(matrix_data[0][0], dict):
            matrix_texts = [[cell.get("cell_text", "") for cell in row] for row in matrix_data]
        else:
            matrix_texts = matrix_data

        if suffix == "math_max_matrix":
            simplified_dict = {
                "headers": headers,
                "semantic_types": semantic_types,
                "semantic_format": "math_assigned",
                "matrix": matrix_texts
            }
        else:
            simplified_dict = {
                "headers": headers,
                "semantic_types": semantic_types,
                "matrix": matrix_texts
            }

        output_filename = f"{base_name}_{suffix}.json"
        output_path = os.path.join(output_dir, output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(simplified_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Matriz simplificada de depuración ({suffix}) guardada en: {output_path}")
        except Exception as e:
            logger.error(f"Error guardando la matriz simplificada de depuración en {output_path}: {e}")

    def _generate_consolidated_excel(self, output_dir: str):
        """
        Genera el archivo Excel consolidado con todos los math_max_matrix encontrados.
        """
        try:
            # Buscar todos los archivos math_max_matrix.json en el directorio
            pattern = os.path.join(output_dir, "*_math_max_matrix.json")
            math_max_files = glob.glob(pattern)
            
            if not math_max_files:
                logger.debug("No se encontraron archivos math_max_matrix.json para consolidar")
                return
            
            matrices_data = []
            
            for file_path in math_max_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        math_max_data = json.load(f)
                    
                    # Extraer document_id del nombre del archivo
                    filename = os.path.basename(file_path)
                    document_id = filename.replace("_math_max_matrix.json", "")
                    math_max_data["document_id"] = document_id
                    
                    matrices_data.append(math_max_data)
                    
                except Exception as e:
                    logger.error(f"Error leyendo {file_path}: {e}")
                    continue
            
            if matrices_data:
                excel_path = self.excel_handler.save_math_max_matrices(
                    matrices_data=matrices_data,
                    output_dir=output_dir,
                    file_name="math_max_resultados_finales.xlsx"
                )
                    
        except Exception as e:
            logger.error(f"Error en generación de Excel consolidado: {e}", exc_info=True)