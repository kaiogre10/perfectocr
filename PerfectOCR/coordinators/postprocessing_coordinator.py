# PerfectOCR/coordinators/postprocessing_coordinator.py
import logging
import os
from typing import Dict, Any, List
# Se importa solo el corrector semántico, como se ha solicitado.
from core.postprocessing.correctors import SemanticTableCorrector
# Se eliminan los imports de TextFormatter, ya que no se usará.

logger = logging.getLogger(__name__)

class PostprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str):
        """
        Inicializa el coordinador de post-procesamiento.
        Por ahora, se enfoca únicamente en la corrección estructural de la tabla.
        """
        self.config = config
        self.project_root = project_root

        semantic_corrector_cfg = self.config.get('semantic_table_correction', {})
        self.semantic_table_corrector = SemanticTableCorrector(config=semantic_corrector_cfg)
        
        # Se elimina la inicialización de TextCorrector y TextFormatter.

        logger.info("PostprocessingCoordinator inicializado para corrección de ESTRUCTURA de tabla.")

    def correct_table_structure(self, extraction_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica la corrección semántica a la matriz de una tabla extraída.
        
        Recibe el payload completo del TableExtractorCoordinator y devuelve
        un payload actualizado con la matriz corregida.
        """
        logger.info("Iniciando corrección semántica de la estructura de la tabla.")
        
        outputs = extraction_payload.get('outputs')
        if not outputs or 'table_matrix' not in outputs or 'header_elements' not in outputs:
            logger.warning("Payload no contiene 'outputs.table_matrix' o 'outputs.header_elements'. No se aplica corrección.")
            return extraction_payload

        # Extraer los datos necesarios del payload
        matrix_to_correct = outputs['table_matrix']
        headers = outputs['header_elements']

        # Delegar la corrección al módulo especializado
        corrected_matrix = self.semantic_table_corrector.correct_matrix(
            matrix=matrix_to_correct,
            header_elements=headers
        )
        
        # Crear una copia del payload para no modificar el original directamente
        corrected_payload = extraction_payload.copy()
        corrected_payload['outputs'] = outputs.copy()  # Asegurar que 'outputs' también sea una copia
        corrected_payload['outputs']['table_matrix'] = corrected_matrix
        
        # Actualizar el estado para reflejar que la corrección semántica fue aplicada
        corrected_payload['status'] = 'success_semantically_corrected'
        corrected_payload['message'] = 'Tabla estructurada y corregida semánticamente.'
        
        logger.info("Corrección semántica de la tabla completada.")
        return corrected_payload

    # El método correct_and_format ha sido eliminado para centrarse en la corrección estructural.