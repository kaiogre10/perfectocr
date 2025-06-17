# PerfectOCR/utils/table_formatter.py
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TableFormatter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        # logger.debug("TableFormatter initialized.") # Descomentar si se necesita log de inicialización

    @staticmethod
    def format_as_markdown(headers: List[str], table_matrix_rows: List[List[Dict[str, Any]]]) -> str:
        """
        Formatea una estructura de tabla (lista de encabezados y lista de filas de celdas)
        en una cadena de texto con formato Markdown.

        Args:
            headers (List[str]): Una lista de strings representando los encabezados de la tabla.
            table_matrix_rows (List[List[Dict[str, Any]]]): Una lista de filas. Cada fila es una lista
                                                            de diccionarios de celda. Se espera que
                                                            cada diccionario de celda tenga una clave 'text'.
        Returns:
            str: Una cadena de texto representando la tabla en formato Markdown.
        """
        if not headers and not table_matrix_rows:
            return "(Tabla vacía)\n"

        markdown_lines = []

        # Preparar encabezados
        if headers:
            header_line = "| " + " | ".join(str(header) for header in headers) + " |"
            markdown_lines.append(header_line)
            # Línea separadora de Markdown para tabla
            separator_line = "| " + " | ".join("---" for _ in headers) + " |"
            markdown_lines.append(separator_line)
        elif table_matrix_rows and table_matrix_rows[0]: # Si no hay headers pero hay filas, crear separador basado en num de cols
            num_cols = len(table_matrix_rows[0])
            separator_line = "| " + " | ".join("---" for _ in range(num_cols)) + " |"
            markdown_lines.append(separator_line)


        # Preparar filas de datos
        for row_cells in table_matrix_rows:
            if not isinstance(row_cells, list):
                logger.warning(f"Se esperaba una lista de celdas para una fila, se obtuvo: {type(row_cells)}. Fila omitida: {row_cells}")
                continue

            # Extraer el texto de cada celda, manejando el caso de que 'text' no exista o la celda no sea un dict
            cell_texts = []
            for cell_data in row_cells:
                if isinstance(cell_data, dict):
                    text = cell_data.get('text', '') 
                elif isinstance(cell_data, str): # Si la celda es solo un string (menos común para la estructura actual)
                    text = cell_data
                else: # Si la celda no es ni dict ni str
                    text = str(cell_data) # Convertir a string como fallback
                
                # Escapar pipes dentro del texto de la celda para que no rompan la tabla Markdown
                escaped_text = text.replace("|", "\\|") if text else ""
                cell_texts.append(escaped_text)
            
            row_line = "| " + " | ".join(cell_texts) + " |"
            markdown_lines.append(row_line)

        return "\n".join(markdown_lines) + "\n"

    # Aquí podrían ir otros métodos de formateo de tablas en el futuro (ej. a HTML, CSV, etc.)