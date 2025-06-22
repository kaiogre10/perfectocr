import logging
import re
from typing import List, Dict, Tuple, Optional
from itertools import permutations

logger = logging.getLogger(__name__)

class MatrixSolver:
    """
    Resuelve una matriz de tickets con datos incompletos basándose en el
    planteamiento de resolución matricial. Su función principal es descubrir
    la asignación correcta de columnas (c, pu, mtl) que satisface la
    aritmética y luego completar los datos faltantes.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.tolerance = self.config.get('arithmetic_tolerance', 0.05)
        logger.info(f"MatrixSolver (Resolución Matricial) inicializado con tolerancia: {self.tolerance}")

    def _str_to_float(self, s: str) -> Optional[float]:
        if not isinstance(s, str) or not s.strip(): return None
        cleaned = re.sub(r'[^\d.]', '', s)
        try:
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    def solve(self, matrix: List[List[str]], semantic_types: List[str], quarantined_data: List[Dict], document_totals: Optional[Dict] = None) -> List[List[str]]:
        """
        Punto de entrada principal.
        """
        quant_indices = [i for i, s_type in enumerate(semantic_types) if s_type == 'cuantitativo']

        if len(quant_indices) < 3:
            logger.warning("No hay suficientes columnas cuantitativas (se necesitan 3) para la resolución matricial.")
            return matrix

        hypotheses = permutations(quant_indices, 3)
        best_solution = None
        min_inconsistencies = float('inf')

        for h in hypotheses:
            c_idx, pu_idx, mtl_idx = h
            column_map = {'c': c_idx, 'pu': pu_idx, 'mtl': mtl_idx}

            # 1. Preparar los datos según la hipótesis actual
            ticket_data = self._prepare_data_for_hypothesis(matrix, column_map)
            
            # (Opcional) Aplicar datos en cuarentena si ayuda a resolver
            # Por ahora, nos enfocamos en la resolución principal

            # 2. Resolver la matriz con la lógica principal
            solved_data, warnings = self._solve_with_logic(ticket_data, document_totals)
            
            # 3. Evaluar la calidad de la solución
            if len(warnings) < min_inconsistencies:
                min_inconsistencies = len(warnings)
                best_solution = {
                    'data': solved_data,
                    'map': column_map
                }
            
            if min_inconsistencies == 0:
                break # Se encontró una solución perfecta

        # 4. Reconstruir la matriz con la mejor solución encontrada
        if best_solution:
            logger.info(f"Mejor hipótesis encontrada: c={best_solution['map']['c']}, pu={best_solution['map']['pu']}, mtl={best_solution['map']['mtl']} con {min_inconsistencies} inconsistencias.")
            return self._reconstruct_matrix(matrix, best_solution['data'], best_solution['map'])
        
        logger.error("No se pudo encontrar una solución aritméticamente consistente.")
        return matrix

    def _prepare_data_for_hypothesis(self, matrix: List[List[str]], column_map: Dict) -> List[Dict]:
        """Convierte la matriz de strings a una lista de diccionarios para el solver."""
        prepared_data = []
        for row_idx, row in enumerate(matrix):
            data_row = {'original_row': row_idx}
            for name, col_idx in column_map.items():
                data_row[name] = self._str_to_float(row[col_idx])
            prepared_data.append(data_row)
        return prepared_data
    
    def _solve_with_logic(self, data: List[Dict], totals: Optional[Dict]) -> Tuple[List[Dict], List[str]]:
        """Lógica de resolución matricial."""
        warnings = []
        # Primera pasada: completar directos
        for row in data:
            c, pu, mtl = row.get('c'), row.get('pu'), row.get('mtl')
            if c is not None and pu is not None and mtl is None: row['mtl'] = c * pu
            elif c is not None and mtl is not None and pu is None and abs(c) > self.tolerance: row['pu'] = mtl / c
            elif pu is not None and mtl is not None and c is None and abs(pu) > self.tolerance: row['c'] = mtl / pu
        
        # Pasadas con totales (si están disponibles)
        if totals:
            total_mtl = totals.get('total_mtl')
            if total_mtl:
                known_mtl = sum(r['mtl'] for r in data if r.get('mtl') is not None)
                missing_rows = [r for r in data if r.get('mtl') is None]
                if len(missing_rows) == 1:
                    missing_rows[0]['mtl'] = total_mtl - known_mtl
                    self._solve_with_logic([missing_rows[0]], None) # Re-resolver fila

        # Verificación de consistencia
        for i, row in enumerate(data):
            c, pu, mtl = row.get('c'), row.get('pu'), row.get('mtl')
            if all(v is not None for v in [c, pu, mtl]):
                if abs(c * pu - mtl) > self.tolerance:
                    warnings.append(f"Inconsistencia Fila {i}: {c} * {pu} = {c*pu} != {mtl}")
        
        return data, warnings
    
    def _reconstruct_matrix(self, original_matrix: List[List[str]], solved_data: List[Dict], column_map: Dict) -> List[List[str]]:
        """Re-inserta los datos resueltos en una copia de la matriz original."""
        reconstructed = [row[:] for row in original_matrix]
        for solved_row in solved_data:
            row_idx = solved_row['original_row']
            for name, col_idx in column_map.items():
                value = solved_row.get(name)
                if value is not None:
                    if abs(value - round(value)) < self.tolerance:
                        reconstructed[row_idx][col_idx] = str(int(round(value)))
                    else:
                        reconstructed[row_idx][col_idx] = f"{value:.2f}"
        return reconstructed

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (pueden tener None para valores faltantes)
    example_data = [
        ["2", "55.50", None],
        [None, "30.00", "30.00"],
        [None, "18.90", "56.70"],
        ["1", "99.50", "99.50"],
    ]
    
    solver = MatrixSolver()
    solved_data, warnings = solver.solve(
        matrix=example_data,
        semantic_types=['cuantitativo', 'cuantitativo', 'cuantitativo'],
        quarantined_data=[],
        document_totals={'total_mtl': 736.48, 'total_c': 16}
    )
    
    print("Datos completados:")
    for row in solved_data:
        print(row)
    
    print("\nAdvertencias:")
    for warning in warnings:
        print(warning)