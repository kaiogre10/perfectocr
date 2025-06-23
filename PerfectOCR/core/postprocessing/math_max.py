import logging
import re
from typing import List, Dict, Tuple, Optional, Any
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

        # --- Tolerancias configurables (se cargan desde YAML) ---
        #  • row_relative_tolerance → margen relativo por fila (0 .5 % por defecto)
        #  • total_mtl_abs_tolerance → margen absoluto para la suma de mtl (5 unidades por defecto)
        self.row_relative_tolerance = self.config.get('row_relative_tolerance', 0.005)   # 0.5 %
        self.total_mtl_abs_tolerance = self.config.get('total_mtl_abs_tolerance', 5.0)   # 5 unidades

        logger.info(
            "MatrixSolver inicializado con tolerancias: "
            f"fila ±{self.row_relative_tolerance*100:.2f}%  |  "
            f"total mtl ±{self.total_mtl_abs_tolerance} unidades."
        )

    def _str_to_float(self, s: str) -> Optional[float]:
        if not isinstance(s, str) or not s.strip(): return None
        cleaned = re.sub(r'[^\d.]', '', s)
        try:
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    def solve(self, matrix: List[List[str]], semantic_types: List[str], quarantined_data: List[Dict], document_totals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Punto de entrada principal.
        """
        logger.info(f"Resolviendo matriz con dimensiones {len(matrix)}x{len(matrix[0])}. Primeras 2 filas: {matrix[:2]}")
        quant_indices = [i for i, s_type in enumerate(semantic_types) if s_type == 'cuantitativo']

        if len(quant_indices) < 3:
            logger.warning("No hay suficientes columnas cuantitativas (se necesitan 3) para la resolución matricial.")
            return {
                "matrix": matrix,
                "semantic_math_types": semantic_types
            }

        hypotheses = permutations(quant_indices, 3)
        best_solution = None
        min_inconsistencies = float('inf')

        for h in hypotheses:
            c_idx, pu_idx, mtl_idx = h
            column_map = {'c': c_idx, 'pu': pu_idx, 'mtl': mtl_idx}

            # 1. Preparar los datos según la hipótesis actual (incluyendo cuarentena)
            quarantine_lookup = {
                (item["original_row"], item["original_col"]): self._str_to_float(item.get("value"))
                for item in quarantined_data
                if self._str_to_float(item.get("value")) is not None
            }
            ticket_data = self._prepare_data_for_hypothesis(matrix, column_map, quarantine_lookup)
            
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
            
            # --- FASE DE CORRECCIÓN ---
            # Se aplica la corrección aritmética a las filas inconsistentes
            # usando la hipótesis ganadora.
            final_data = self._correct_inconsistencies(best_solution['data'])
            
            final_matrix = self._reconstruct_matrix(matrix, final_data, best_solution['map'])

            # --- Generar semantic_math_types ---
            column_map = best_solution['map']
            semantic_math_types = []
            for idx, sem_type in enumerate(semantic_types):
                math_var = None
                for var, col_idx in column_map.items():
                    if col_idx == idx:
                        math_var = var
                        break
                if sem_type == "cuantitativo" and math_var:
                    semantic_math_types.append(f"{sem_type}, {math_var}")
                else:
                    semantic_math_types.append(sem_type)

            return {
                "matrix": final_matrix,
                "semantic_math_types": semantic_math_types
            }
        
        logger.error("No se pudo encontrar una solución aritméticamente consistente.")
        return {
            "matrix": matrix,
            "semantic_math_types": semantic_types
        }

    def _prepare_data_for_hypothesis(
        self,
        matrix: List[List[str]],
        column_map: Dict,
        quarantine_lookup: Dict[Tuple[int, int], float]
    ) -> List[Dict]:
        """Convierte la matriz de strings a una lista de diccionarios para el solver."""
        prepared_data = []
        for row_idx, row in enumerate(matrix):
            data_row = {'original_row': row_idx}
            for name, col_idx in column_map.items():
                value = self._str_to_float(row[col_idx])
                # Si la celda está vacía o inválida, probar primero valores en cuarentena
                if value is None:
                    value = quarantine_lookup.get((row_idx, col_idx))
                data_row[name] = value
            prepared_data.append(data_row)
        return prepared_data
    
    def _solve_with_logic(self, data: List[Dict], totals: Optional[Dict]) -> Tuple[List[Dict], List[str]]:
        """Lógica de resolución matricial."""
        warnings = []
        # Primera pasada: completar directos
        for row in data:
            c, pu, mtl = row.get('c'), row.get('pu'), row.get('mtl')
            if c is not None and pu is not None and mtl is None: row['mtl'] = c * pu
            elif c is not None and mtl is not None and pu is None and abs(c) > 1e-9: row['pu'] = mtl / c
            elif pu is not None and mtl is not None and c is None and abs(pu) > 1e-9: row['c'] = mtl / pu
        
        # Pasadas con totales (si están disponibles)
        if totals:
            total_mtl = totals.get('total_mtl')
            if total_mtl:
                known_mtl = sum(r['mtl'] for r in data if r.get('mtl') is not None)
                missing_rows = [r for r in data if r.get('mtl') is None]
                if len(missing_rows) == 1:
                    missing_rows[0]['mtl'] = total_mtl - known_mtl
                    self._solve_with_logic([missing_rows[0]], None) # Re-resolver fila

        # --- Verificación de consistencia fila por fila ---
        for i, row in enumerate(data):
            c, pu, mtl = row.get('c'), row.get('pu'), row.get('mtl')
            if all(v is not None for v in [c, pu, mtl]):
                diff = abs(c * pu - mtl)
                if diff > self.row_relative_tolerance * mtl:
                    warnings.append(
                        f"Inconsistencia Fila {i}: {c} * {pu} = {c*pu} "
                        f"(desviación {diff}) != {mtl}"
                    )

        # --- Verificación de totales (tolerancia absoluta) ---
        if totals and totals.get("total_mtl") is not None:
            calc_total_mtl = sum(r['mtl'] for r in data if r.get('mtl') is not None)
            total_mtl = totals["total_mtl"]
            if abs(calc_total_mtl - total_mtl) > self.total_mtl_abs_tolerance:
                warnings.append(
                    f"Diferencia en total mtl: calculado {calc_total_mtl:.2f} "
                    f"vs documento {total_mtl:.2f}"
                )
        
        return data, warnings
    
    def _correct_inconsistencies(self, solved_data: List[Dict]) -> List[Dict]:
        """
        Recorre los datos resueltos y corrige las filas que son aritméticamente
        inconsistentes. La estrategia de corrección es asumir que 'c' y 'pu'
        son correctos y recalcular 'mtl'.
        """
        for row in solved_data:
            c, pu, mtl = row.get('c'), row.get('pu'), row.get('mtl')

            if all(v is not None for v in [c, pu, mtl]):
                expected_mtl = c * pu
                # Si mtl es muy pequeño, la tolerancia relativa puede fallar. Usamos también una tolerancia absoluta pequeña.
                is_inconsistent = abs(expected_mtl - mtl) > max(self.row_relative_tolerance * abs(mtl), 0.01)

                if is_inconsistent:
                    logger.warning(
                        f"Corrigiendo Fila {row['original_row']}: "
                        f"c({c}) * pu({pu}) = {expected_mtl:.2f}. "
                        f"Valor mtl original ({mtl}) descartado."
                    )
                    row['mtl'] = expected_mtl  # Aplicar la corrección
        return solved_data

    def _reconstruct_matrix(self, original_matrix: List[List[str]], solved_data: List[Dict], column_map: Dict) -> List[List[str]]:
        """Re-inserta los datos resueltos en una copia de la matriz original."""
        reconstructed = [row[:] for row in original_matrix]
        for solved_row in solved_data:
            row_idx = solved_row['original_row']
            for name, col_idx in column_map.items():
                value = solved_row.get(name)
                if value is not None:
                    if abs(value - round(value)) < 1e-6:
                        reconstructed[row_idx][col_idx] = str(int(round(value)))
                    else:
                        reconstructed[row_idx][col_idx] = f"{value:.2f}"
        return reconstructed

# Ejemplo de uso eliminado para simplificar el módulo