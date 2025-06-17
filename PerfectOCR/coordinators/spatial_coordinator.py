# PerfectOCR/coordinators/spatial_analyzer_coordinator.py
import copy
import re
import logging
import numpy as np
import hashlib
import os
from typing import Dict, Any, Optional, List, Tuple
from scipy.ndimage import label
from scipy.signal import find_peaks
from utils.geometric import get_polygon_bounds
from core.spatial_analysis.density_calculator import calculate_density_map 
from core.spatial_analysis.gradient_calculator import detect_noise_regions_combined, detect_noise_regions_intelligent

logger = logging.getLogger(__name__)

class SpatialAnalyzerCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.specific_config = config # Esta es la sección 'spatial_analyzer'
        self.project_root = project_root
        self.binary_matrix = None
        self.density_map = None
        self.page_dimensions = None
        self.table_region_stats_from_validation = None
        self.spatial_matrix = None
        self._density_cache = {}
        self._gradient_cache = {}
        self.max_spatial_workers = min(2, os.cpu_count())  # Máximo 2 hilos para spatial

        # Configuración para corroboración y análisis de bandas
        reassignment_config = self.specific_config.get('geo_spatial_corroboration', {})
        self.band_analysis_config = {
            'vertical_padding': int(reassignment_config.get('band_vertical_padding', 2)),
            'min_valley_prominence': float(reassignment_config.get('min_valley_prominence', 0.3))
        }
        self.quantitative_whitelist: set = set(reassignment_config.get('quantitative_whitelist', ['N/A', 'S/D', '-', '', 'n/a', 'N.A.', '$']))

        # Cargar configuración de ventana de densidad
        default_window_size_from_yaml = 23
        self.density_map_window_size = int(self.specific_config.get('density_map_window_size', default_window_size_from_yaml))

        if self.density_map_window_size <= 0 or self.density_map_window_size % 2 == 0:
            logger.warning(
                f"density_map_window_size ({self.density_map_window_size}) en config no es válido "
                f"(debe ser impar y >0). Usando valor por defecto: {default_window_size_from_yaml}."
            )
            self.density_map_window_size = default_window_size_from_yaml
                
    def analyze_image(self, binary_image: np.ndarray) -> Dict[str, Any]:
        if binary_image is None:
            return {"status": "error", "message": "Input image was None", "image_dims": None}

        h, w = binary_image.shape[:2]
        self.page_dimensions = {"width": w, "height": h}
        logger.info(f"SpatialAnalyzer iniciando análisis para imagen de {w}x{h}.")

        self.binary_matrix = binary_image.copy()
        
        # Normalizar imagen para el cálculo de densidad
        mean_pixel_value = np.mean(binary_image)
        if mean_pixel_value > 127: 
            normalized_image = (binary_image == 0).astype(np.float32)
        else: 
            normalized_image = (binary_image == 255).astype(np.float32)
        
        # Calcular y almacenar el mapa de densidad
        density_window_size = self.specific_config.get('density_map_window_size', 23)
        self.density_map = calculate_density_map(normalized_image, density_window_size)

        logger.info("Análisis de densidad completado por SpatialAnalyzerCoordinator.")
        return {"status": "success", "message": "Análisis de densidad completado", "image_dims": self.page_dimensions}

    def _find_column_boundaries_by_spaces(self, band_density: np.ndarray, num_columns: int, x_min_table: int) -> List[int]:
        """
        Encuentra los límites de columna usando los H-1 espacios más grandes por fila de la banda,
        promediando las posiciones de corte para obtener los límites globales de columna.
        Solo considera espacios entre grupos de tinta, nunca en los bordes.
        """
        H = num_columns
        cortes_por_fila = []
        alto, ancho = band_density.shape
        for fila in band_density:
            # Considerar como tinta los valores altos (asume binaria: 1=tinta, 0=fondo)
            if np.all(fila == 0) or np.all(fila == 1):
                continue
            indices_tinta = np.where(fila == 1)[0]
            if len(indices_tinta) < 2:
                continue
            # Solo considerar espacios entre grupos de tinta internos
            espacios = []
            for i in range(len(indices_tinta) - 1):
                x1, x2 = indices_tinta[i], indices_tinta[i+1]
                # Solo considerar si ambos no son el primer ni el último píxel de la fila
                if x1 == 0 or x2 == ancho - 1:
                    continue
                espacio = x2 - x1 - 1
                if espacio > 0:
                    espacios.append((espacio, x1, x2))
            if len(espacios) < H-1:
                continue
            espacios_mayores = sorted(espacios, reverse=True)[:H-1]
            espacios_mayores = sorted(espacios_mayores, key=lambda x: x[1])
            cortes = [ (x1 + x2) // 2 for _, x1, x2 in espacios_mayores ]
            cortes_por_fila.append(cortes)
        if not cortes_por_fila or len(cortes_por_fila[0]) != H-1:
            return []
        cortes_por_indice = list(zip(*cortes_por_fila))
        limites = [int(np.mean(cortes)) + x_min_table for cortes in cortes_por_indice]
        return sorted(limites)

    def _analyze_band(self, y_min: int, y_max: int, num_columns: int, x_min_table: int, x_max_table: int) -> List[int]:
        """
        Analiza una banda horizontal del mapa de densidad para encontrar límites de columna.
        """
        if self.density_map is None or num_columns <= 1:
            return []

        pad = self.band_analysis_config['vertical_padding']
        y_min_p = max(0, y_min - pad)
        y_max_p = min(self.page_dimensions['height'], y_max + pad)
        band_density = self.density_map[y_min_p:y_max_p, x_min_table:x_max_table]
        if band_density.size == 0:
            return []

        # Tomar la submatriz binaria original (tinta negra=0). Convertir a 1=tinta, 0=fondo
        band_src = self.binary_matrix[y_min_p:y_max_p, x_min_table:x_max_table]
        band_bin = (band_src == 0).astype(np.uint8)
        # Usar el nuevo método espacial puro
        limites = self._find_column_boundaries_by_spaces(band_bin, num_columns, x_min_table)
        if limites:
            return limites
        # Fallback: método anterior
        horizontal_density_profile = np.sum(band_density, axis=0)
        inverted_profile = np.max(horizontal_density_profile) - horizontal_density_profile
        band_width = x_max_table - x_min_table
        min_dist = band_width / (num_columns * 2)
        valleys_relative, props = find_peaks(
            inverted_profile,
            distance=min_dist,
            prominence=(self.band_analysis_config['min_valley_prominence'] * np.max(inverted_profile) if np.max(inverted_profile) > 0 else None)
        )
        if len(valleys_relative) > num_columns - 1:
            prominences = props['prominences']
            most_prominent_indices = np.argsort(prominences)[::-1][:num_columns - 1]
            valleys_relative = valleys_relative[most_prominent_indices]
        valleys_absolute = sorted([x + x_min_table for x in valleys_relative])
        return valleys_absolute

    def _build_spatial_matrix(self, row_bands: List[Tuple[int, int]], num_columns: int, x_min_table: int, x_max_table: int) -> List[List[Dict]]:
        """
        Construye T_spatial usando las bandas Y de cada fila.
        """
        self.spatial_matrix = []
        for y_min, y_max in row_bands:
            column_boundaries = self._analyze_band(y_min, y_max, num_columns, x_min_table, x_max_table)
            
            # Si no se encontraron suficientes límites, la fila espacial no es válida
            if len(column_boundaries) < num_columns - 1:
                self.spatial_matrix.append([]) # Fila vacía para indicar fallo
                continue

            row_regions: List[Dict] = []
            current_x = x_min_table
            for boundary in column_boundaries:
                row_regions.append({'region': [current_x, y_min, boundary, y_max]})
                current_x = boundary
            # Añadir la última columna
            row_regions.append({'region': [current_x, y_min, x_max_table, y_max]})
            
            self.spatial_matrix.append(row_regions)
        
        return self.spatial_matrix

    def _find_row_boundaries(self, table_roi_binary: np.ndarray, num_rows: int) -> List[int]:
        """
        Encuentra los N-1 límites de fila usando los valles de densidad más prominentes.
        """
        if table_roi_binary.size == 0 or num_rows <= 1:
            return []

        h, w = table_roi_binary.shape

        # Proyección horizontal (promedio de píxeles de tinta por fila)
        # Asumimos que tinta es 0, fondo es 255. Queremos filas con mucho fondo.
        if np.max(table_roi_binary) > 1:
            # Convertimos a 0-1, donde 1 es fondo/blanco.
            projection_friendly_img = (255 - table_roi_binary).astype(np.float32) / 255.0
        else: 
            # Asume 0 es tinta, 1 es fondo.
            projection_friendly_img = (1 - table_roi_binary).astype(np.float32)
        
        horizontal_profile = np.mean(projection_friendly_img, axis=1)

        # Para encontrar valles de tinta, buscamos picos de blancura.
        avg_row_height = h / num_rows if num_rows > 0 else h
        min_dist = max(1, int(avg_row_height * 0.4)) # Distancia mínima para evitar picos en la misma línea de texto

        # La prominencia es clave. Un valor dinámico podría ser mejor.
        # Por ahora, usamos un valor fijo o un % del rango del perfil.
        min_prominence = (np.max(horizontal_profile) - np.min(horizontal_profile)) * 0.1

        valleys, properties = find_peaks(horizontal_profile, 
                                         distance=min_dist,
                                         prominence=min_prominence) 

        if len(valleys) < num_rows - 1:
            logger.warning(f"Se esperaban {num_rows - 1} separadores de fila, pero solo se encontraron {len(valleys)}. "
                           "La estructura de filas puede ser imprecisa. Se usarán los encontrados.")
        
        if len(valleys) > num_rows - 1:
            prominences = properties['prominences']
            most_prominent_indices = np.argsort(prominences)[::-1][:num_rows - 1]
            valleys = valleys[most_prominent_indices]

        return sorted(list(valleys))

    def _get_bbox_intersection_area(self, boxA: List[int], boxB: List[int]) -> float:
        """Calcula el área de intersección de dos bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # Asegurarse de que el área no sea negativa si no hay solapamiento
        interArea = max(0, xB - xA) * max(0, yB - yA)
        return float(interArea)

    def structure_table_spatially(
        self,
        all_words: List[Dict],
        num_rows: int,
        num_columns: int,
        table_bbox: Tuple[int, int, int, int]
    ) -> List[List[Dict]]:
        """
        (NUEVO) Construye la matriz final de la tabla usando una cuadrícula espacial como referencia absoluta.
        El texto del OCR se recorta y se asigna a las celdas de la cuadrícula.
        """
        logger.info("Iniciando estructuración espacial de la tabla (nuevo método).")
        if self.binary_matrix is None:
            logger.error("No se puede estructurar la tabla sin una matriz binaria. Abortando.")
            return [[{'words': [], 'cell_text': ''} for _ in range(num_columns)] for _ in range(num_rows)]

        x_min, y_min, x_max, y_max = table_bbox
        table_roi_binary = self.binary_matrix[y_min:y_max, x_min:x_max]

        row_boundaries = self._find_row_boundaries(table_roi_binary, num_rows)
        column_boundaries_relative = self._analyze_band(y_min, y_max, num_columns, x_min, x_max)
        column_boundaries = [b - x_min for b in column_boundaries_relative] # Convertir a relativas a la ROI

        if len(row_boundaries) < num_rows - 1:
            logger.error(f"No se pudieron determinar suficientes límites de fila ({len(row_boundaries)} de {num_rows - 1}). Abortando fusión.")
            return [[{'words': [], 'cell_text': ''} for _ in range(num_columns)] for _ in range(num_rows)]
        if len(column_boundaries) < num_columns - 1:
            logger.error(f"No se pudieron determinar suficientes límites de columna ({len(column_boundaries)} de {num_columns - 1}). Abortando fusión.")
            return [[{'words': [], 'cell_text': ''} for _ in range(num_columns)] for _ in range(num_rows)]

        # Paso 2: Construir la cuadrícula de celdas (bboxes)
        spatial_grid = []
        y_starts = [0] + row_boundaries
        y_ends = row_boundaries + [table_roi_binary.shape[0]]

        for r_idx, (y_start, y_end) in enumerate(zip(y_starts, y_ends)):
            row_cells = []
            x_starts = [0] + column_boundaries
            x_ends = column_boundaries + [table_roi_binary.shape[1]]
            for c_idx, (x_start, x_end) in enumerate(zip(x_starts, x_ends)):
                cell_bbox_abs = [x_start + x_min, y_start + y_min, x_end + x_min, y_end + y_min]
                row_cells.append({'bbox': cell_bbox_abs, 'words': [], 'cell_text': ''})
            spatial_grid.append(row_cells)

        # Paso 3: Asignar palabras a la cuadrícula (el corazón de la fusión)
        logger.info("Asignando palabras del OCR a la cuadrícula espacial.")
        
        match_threshold = self.specific_config.get('geo_spatial_corroboration', {}).get('spatial_text_match_threshold', 0.7)

        for word in all_words:
            try:
                word_bbox = [int(word['xmin']), int(word['ymin']), int(word['xmax']), int(word['ymax'])]
                word_area = (word_bbox[2] - word_bbox[0]) * (word_bbox[3] - word_bbox[1])
                if word_area == 0:
                    continue
            except (KeyError, ValueError) as e:
                logger.warning(f"Palabra omitida por no tener bbox válido: {word.get('text_raw', 'N/A')}. Error: {e}")
                continue

            best_match = {'r': -1, 'c': -1, 'overlap_ratio': -1.0}

            for r_idx, row in enumerate(spatial_grid):
                for c_idx, cell in enumerate(row):
                    cell_bbox = cell['bbox']
                    intersection_area = self._get_bbox_intersection_area(word_bbox, cell_bbox)
                    
                    overlap_ratio = intersection_area / word_area if word_area > 0 else 0

                    if overlap_ratio > best_match['overlap_ratio']:
                        best_match = {'r': r_idx, 'c': c_idx, 'overlap_ratio': overlap_ratio}
            
            if best_match['overlap_ratio'] > match_threshold:
                r, c = best_match['r'], best_match['c']
                spatial_grid[r][c]['words'].append(word)

        # Paso 4: Construir la matriz final a partir de las palabras asignadas
        final_matrix = [[cell.copy() for cell in row] for row in spatial_grid]
        for r_idx, row in enumerate(final_matrix):
            for c_idx, cell in enumerate(row):
                # Ordenar palabras por X y luego concatenar
                sorted_words = sorted(cell['words'], key=lambda w: w.get('xmin', 0))
                cell['cell_text'] = " ".join([w.get('text_raw', '') for w in sorted_words]).strip()
                del cell['bbox']
                # Mantener 'words' puede ser útil para depuración, se puede quitar si es necesario
        
        logger.info("Fusión de texto en cuadrícula espacial completada.")
        return final_matrix

    def corroborate_and_refine_table(
        self,
        geometric_matrix: List[List[Dict]],
        header_info: List[Dict],
        row_bands: List[Tuple[int, int]],
        x_min_table: int,
        x_max_table: int
    ) -> List[List[Dict]]:
        """
        DEPRECATED: Este método ha sido reemplazado por `structure_table_spatially`.
        La lógica anterior se basaba en reasignación por centroides, que es menos robusta.
        Se mantiene temporalmente por compatibilidad hasta que se refactorice el llamador.
        """
        logger.warning("Llamada a método deprecado 'corroborate_and_refine_table'. "
                       "La nueva arquitectura usa 'structure_table_spatially'. Se devuelve la matriz geométrica sin cambios.")
        return geometric_matrix

    def _final_cleanup(self, matrix: List[List[Dict]]) -> List[List[Dict]]:
        """Ejecuta limpiezas finales que antes estaban en MaxValidation."""
        # Ejemplo: eliminar filas completamente vacías
        non_empty_matrix = [row for row in matrix if any(cell.get('cell_text') for cell in row)]
        logger.info(f"Limpieza final: eliminadas {len(matrix) - len(non_empty_matrix)} filas vacías.")
        return non_empty_matrix

    def detect_noise_regions(self) -> List[List[int]]:
        noise_params = self.specific_config.get('ocr_noise_filtering', {})
        method = noise_params.get('method', 'original')

        if not noise_params.get('enabled', False):
            return []

        if method == 'original':
            return self.detect_noise_regions_original()
        elif method == 'gradient':
            return self.detect_noise_regions_gradient_only()
        elif method == 'enhanced':
            return self.detect_noise_regions_enhanced()
        elif method == 'intelligent':
            return self.detect_noise_regions_intelligent()
        else:
            logger.warning(f"Método de detección de ruido '{method}' no reconocido. Usando 'original'.")
            return self.detect_noise_regions_original()

    def detect_noise_regions_original(self) -> List[List[int]]:
        """
        Detecta regiones de ruido (líneas) basándose en anomalías de densidad vertical.
        """
        if self.binary_matrix is None:
            logger.error("No se puede detectar ruido sin una matriz binaria.")
            return []

        noise_params = self.specific_config.get('ocr_noise_filtering', {})
        intensity_factor = noise_params.get('line_intensity_factor', 3.0)
        min_line_height = noise_params.get('min_line_height_for_noise_px', 8)

        # Proyección vertical (suma de píxeles por fila)
        vertical_projection = np.mean(self.binary_matrix, axis=1)
        
        # Normalizar si la imagen es 0-255
        if np.max(vertical_projection) > 1:
            vertical_projection /= 255.0

        # Umbral para detectar anomalías (líneas mucho más densas que el promedio)
        mean_density = np.mean(vertical_projection[vertical_projection > 0])
        std_density = np.std(vertical_projection[vertical_projection > 0])
        threshold = mean_density + intensity_factor * std_density
        
        # Identificar filas que superan el umbral
        anomaly_rows = (vertical_projection > threshold).astype(int)

        # Agrupar regiones consecutivas
        labeled_array, num_features = label(anomaly_rows)
        if num_features == 0:
            return []

        # Extraer bounding boxes y filtrar por altura
        noise_regions = []
        for i in range(1, num_features + 1):
            rows = np.where(labeled_array == i)[0]
            if not rows.any(): continue
            
            y_min, y_max = rows.min(), rows.max()
            height = y_max - y_min + 1

            if height <= min_line_height:
                expansion_factor = noise_params.get('vertical_expansion_factor', 2.5) # Default de 2.5 si no está
                expansion_pixels = int(height * expansion_factor)
                expanded_y_min = max(0, y_min - expansion_pixels)
                expanded_y_max = min(self.page_dimensions['height'], y_max + expansion_pixels)
                
                logger.debug(f"Región de ruido en Y:[{y_min}, {y_max}] (altura: {height}px) expandida a Y:[{expanded_y_min}, {expanded_y_max}] con factor {expansion_factor}")
                noise_regions.append([0, expanded_y_min, self.page_dimensions['width'], expanded_y_max])
                
        logger.info(f"Método original detectó {len(noise_regions)} regiones de ruido candidatas.")
        
        merged_regions = self._merge_close_regions(noise_regions, max_gap=5)
        
        logger.info(f"Después de fusionar, quedaron {len(merged_regions)} regiones de ruido.")
        
        return merged_regions

    def _merge_close_regions(self, regions: List[List[int]], max_gap: int) -> List[List[int]]:
        """
        Fusiona regiones (bboxes) que están verticalmente cerca.
        """
        if not regions:
            return []

        # Ordenar regiones por su coordenada Y inicial
        regions.sort(key=lambda r: r[1])

        merged = []
        current_region = regions[0]

        for i in range(1, len(regions)):
            next_region = regions[i]
            # Si el espacio vertical es menor o igual al máximo permitido
            if next_region[1] - current_region[3] <= max_gap:
                # Fusionar: tomar el y_max más grande
                current_region[3] = max(current_region[3], next_region[3])
            else:
                merged.append(current_region)
                current_region = next_region
        
        merged.append(current_region)
        return merged

    def detect_noise_regions_gradient_only(self) -> List[List[int]]:
        """
        Detecta líneas de ruido utilizando solo el método de gradientes.
        """
        if self.binary_matrix is None:
            logger.error("No se puede detectar ruido con gradientes sin una matriz binaria.")
            return []
        
        noise_params = self.specific_config.get('ocr_noise_filtering', {})

        # Llamar a la función del módulo de gradientes
        # Se asume que esta función devuelve bboxes [xmin, ymin, xmax, ymax]
        regions = detect_noise_regions_combined(
            binary_image=self.binary_matrix,
            use_density=False,  # Solo gradiente
            use_gradient=True,
            gradient_threshold_factor=noise_params.get('gradient_threshold_factor', 2.0),
            min_line_length_ratio=noise_params.get('min_line_length_ratio', 0.2),
            max_line_height_px=noise_params.get('max_line_height_px', 8),
            smoothing_sigma=noise_params.get('smoothing_sigma', 1.0)
        )
        
        logger.info(f"Método de gradiente detectó {len(regions)} regiones de ruido.")
        return regions

    def detect_noise_regions_enhanced(self) -> List[List[int]]:
        """
        Implementación mejorada que combina densidad y gradientes.
        """
        if self.binary_matrix is None:
            logger.error("No se puede detectar ruido (enhanced) sin una matriz binaria.")
            return []

        noise_params = self.specific_config.get('ocr_noise_filtering', {})

        # Llamar a la función combinada del módulo de gradientes
        regions = detect_noise_regions_combined(
            binary_image=self.binary_matrix,
            use_density=True,
            use_gradient=True,
            line_intensity_factor=noise_params.get('line_intensity_factor', 3.0),
            min_line_height_for_noise_px=noise_params.get('min_line_height_for_noise_px', 8),
            gradient_threshold_factor=noise_params.get('gradient_threshold_factor', 2.0),
            min_line_length_ratio=noise_params.get('min_line_length_ratio', 0.2),
            max_line_height_px=noise_params.get('max_line_height_px', 8),
            smoothing_sigma=noise_params.get('smoothing_sigma', 1.0)
        )
        
        # Filtrado final basado en propiedades geométricas
        filtered_regions = self._filter_noise_regions(regions)
        
        logger.info(f"Método mejorado detectó {len(regions)} regiones candidatas, "
                    f"quedaron {len(filtered_regions)} después del filtrado.")
        
        return filtered_regions

    def _filter_noise_regions(self, regions: List[List[int]]) -> List[List[int]]:
        """
        Filtra una lista de regiones de ruido candidatas basándose en propiedades
        geométricas y de consistencia.
        """
        if not regions or self.binary_matrix is None:
            return []

        noise_params = self.specific_config.get('ocr_noise_filtering', {})
        page_width = self.page_dimensions['width']
        
        # Cargar parámetros de configuración
        min_width_ratio = noise_params.get('min_region_width_ratio', 0.1)
        max_height_px = noise_params.get('max_region_height_px', 15)
        min_density = noise_params.get('min_pixel_density', 0.3)
        min_consistency = noise_params.get('min_horizontal_consistency', 0.6)

        final_regions = []
        for xmin, ymin, xmax, ymax in regions:
            width = xmax - xmin
            height = ymax - ymin

            # 1. Filtrar por tamaño
            if width < page_width * min_width_ratio:
                logger.debug(f"Región de ruido descartada por ancho insuficiente: {width}px")
                continue
            if height > max_height_px:
                logger.debug(f"Región de ruido descartada por altura excesiva: {height}px")
                continue

            # 2. Filtrar por densidad de píxeles
            region_slice = self.binary_matrix[ymin:ymax, xmin:xmax]
            if region_slice.size == 0: continue
            
            pixel_density = np.mean(region_slice > 0)
            if pixel_density < min_density:
                logger.debug(f"Región de ruido descartada por baja densidad: {pixel_density:.2f}")
                continue

            # 3. Filtrar por consistencia horizontal (qué tan 'sólida' es la línea)
            consistency = self._check_horizontal_line_consistency(region_slice)
            if consistency < min_consistency:
                logger.debug(f"Región de ruido descartada por baja consistencia horizontal: {consistency:.2f}")
                continue
            
            final_regions.append([xmin, ymin, xmax, ymax])

        return final_regions

    def _check_horizontal_line_consistency(self, region_data: np.ndarray) -> float:
        """
        Calcula qué tan 'completa' es una línea horizontal.
        Devuelve un ratio de columnas que tienen al menos un píxel activado.
        """
        if region_data.size == 0:
            return 0.0
        
        # Proyección vertical dentro de la región
        cols_with_pixels = np.sum(region_data, axis=0)
        
        # Ratio de columnas no vacías
        num_cols_with_pixels = np.sum(cols_with_pixels > 0)
        total_cols = region_data.shape[1]
        
        return num_cols_with_pixels / total_cols if total_cols > 0 else 0.0

    def detect_noise_regions_intelligent(self) -> List[List[int]]:
        """
        Detecta líneas de ruido usando análisis inteligente de gradientes + Hessiana selectiva.
        Se adapta automáticamente a cualquier formato de documento.
        """
        if self.binary_matrix is None:
            logger.error("No se puede detectar ruido inteligente sin una matriz binaria.")
            return []
        
        noise_params = self.specific_config.get('ocr_noise_filtering', {})
        
        # Configuración para detección inteligente
        intelligent_config = {
            'min_line_length_ratio': noise_params.get('min_line_length_ratio', 0.2),
            'max_line_height_px': noise_params.get('max_line_height_px', 10),
            'artificiality_threshold': noise_params.get('artificiality_threshold', 0.6),
            'hessian_ambiguous_range': noise_params.get('hessian_ambiguous_range', [0.4, 0.8]),
            'hessian_edge_threshold': noise_params.get('hessian_edge_threshold', 0.15),
            'expansion_factor': noise_params.get('vertical_expansion_factor', 2.0)
        }
        
        # Llamar al detector inteligente
        regions = detect_noise_regions_intelligent(
            binary_image=self.binary_matrix,
            config=intelligent_config
        )
        
        logger.info(f"Método inteligente detectó {len(regions)} regiones de ruido adaptativas.")
        return regions

    def clear_data(self):
        """Limpia los datos para procesar un nuevo archivo."""
        self.binary_matrix = None
        self.density_map = None
        self.page_dimensions = None
        logger.debug("Datos de SpatialAnalyzerCoordinator limpiados.")

    def _get_image_hash(self, image: np.ndarray) -> str:
        """Genera hash rápido de la imagen para cache."""
        return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def analyze_spatial_structure_cached(self, spatial_image: np.ndarray, page_dimensions: Tuple[int, int]) -> Dict[str, Any]:
        """
        Análisis espacial con cache para evitar recálculos.
        """
        image_hash = self._get_image_hash(spatial_image)
        
        if image_hash in self._density_cache:
            logger.debug("Usando densidad desde cache")
            density_results = self._density_cache[image_hash]
        else:
            density_results = self._calculate_density_analysis(spatial_image)
            self._density_cache[image_hash] = density_results
        
        # Corregir: obtener gradient_results también
        if image_hash in self._gradient_cache:
            logger.debug("Usando gradientes desde cache")
            gradient_results = self._gradient_cache[image_hash]
        else:
            gradient_results = self._calculate_gradient_analysis(spatial_image)
            self._gradient_cache[image_hash] = gradient_results
        
        return self._combine_spatial_results(density_results, gradient_results)
    
    def _calculate_density_analysis(self, spatial_image: np.ndarray) -> Dict[str, Any]:
        """
        Calcula análisis de densidad para una imagen espacial.
        """
        from core.spatial_analysis.density_calculator import calculate_density_map
        
        # Normalizar imagen
        if np.max(spatial_image) > 127:
            normalized_image = (spatial_image == 0).astype(np.float32)
        else:
            normalized_image = (spatial_image == 255).astype(np.float32)
        
        density_map = calculate_density_map(normalized_image, self.density_map_window_size)
        
        return {
            'density_map': density_map,
            'normalized_image': normalized_image,
            'analysis_type': 'density'
        }
    
    def _calculate_gradient_analysis(self, spatial_image: np.ndarray) -> Dict[str, Any]:
        """
        Calcula análisis de gradientes para una imagen espacial.
        """
        from core.spatial_analysis.gradient_calculator import calculate_gradient_maps
        
        gradient_maps = calculate_gradient_maps(spatial_image, normalize_input=True)
        
        return {
            'gradient_maps': gradient_maps,
            'analysis_type': 'gradient'
        }
    
    def _combine_spatial_results(self, density_results: Dict[str, Any], gradient_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina resultados de análisis de densidad y gradientes.
        """
        return {
            'status': 'success',
            'density_analysis': density_results,
            'gradient_analysis': gradient_results,
            'combined_analysis': True
        }

    def analyze_spatial_structure_parallel(self, spatial_image: np.ndarray, page_dimensions: Tuple[int, int]) -> Dict[str, Any]:
        """
        Análisis espacial con paralelización de subtareas manteniendo calidad.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.max_spatial_workers) as executor:
            # Lanzar tareas en paralelo
            density_future = executor.submit(
                self._calculate_density_analysis, 
                spatial_image
            )
            
            gradient_future = executor.submit(
                self._calculate_gradient_analysis, 
                spatial_image
            )
            
            # Recoger resultados
            density_results = density_future.result()
            gradient_results = gradient_future.result()
            
            # Combinar resultados
            return self._combine_spatial_results(density_results, gradient_results)