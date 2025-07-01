import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class GeoCosmicalStructurer:
    def __init__(self, encoder_path: str = "codificador.csv"):
        """
        Inicializa el segmentador con el codificador de caracteres.
        Args:
            encoder_path: Ruta al archivo CSV con el mapeo carácter-valor.
        """
        self.encoder = self._load_encoder(encoder_path)
        self.window_size = 3  # Ventana impar para desplazamiento
        self.cohesion_threshold = 0.7  # Umbral de similitud de coseno

    def _load_encoder(self, path: str) -> Dict[str, float]:
        """Carga el codificador desde el CSV y devuelve un diccionario."""
        encoder = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                char, val = line.strip().split(',')
                encoder[char] = float(val)
        return encoder

    def _get_z(self, char: str) -> float:
        """Devuelve el valor z para un carácter, con manejo de fallos."""
        return self.encoder.get(char, 0.0)

    def _calculate_cosine(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Calcula la similitud de coseno entre dos vectores (x, z)."""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        return dot_product / (mag_v1 * mag_v2) if (mag_v1 * mag_v2) != 0 else 0.0

    def segment_line(self, line_elements: List[Dict[str, Any]], H: int) -> List[List[Dict[str, Any]]]:
        """
        Segmenta una línea de texto en H columnas basándose en (x, z).
        Args:
            line_elements: Lista de palabras/dicts con 'text', 'xmin', 'xmax', 'cx'.
            H: Número de columnas esperadas (de los headers).
        Returns:
            Lista de H listas (columnas), cada una con los elementos asignados.
        """
        if not line_elements or H == 0:
            return [[] for _ in range(H)]

        # 1. Construir vectores (x, z) para cada carácter
        vectors = []
        for elem in line_elements:
            text = elem.get('text', '')
            x_center = elem.get('cx', (elem.get('xmin', 0) + elem.get('xmax', 0)) / 2)
            for char in text:
                z = self._get_z(char)
                vectors.append((x_center, z))  # Usamos el centro x del elemento para simplificar

        if len(vectors) < H:
            return self._handle_insufficient_elements(line_elements, H)

        # 2. Desplazamiento por ventana impar para detectar rupturas
        rupture_scores = []
        for i in range(1, len(vectors) - 1):
            window = [vectors[i-1], vectors[i], vectors[i+1]]
            cohesion = np.mean([
                self._calculate_cosine(window[0], window[1]),
                self._calculate_cosine(window[1], window[2]),
                self._calculate_cosine(window[0], window[2])
            ])
            rupture_scores.append((i, cohesion))

        # 3. Seleccionar las H-1 rupturas con menor cohesión
        rupture_scores.sort(key=lambda x: x[1])
        cut_indices = [x[0] for x in rupture_scores[:H-1]]
        cut_indices.sort()

        # 4. Asignar elementos a columnas basándose en los cortes
        columns = [[] for _ in range(H)]
        current_col = 0
        for i, elem in enumerate(line_elements):
            elem_x = elem.get('cx', 0)
            if current_col < len(cut_indices) and elem_x > vectors[cut_indices[current_col]][0]:
                current_col += 1
            columns[current_col].append(elem)

        return columns

    def _handle_insufficient_elements(self, elements: List[Dict[str, Any]], H: int) -> List[List[Dict[str, Any]]]:
        """Maneja casos donde hay menos elementos que columnas (L_k < H)."""
        columns = [[] for _ in range(H)]
        for i, elem in enumerate(elements):
            columns[i % H].append(elem)  # Distribución cíclica simple
        return columns