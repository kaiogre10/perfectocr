# PerfectOCR/core/spatial_analysis/__init__.py
from .density_calculator import calculate_density_map
from .gradient_calculator import (
    detect_noise_regions_combined
)
import numpy as np
from typing import List, Dict, Tuple

__all__ = [
    'calculate_density_map',
    'detect_noise_regions_combined',
]

def map_line_to_pixel_gaps(binary_line: np.ndarray, words: List[Dict], num_cols: int) -> Tuple[List[str], List[int]]:
    """
    binary_line : fila binaria recortada (0=tinta, 255=fondo)
    words       : [{'text_raw':str, 'xmin':int, 'xmax':int}, ...] (ordenados)
    num_cols    : H
    Returns     : (lista_textos_ordenados, cortes_x_relativos)
    """
    # 1) Pintar la línea: fila_binaria == 0 (tinta) o 1 (fondo)
    fila = (binary_line == 0).astype(np.uint8)        # 1=tinta
    # 2) Calcular secuencias de fondo → huecos
    gaps = []
    in_gap = False
    start_gap = 0
    for x, val in enumerate(fila):
        if val == 0 and not in_gap:      # empieza gap
            in_gap = True; start_gap = x
        elif val == 1 and in_gap:        # fin de gap
            gaps.append((start_gap, x-1))
            in_gap = False
    if in_gap:
        gaps.append((start_gap, len(fila)-1))

    # 3) Elegir H-1 huecos más anchos
    gap_widths = [(g[1]-g[0]+1, g[0]) for g in gaps]  # (ancho, x_inicio)
    gap_widths.sort(reverse=True)
    cuts   = sorted([g[1] for g in gap_widths[:max(0,num_cols-1)]])  # x de corte
    textos = [w['text_raw'] for w in words]   # orden ya viene del Reconstructor
    return textos, cuts
