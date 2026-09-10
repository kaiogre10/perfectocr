import numpy as np
from typing import Any

def load(filepath: str) -> int:
    """Carga una imagen desde disco y devuelve un handle opaco."""
def get_array(ptr_addr: int) -> np.ndarray[Any, np.dtype[np.uint8]]:  ...
def release(ptr_addr: int) -> None: ...