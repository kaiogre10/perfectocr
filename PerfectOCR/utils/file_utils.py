import os
from pathlib import Path

def ensure_dir_exists(file_path: str) -> None:
    """
    Asegura que el directorio para un archivo exista, creándolo si es necesario.
    
    Args:
        file_path (str): Ruta del archivo
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True) 