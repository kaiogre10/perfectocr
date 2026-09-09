from typing import List, Dict, Optional
import logging
import ctypes
from domain.class_models import TypeModels

LIB: ctypes.CDLL
LOAD: ctypes.CDLL

logger = logging.getLogger(__name__)

def storage_config(config: Dict[str, List[str]]) -> None:
    global LIB, LOAD # type: ignore
    loader_bin_path = config.get("loader", "")
    storage_bin_path = config.get("buffer_handler", "")
    try:
        LOAD = ctypes.CDLL(loader_bin_path) # type: ignore
        if not LOAD:
            raise OSError("ERROR CARGANDO CONTENEDOR")
        LOAD.load_image.argtypes = [ctypes.c_char_p]
        LOAD.load_image.restype = None
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
        raise

    try:
        LIB = ctypes.CDLL(storage_bin_path) # type: ignore
        if not LIB:
            raise OSError("ERROR CARGANDO BUFFER")
        
        LIB.create_deque.argtypes = []
        LIB.create_deque.restype = None
        
        LIB.reserve_buffer.argtypes = [ctypes.c_size_t]
        LIB.reserve_buffer.restype = ctypes.c_void_p
        LIB.commit_buffer.argtypes = [ctypes.c_int]
        LIB.commit_buffer.restype = None
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO BUFFER HANDLER: {e}", exc_info=True)
        raise

def load_image(path: str):
    LOAD.load_image(path.encode(TypeModels.UTF8.value))

def storage_data(texts: List[str]) -> Optional[List[int]]:
    buffers: List[int] = []
    for plain_texts in texts:
        raw_payload = plain_texts.encode(TypeModels.ASCII.value, "ignore")
        if not raw_payload:
            raise EncodingWarning("ERROR CONVIRTIENDO BYTES")
        
        len_bytes = len(raw_payload) * 2
        
        if len_bytes < len(plain_texts):
            raise ArithmeticError("ERROR TAMAÑO BYTES")

        ptr = LIB.reserve_buffer(len_bytes) # Solicitar memoria
        if not ptr:
            raise ctypes.ArgumentError("ERROR DE PUNTEROS")
            
        ctypes.memmove(ptr, raw_payload.decode(TypeModels.ASCII.value).encode(TypeModels.UTF16_CSHARP.value), len_bytes) # Guardar bytes typados
        LIB.commit_buffer(1) # Avisar que ya están guardados
        buffers.append(len_bytes)

    return buffers