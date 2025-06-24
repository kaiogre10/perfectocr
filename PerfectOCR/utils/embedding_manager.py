# PerfectOCR/utils/embedding_manager.py
import logging
from typing import Optional, List, Union
import numpy as np
from threading import Lock
import time
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Gestor centralizado de embeddings para optimizar el uso del modelo SentenceTransformer.
    Implementa patrón Singleton para evitar múltiples cargas del modelo.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._model = None
        self._model_name = 'all-MiniLM-L6-v2'
        self._is_loaded = False
        self._initialized = True
        self._loading_thread = None
        self._load_start_time = None
        
        # 🚀 NUEVO: Sistema de cache y métricas
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_encodings': 0,
            'batch_encodings': 0
        }
        
        logger.info("EmbeddingManager inicializado (Singleton)")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Carga el modelo de embeddings de forma lazy con fallback.
        """
        if self._is_loaded:
            return True
        
        if model_name:
            self._model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Descargando/Cargando modelo: {self._model_name}")
            
            # Lista de modelos ordenados por preferencia
            models_to_try = [
                self._model_name,
                'all-MiniLM-L6-v2',  # Tu modelo preferido
                'paraphrase-multilingual-MiniLM-L12-v2',  # El que ya tenías
                'distiluse-base-multilingual-cased'  # Alternativa
            ]
            
            for model_attempt in models_to_try:
                try:
                    logger.info(f"🔄 Intentando: {model_attempt}")
                    self._model = SentenceTransformer(model_attempt)
                    self._model_name = model_attempt
                    self._is_loaded = True
                    logger.info(f"✅ Éxito con: {model_attempt}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Falló {model_attempt}: {str(e)[:100]}...")
                    continue
            
            logger.error("❌ No se pudo cargar ningún modelo")
            return False
        
        except ImportError:
            logger.error("sentence-transformers no está instalado")
            return False
        except Exception as e:
            logger.error(f"Error general: {e}")
            return False
    
    def get_model(self):
        """Mantener carga lazy pero optimizada"""
        if not self._is_loaded:
            logger.info("Cargando modelo de embeddings bajo demanda...")
            start_time = time.perf_counter()
            self.load_model()
            load_time = time.perf_counter() - start_time
            logger.info(f"Modelo cargado en {load_time:.2f}s")
        return self._model
    
    @lru_cache(maxsize=512)
    def _cached_encode_single(self, text: str, convert_to_tensor: bool = True):
        """Cache para textos individuales usando LRU."""
        self._stats['cache_misses'] += 1
        return self._model.encode(text, convert_to_tensor=convert_to_tensor)
    
    def encode(self, texts: Union[str, List[str]], convert_to_tensor: bool = True):
        """
        Codifica textos a embeddings con cache optimizado.
        """
        model = self.get_model()
        if model is None:
            raise RuntimeError("Modelo de embeddings no disponible")
        
        self._stats['total_encodings'] += 1
        
        if isinstance(texts, str):
            # Usar cache para textos individuales
            try:
                # Intentar obtener del cache
                result = self._cached_encode_single(texts, convert_to_tensor)
                self._stats['cache_hits'] += 1
                return result
            except:
                # Si falla el cache, usar el método normal
                self._stats['cache_misses'] += 1
                return model.encode(texts, convert_to_tensor=convert_to_tensor)
        else:
            # Para listas, usar encoding batch (más eficiente)
            self._stats['batch_encodings'] += 1
            return model.encode(texts, convert_to_tensor=convert_to_tensor)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud coseno entre dos textos.
        
        Args:
            text1: Primer texto
            text2: Segundo texto
            
        Returns:
            float: Similitud coseno (0-1)
        """
        try:
            model = self.get_model()
            if model is None:
                return 0.0
            
            from sentence_transformers.util import cos_sim
            embeddings = model.encode([text1, text2], convert_to_tensor=True)
            similarity = cos_sim(embeddings[0:1], embeddings[1:2])[0][0].item()
            return similarity
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0
    
    def calculate_batch_similarity(self, reference_text: str, candidate_texts: List[str]) -> List[float]:
        """
        Calcula similitud entre un texto de referencia y una lista de candidatos.
        
        Args:
            reference_text: Texto de referencia
            candidate_texts: Lista de textos candidatos
            
        Returns:
            List[float]: Lista de similitudes
        """
        try:
            model = self.get_model()
            if model is None:
                return [0.0] * len(candidate_texts)
            
            from sentence_transformers.util import cos_sim
            all_texts = [reference_text] + candidate_texts
            embeddings = model.encode(all_texts, convert_to_tensor=True)
            similarities = cos_sim(embeddings[0:1], embeddings[1:])
            return [sim.item() for sim in similarities[0]]
        except Exception as e:
            logger.error(f"Error calculando similitud batch: {e}")
            return [0.0] * len(candidate_texts)
    
    def is_available(self) -> bool:
        """
        Verifica si el modelo está disponible.
        
        Returns:
            bool: True si el modelo está cargado y disponible
        """
        return self._is_loaded and self._model is not None
    
    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        return {
            'model_name': self._model_name,
            'is_loaded': self._is_loaded,
            'is_available': self.is_available()
        }
    
    def ensure_model_ready(self, timeout: float = 10.0) -> bool:
        """
        Asegura que el modelo esté listo, cargándolo si es necesario.
        Retorna True si está listo, False si timeout.
        """
        if self._is_loaded:
            return True
        
        if not self._loading_thread:
            # Inicia carga en background solo si no está ya cargando
            self._start_background_loading()
        
        # Espera con timeout
        start_wait = time.perf_counter()
        while not self._is_loaded and (time.perf_counter() - start_wait) < timeout:
            time.sleep(0.1)
        
        return self._is_loaded
    
    def _start_background_loading(self):
        """Inicia la carga en background."""
        def _load():
            self._load_start_time = time.perf_counter()
            logger.info("Iniciando carga de embeddings en background...")
            self.load_model()
            load_time = time.perf_counter() - self._load_start_time
            logger.info(f"Embeddings cargados en background: {load_time:.2f}s")
        
        self._loading_thread = threading.Thread(target=_load, daemon=True)
        self._loading_thread.start()
    
    def get_performance_stats(self) -> dict:
        """Obtiene estadísticas de rendimiento."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'total_encoding_requests': total_requests,
            'batch_vs_individual_ratio': round((self._stats['batch_encodings'] / self._stats['total_encodings'] * 100) if self._stats['total_encodings'] > 0 else 0, 2),
            'model_name': self._model_name,
            'is_loaded': self._is_loaded,
            **self._stats
        }
    
    def clear_cache(self):
        """Limpia el cache LRU."""
        self._cached_encode_single.cache_clear()
        logger.info("Cache de embeddings limpiado")
    
    def get_cache_info(self):
        """Obtiene información del cache."""
        return self._cached_encode_single.cache_info()

# Instancia global para uso fácil
embedding_manager = EmbeddingManager()