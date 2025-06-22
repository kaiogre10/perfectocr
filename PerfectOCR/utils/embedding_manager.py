# PerfectOCR/utils/embedding_manager.py
import logging
from typing import Optional, List, Union
import numpy as np
from threading import Lock

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
        self._model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self._is_loaded = False
        self._initialized = True
        logger.info("EmbeddingManager inicializado (Singleton)")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Carga el modelo de embeddings de forma lazy.
        
        Args:
            model_name: Nombre del modelo a cargar (opcional)
            
        Returns:
            bool: True si se cargó exitosamente, False en caso contrario
        """
        if self._is_loaded:
            return True
            
        if model_name:
            self._model_name = model_name
            
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Cargando modelo de embeddings: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._is_loaded = True
            logger.info(f"Modelo de embeddings cargado exitosamente")
            return True
        except ImportError:
            logger.error("sentence-transformers no está instalado")
            return False
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            return False
    
    def get_model(self):
        """
        Obtiene el modelo cargado. Lo carga si no está disponible.
        
        Returns:
            SentenceTransformer: El modelo de embeddings
        """
        if not self._is_loaded:
            self.load_model()
        return self._model
    
    def encode(self, texts: Union[str, List[str]], convert_to_tensor: bool = True):
        """
        Codifica textos a embeddings.
        
        Args:
            texts: Texto o lista de textos a codificar
            convert_to_tensor: Si convertir a tensor PyTorch
            
        Returns:
            Embeddings codificados
        """
        model = self.get_model()
        if model is None:
            raise RuntimeError("Modelo de embeddings no disponible")
        
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

# Instancia global para uso fácil
embedding_manager = EmbeddingManager()