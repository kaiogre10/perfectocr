import os
import cv2
import numpy as np
import base64
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ImageReceptionWorker:
    """
    Worker del módulo de recepción de imágenes preprocesadas.
    Su única responsabilidad es convertir imágenes base64 a numpy arrays
    y guardarlas para que main.py las pueda encontrar.
    NO hace OCR - esa responsabilidad es exclusiva del OCREngineCoordinator.
    """
    
    def __init__(self, enhanced_input_folder: str = "enhanced_input"):
        """
        Inicializa el worker de recepción.
        
        Args:
            enhanced_input_folder: Carpeta donde guardar las imágenes recibidas
        """
        self.enhanced_input_folder = enhanced_input_folder
        os.makedirs(self.enhanced_input_folder, exist_ok=True)
        logger.info(f"✅ Worker de recepción inicializado - Carpeta: {self.enhanced_input_folder}")
    
    def receive_and_save_images(self, doc_id: str, ocr_images_b64: Dict[str, str], 
                               noise_regions: List[List[int]]) -> Dict[str, Any]:
        """
        Recibe imágenes en base64, las convierte a numpy arrays y las guarda.
        
        Args:
            doc_id: ID del documento
            ocr_images_b64: Diccionario con imágenes en base64 por motor
            noise_regions: Lista de regiones de ruido (se guarda como metadatos)
            
        Returns:
            Resultado de la recepción (sin OCR)
        """
        logger.info(f"📥 Recibiendo imágenes preprocesadas para documento: {doc_id}")
        
        saved_images = {}
        
        # Convertir cada imagen base64 a numpy array y guardarla
        for engine, base64_img in ocr_images_b64.items():
            img_numpy = self._decode_base64_image(base64_img)
            if img_numpy is not None:
                # Guardar imagen como archivo temporal para que main.py la encuentre
                image_filename = f"{doc_id}_{engine}_preprocessed.png"
                image_path = os.path.join(self.enhanced_input_folder, image_filename)
                
                # Guardar imagen
                success = cv2.imwrite(image_path, img_numpy)
                if success:
                    saved_images[engine] = image_path
                    logger.info(f"✅ Imagen {engine} guardada: {image_filename}")
                else:
                    logger.error(f"❌ Error guardando imagen {engine} para {doc_id}")
                    return {"error": f"Error guardando imagen {engine}"}
            else:
                logger.error(f"❌ No se pudo decodificar imagen {engine} para {doc_id}")
                return {"error": f"Error decodificando imagen {engine}"}
        
        if not saved_images:
            logger.error(f"❌ No hay imágenes válidas para recibir en {doc_id}")
            return {"error": "No hay imágenes válidas para recibir"}
        
        # Guardar metadatos de ruido (opcional, para uso futuro)
        if noise_regions:
            self._save_noise_metadata(doc_id, noise_regions)
        
        logger.info(f"✅ Recepción completada para {doc_id} - {len(saved_images)} imágenes guardadas")
        
        return {
            "status": "received",
            "document_id": doc_id,
            "saved_images": saved_images,
            "noise_regions": noise_regions,
            "message": f"Imágenes preprocesadas recibidas y guardadas correctamente"
        }
    
    def _decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        """
        Decodifica una imagen en formato base64 a numpy array.
        
        Args:
            base64_string: Imagen codificada en base64
            
        Returns:
            Imagen como numpy array o None si hay error
        """
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decodificando imagen base64: {e}")
            return None
    
    def _save_noise_metadata(self, doc_id: str, noise_regions: List[List[int]]):
        """
        Guarda metadatos de regiones de ruido para uso futuro.
        
        Args:
            doc_id: ID del documento
            noise_regions: Lista de regiones de ruido
        """
        try:
            import json
            metadata_path = os.path.join(self.enhanced_input_folder, f"{doc_id}_noise_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({"document_id": doc_id, "noise_regions": noise_regions}, f, indent=2)
            logger.debug(f"📄 Metadatos de ruido guardados: {metadata_path}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron guardar metadatos de ruido para {doc_id}: {e}")

# Instancia global para uso desde el servidor
reception_worker = ImageReceptionWorker()
