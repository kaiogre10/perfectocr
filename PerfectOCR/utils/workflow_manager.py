# PerfectOCR/utils/workflow_manager.py
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.batch_tools import chunked, get_optimal_workers, estimate_processing_time
from main import PerfectOCRWorkflow
from utils.config_loader import ConfigLoader
from utils.output_handlers import ExcelOutputHandler

logger = logging.getLogger(__name__)

# Variable global para workers
workflow_instance = None

def _init_worker(cfg_path: str):
    global workflow_instance
    workflow_instance = PerfectOCRWorkflow(cfg_path)

def process_image_worker(image_path: str) -> Dict[str, Any]:
    """Procesa una imagen usando el workflow del worker."""
    global workflow_instance
    if workflow_instance is None:
        raise RuntimeError("Worker no inicializado correctamente")
    
    try:
        return workflow_instance.run_single_image(image_path)
    except Exception as e:
        logger.error(f"Error en worker procesando {image_path}: {e}")
        return {"error": str(e), "image": image_path}

class WorkflowManager:
    def __init__(self, config_path: str, max_workers_override: int = None):
        """Inicializa el gestor.

        Args:
            config_path: Ruta al YAML maestro.
            max_workers_override: Si se proporciona, fuerza este nº de workers
                                   físicos/procesos, ignorando la detección
                                   automática.
        """
        self.config_path = config_path
        self.small_batch_limit = 5
        self.excel_handler = ExcelOutputHandler()

        if max_workers_override and max_workers_override > 0:
            # El usuario manda
            self.max_physical_cores = max_workers_override
        else:
            # Intentar obtener un valor recomendado desde el YAML → ConfigLoader
            try:
                loader = ConfigLoader(config_path)
                self.max_physical_cores = loader.get_max_workers_for_cpu()
            except Exception:
                # Fallback conservador: CPU - 2 hilos o 4 si no se puede detectar
                self.max_physical_cores = max(1, (os.cpu_count() or 4) - 2)
        
    def should_use_batch_mode(self, num_images: int) -> bool:
        """Decide si usar modo lote basado en el número de imágenes."""
        return num_images > self.small_batch_limit
    
    def process_images(self, image_paths: List[Path], output_dir: str, force_mode: str = None, workers_override: int = None) -> Dict[str, Any]:
        """
        Procesa imágenes usando el modo óptimo.
        
        Args:
            image_paths: Lista de rutas de imágenes
            output_dir: Directorio de salida
            force_mode: 'interactive' o 'batch' para forzar un modo específico
            workers_override: Si se proporciona, fuerza este nº de workers
        """
        num_images = len(image_paths)
        
        # Permitir override de workers en caliente
        if workers_override and workers_override > 0:
            self.max_physical_cores = workers_override

        # Decidir modo
        if force_mode == 'interactive':
            use_batch = False
        elif force_mode == 'batch':
            use_batch = True
        else:
            use_batch = self.should_use_batch_mode(num_images)
        
        # Mostrar estimación
        estimation = estimate_processing_time(num_images)
        logger.info(f"Procesando {num_images} imágenes en modo {'LOTE' if use_batch else 'INTERACTIVO'}")
        logger.info(f"Tiempo estimado: {estimation['parallel_minutes']:.1f} min con {estimation['workers']} workers")
        
        if use_batch:
            result = self._process_batch_mode(image_paths, output_dir, workers_override)
        else:
            result = self._process_interactive_mode(image_paths, output_dir)
        
        # Generar archivo Excel consolidado con todas las matrices
        self._generate_consolidated_excel(result, output_dir)
        
        return result
    
    def _process_interactive_mode(self, image_paths: List[Path], output_dir: str) -> Dict[str, Any]:
        """Modo interactivo: un solo proceso, carga única de modelos."""
        logger.info("Iniciando modo INTERACTIVO")
        
        # Importar aquí para evitar dependencias circulares
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Crear workflow una sola vez
        workflow = PerfectOCRWorkflow(self.config_path)
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Procesando imagen {i}/{len(image_paths)}: {image_path.name}")
            try:
                result = workflow.run_single_image(str(image_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error procesando {image_path}: {e}")
                results.append({"error": str(e), "image": str(image_path)})
        
        return {
            "mode": "interactive",
            "processed": len(results),
            "results": results
        }
    
    def _process_batch_mode(self, image_paths: List[Path], output_dir: str, workers_override: int = None) -> Dict[str, Any]:
        """Modo lote: múltiples procesos con inicialización optimizada."""
        logger.info("Iniciando modo LOTE")
        
        workers = workers_override if (workers_override and workers_override > 0) else get_optimal_workers(len(image_paths), self.max_physical_cores)
        batch_size = workers * 2
        
        results = []
        processed_count = 0
        
        # Usar funciones globales para evitar problemas de pickling
        with ProcessPoolExecutor(
            max_workers=workers, 
            initializer=_init_worker,
            initargs=(self.config_path,)
        ) as executor:
            
            # Procesar en chunks para controlar memoria
            for chunk in chunked(image_paths, batch_size):
                logger.info(f"Procesando chunk de {len(chunk)} imágenes...")
                
                # Enviar trabajos del chunk actual
                futures = {
                    executor.submit(process_image_worker, str(path)): path 
                    for path in chunk
                }
                
                # Recoger resultados del chunk
                for future in as_completed(futures):
                    image_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            logger.info(f"Progreso: {processed_count}/{len(image_paths)} imágenes completadas")
                            
                    except Exception as e:
                        logger.error(f"Error obteniendo resultado para {image_path}: {e}")
                        results.append({"error": str(e), "image": str(image_path)})
        
        return {
            "mode": "batch",
            "workers_used": workers,
            "batch_size": batch_size,
            "processed": len(results),
            "results": results
        }
    
    def _generate_consolidated_excel(self, processing_result: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Genera un archivo Excel consolidado con todas las matrices semánticamente corregidas.
        
        Args:
            processing_result: Resultado del procesamiento (modo batch o interactive)
            output_dir: Directorio de salida
            
        Returns:
            Ruta al archivo Excel generado o None si hubo error
        """
        logger.info("Generando archivo Excel consolidado con matrices semánticamente corregidas...")
        
        matrices_data = []
        results = processing_result.get("results", [])
        
        for result in results:
            if "error" in result:
                logger.warning(f"Omitiendo resultado con error: {result.get('image', 'unknown')}")
                continue
                
            # Extraer información de la matriz semánticamente corregida
            document_id = result.get("document_id", "unknown")
            
            # Buscar el archivo JSON de la matriz semánticamente corregida
            outputs = result.get("outputs", {})
            if "structured_table_json" in outputs:
                # Intentar cargar el JSON correspondiente de matriz semánticamente corregida
                structured_table_path = outputs["structured_table_json"]
                if structured_table_path:
                    # Construir la ruta del archivo de matriz semánticamente corregida
                    base_dir = os.path.dirname(structured_table_path)
                    base_name = document_id
                    semantic_matrix_path = os.path.join(base_dir, f"{base_name}_semantically_corrected_matrix.json")
                    
                    if os.path.exists(semantic_matrix_path):
                        try:
                            with open(semantic_matrix_path, 'r', encoding='utf-8') as f:
                                semantic_data = json.load(f)
                                
                            matrix_data = {
                                "document_id": document_id,
                                "headers": semantic_data.get("headers", []),
                                "semantic_types": semantic_data.get("semantic_types", []),
                                "matrix": semantic_data.get("matrix", [])
                            }
                            matrices_data.append(matrix_data)
                            
                        except Exception as e:
                            logger.error(f"Error cargando matriz semántica para {document_id}: {e}")
                    else:
                        logger.warning(f"Archivo de matriz semántica no encontrado: {semantic_matrix_path}")
        
        if not matrices_data:
            logger.warning("No se encontraron matrices semánticamente corregidas para consolidar en Excel")
            return None
        
        # Generar archivo Excel
        excel_path = self.excel_handler.save_semantically_corrected_matrices(
            matrices_data=matrices_data,
            output_dir=output_dir,
            file_name="ground_truth_batch.xlsx"
        )
        
        if excel_path:
            logger.info(f"Archivo Excel consolidado generado exitosamente: {excel_path}")
            logger.info(f"Total de matrices incluidas: {len(matrices_data)}")
        
        return excel_path