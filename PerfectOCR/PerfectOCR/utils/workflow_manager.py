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
        
        # Generar el único y final archivo Excel consolidado
        self._generate_final_results_excel(result, output_dir)
        
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
        
        results_map = {}
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Procesando imagen {i}/{len(image_paths)}: {image_path.name}")
            try:
                result = workflow.run_single_image(str(image_path))
                doc_id = result.get("document_id", image_path.stem)
                results_map[doc_id] = result
            except Exception as e:
                logger.error(f"Error procesando {image_path}: {e}", exc_info=True)
                doc_id = image_path.stem
                results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        
        return {
            "mode": "interactive",
            "processed": len(results_map),
            "results": results_map
        }
    
    def _process_batch_mode(self, image_paths: List[Path], output_dir: str, workers_override: int = None) -> Dict[str, Any]:
        """Modo lote: múltiples procesos con inicialización optimizada."""
        logger.info("Iniciando modo LOTE")
        
        workers = workers_override if (workers_override and workers_override > 0) else get_optimal_workers(len(image_paths), self.max_physical_cores)
        batch_size = workers * 2
        
        results_map = {}
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
                    doc_id = image_path.stem
                    try:
                        result = future.result()
                        results_map[doc_id] = result
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            logger.info(f"Progreso: {processed_count}/{len(image_paths)} imágenes completadas")
                            
                    except Exception as e:
                        logger.error(f"Error obteniendo resultado para {image_path}: {e}", exc_info=True)
                        results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        
        return {
            "mode": "batch",
            "workers_used": workers,
            "batch_size": batch_size,
            "processed": len(results_map),
            "results": results_map
        }
    
    def _generate_final_results_excel(self, processing_result: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Genera un archivo Excel consolidado con los resultados finales de math_max.
        """
        logger.info("Generando archivo Excel consolidado con matrices math_max (resultados finales)...")
        
        matrices_data = []
        results_map = processing_result.get("results", {})
        
        for document_id, doc_result in results_map.items():
            if "error" in doc_result and doc_result.get("error"):
                matrices_data.append({
                    "document_id": document_id, "error": doc_result.get("error", "Error desconocido")
                })
                continue
                
            math_max_path = os.path.join(output_dir, f"{document_id}_math_max_matrix.json")
            
            if os.path.exists(math_max_path):
                try:
                    with open(math_max_path, 'r', encoding='utf-8') as f:
                        math_max_data = json.load(f)
                    
                    math_max_data["document_id"] = document_id
                    matrices_data.append(math_max_data)
                    
                except Exception as e:
                    logger.error(f"Error cargando math_max_matrix para {document_id}: {e}")
                    matrices_data.append({"document_id": document_id, "error": f"Error al leer JSON: {e}"})
            else:
                logger.warning(f"Archivo de resultados finales no encontrado: {math_max_path}")
                matrices_data.append({"document_id": document_id, "error": "math_max_matrix.json no encontrado"})
        
        if not matrices_data:
            logger.warning("No se encontraron datos de resultados para consolidar en Excel.")
            return None
        
        excel_path = self.excel_handler.save_math_max_matrices(
            matrices_data=matrices_data,
            output_dir=output_dir,
            file_name="math_max_resultados_finales.xlsx"
        )
        
        if excel_path:
            logger.info(f"📊 Archivo Excel de RESULTADOS FINALES generado: {excel_path}")
            success_count = len([m for m in matrices_data if not m.get("error")])
            logger.info(f"📋 Total de documentos incluidos: {len(matrices_data)} (Exitosos: {success_count}, Errores: {len(matrices_data) - success_count})")
        
        return excel_path