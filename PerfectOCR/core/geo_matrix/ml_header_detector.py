# PerfectOCR/core/geo_matrix/ml_header_detector.py
import lightgbm
import joblib
import pandas as pd
import os
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MLHeaderDetector:
    def __init__(self, model_path: str):
        """
        Inicializa el detector cargando el modelo de Machine Learning entrenado.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encontró en la ruta: {model_path}")
        self.model = joblib.load(model_path)
        self.feature_columns = [
            'length', 'num_digits', 'num_alpha', 'is_upper',
            'rel_y_center', 'rel_x_center', 'rel_width', 'rel_height'
        ]
        logger.info(f"MLHeaderDetector inicializado con modelo: {model_path}")

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera las mismas características que se usaron en el entrenamiento.
        """
        # Usar 'text_raw' en lugar de 'text' para compatibilidad con la estructura de datos del OCR
        text_column = 'text_raw' if 'text_raw' in df.columns else 'text'
        df[text_column] = df[text_column].astype(str).fillna('')
        df['length'] = df[text_column].str.len()
        df['num_digits'] = df[text_column].str.count(r'\d')
        df['num_alpha'] = df[text_column].str.count(r'[a-zA-Z]')
        df['is_upper'] = df[text_column].str.isupper().astype(int)

        df['page_h'] = df['page_h'].replace(0, 1)
        df['page_w'] = df['page_w'].replace(0, 1)

        df['x_center'] = (df['xmin'] + df['xmax']) / 2
        df['y_center'] = (df['ymin'] + df['ymax']) / 2
        df['width'] = df['xmax'] - df['xmin']
        df['height'] = df['ymax'] - df['ymin']

        df['rel_y_center'] = df['y_center'] / df['page_h']
        df['rel_x_center'] = df['x_center'] / df['page_w']
        df['rel_width'] = df['width'] / df['page_w']
        df['rel_height'] = df['height'] / df['page_h']

        # Verificar que todas las columnas necesarias estén presentes
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Columnas faltantes en las características: {missing_columns}")
            logger.error(f"Columnas disponibles: {list(df.columns)}")
            raise ValueError(f"Faltan columnas de características: {missing_columns}")

        return df[self.feature_columns]

    def detect(self, all_words: List[Dict[str, Any]], page_w: int, page_h: int) -> List[Dict[str, Any]]:
        """
        Toma TODAS las palabras del OCR, predice cuáles son encabezados
        y devuelve una lista solo con ellas.
        """
        if not all_words:
            logger.warning("MLHeaderDetector: No se recibieron palabras para analizar")
            return []

        logger.info(f"MLHeaderDetector: Analizando {len(all_words)} palabras con dimensiones {page_w}x{page_h}")

        # Convertir a DataFrame y añadir dimensiones
        words_df = pd.DataFrame(all_words)
        words_df['page_w'] = page_w
        words_df['page_h'] = page_h

        # Verificar columnas requeridas
        required_columns = ['xmin', 'xmax', 'ymin', 'ymax']
        missing_required = [col for col in required_columns if col not in words_df.columns]
        if missing_required:
            logger.error(f"Columnas requeridas faltantes: {missing_required}")
            logger.error(f"Columnas disponibles: {list(words_df.columns)}")
            return []

        # Generar características
        try:
            features = self._generate_features(words_df)
            logger.debug(f"Características generadas: {features.shape}")
        except Exception as e:
            logger.error(f"Error generando características: {e}")
            return []

        # Predecir con el modelo
        try:
            predictions = self.model.predict(features)
            logger.info(f"Predicciones realizadas: {len(predictions)} valores, {sum(predictions)} positivos")
            
            # Log de algunas predicciones para debug
            if len(predictions) > 0:
                sample_texts = [all_words[i].get('text_raw', 'N/A')[:20] for i in range(min(5, len(all_words)))]
                sample_preds = predictions[:5]
                logger.debug(f"Muestra de predicciones: {list(zip(sample_texts, sample_preds))}")
        except Exception as e:
            logger.error(f"Error en predicción del modelo: {e}")
            return []

        # Filtrar y devolver solo las palabras clasificadas como encabezado
        header_words = [word for i, word in enumerate(all_words) if predictions[i] == 1]
        
        logger.info(f"MLHeaderDetector: {len(header_words)} palabras identificadas como cabecera")
        
        if header_words:
            # Log de las palabras de cabecera encontradas
            header_texts = [word.get('text_raw', 'N/A') for word in header_words]
            logger.debug(f"Palabras de cabecera: {header_texts}")

        # Opcional: Ordenar por posición X para los módulos siguientes
        header_words.sort(key=lambda w: w.get('xmin', 0))

        return header_words