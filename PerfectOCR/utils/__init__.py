# PerfectOCR/utils/__init__.py
from .encoders import NumpyEncoder
from .geometric import (
    get_polygon_bounds, get_polygon_height, get_polygon_width,
    get_polygon_y_center, get_polygon_x_center, get_shapely_polygon,
    calculate_iou, enrich_word_data_with_geometry, tighten_geometry
)
from .data_preparation import prepare_unified_text_elements 
from .table_formatter import TableFormatter
from .output_handlers import JsonOutputHandler, TextOutputHandler, MarkdownOutputHandler
from .geometry_transformers import (
    convert_to_vector_representation, 
    restore_from_vector_representation,
    vectorize_element_list,
    devectorize_element_list
)
from .spatial_utils import (
    crop_json_lines, crop_binary_matrix
)
__all__ = [
    'NumpyEncoder',
    'get_polygon_bounds', 'get_polygon_height', 'get_polygon_width',
    'get_polygon_y_center', 'get_polygon_x_center', 'get_shapely_polygon',
    'calculate_iou', 'enrich_word_data_with_geometry', 'tighten_geometry',
    'prepare_unified_text_elements',
    'TableFormatter', 
    'JsonOutputHandler', 'TextOutputHandler', 'MarkdownOutputHandler',
    'convert_to_vector_representation', 'restore_from_vector_representation',
    'vectorize_element_list', 'devectorize_element_list', 
    'crop_json_lines', 'crop_binary_matrix', 'tighten_geometry'
    ]
