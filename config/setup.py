import os
import sys
from services.log_service import basic_exc_logger
from setuptools import setup, Extension
from Cython.Build import cythonize
import services.system_service as system_service
from typing import Dict

def build_extensions(project_root: str, config: Dict[str, str]):
    PYX_FILE = config.get("pxy_file_path", "")
    comp_utils_name = config.get("comp_utils_name", "")
    libs_dir = config.get("libs_dir", "")
    header_path = config.get("header_path", "")
    IMAGE_FILE = os.path.join(project_root, "utils", "compiled_services", "image.pyx")
    imge_name = "utils.compiled_services.image"
    
    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[PYX_FILE],
        ),
        Extension(
            name=imge_name,
            sources=[IMAGE_FILE],
            include_dirs=[header_path],
            library_dirs=["C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/lib"],
            extra_objects = [libs_dir],
            language = "c++"
        )
    ]
    command = config["compile_command"]
    old_argv = sys.argv
    sys.argv = command
    try:
        setup(
            ext_modules=cythonize(
                extensions,
                compiler_directives={"language_level": "3"},
            ),
        )
    finally:
        sys.argv = old_argv

    system_service.set_system_config(project_root, {})
    system_service.cleanup_project()