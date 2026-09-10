import os
import sys
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
import services.system_service as system_service
from typing import Dict

def build_extensions(PROJECT_ROOT: str, config: Dict[str, str]):
    PYX_FILE = config.get("pxy_file_path", "")
    comp_utils_name = config.get("comp_utils_name", "")
    libs_dir = config.get("libs_dir", "")
    header_path = config.get("header_path", "")
    IMAGE_FILE = os.path.join(PROJECT_ROOT, "utils", "compiled_services", "image.pyx")
    imge_name = "utils.compiled_services.image"
    
    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[PYX_FILE],
        ),
        Extension(
            name=imge_name,
            sources=[IMAGE_FILE],
            include_dirs=[
                os.path.join(PROJECT_ROOT, "components", "image_container"),
                os.path.join(PROJECT_ROOT, "components", "image_loader"),
                np.get_include(),
            ],
            library_dirs=[os.path.join(PROJECT_ROOT, "bin")],
            libraries=["image_container", "image_loader"],
            language="c++",
            extra_compile_args=["-std=c++20"],
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
                include_path=[os.path.join(PROJECT_ROOT, "utils", "compiled_services")]
            ),
        )
    finally:
        sys.argv = old_argv

    system_service.set_system_config(PROJECT_ROOT, {})
    system_service.cleanup_project()