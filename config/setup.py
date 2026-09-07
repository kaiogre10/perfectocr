import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import services.system_service as system_service
from typing import Dict

def build_extensions(project_root: str, config: Dict[str, str]):
    PYX_FILE = config.get("pxy_file_path", "")
    comp_utils_name = config.get("comp_utils_name", "")
    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[PYX_FILE],
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