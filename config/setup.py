import sys
import os
import sysconfig
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
import numpy as np
from Cython.Build import cythonize
import services.system_service as system_service
from typing import Dict, List, Any


# ─── Config Intel oneAPI ───
INTEL_BIN = "/opt/intel/oneapi/compiler/2026.0/bin"
CONDA_ENV       = "/home/kaiogre05/miniforge3/envs/intel"
OPENCV_ROOT     = "/home/kaiogre05/PerfectOCR/components/opencv_dep/install"
GCC_INSTALL_DIR = f"{CONDA_ENV}/lib/gcc/x86_64-conda-linux-gnu/15.2.0"


class IntelBuildExt(build_ext):
    def build_extensions(self):
        cxx = f"{INTEL_BIN}/icpx"
        cc  = f"{INTEL_BIN}/icx"
        self.compiler.set_executable("compiler_so", [cxx])
        self.compiler.set_executable("compiler_cxx", [cxx])
        self.compiler.set_executable("linker_so", [cxx])
        self.compiler.set_executable("compiler", [cc])
        build_ext.build_extensions(self)


def build_extensions(PROJECT_ROOT: str, config: Dict[str, Any]):
    PYX_FILE = config.get("comp_funcs_file", "")
    comp_utils_name = config.get("comp_funcs_name", "")

    libs_dir = config.get("libs_path", "")
    IMAGE_FILE = config.get("comp_services_file", "")
    compiled_services_path = config.get("comp_services_path", "")
    imge_name = "utils.compiled_services.image"

    components_paths: List[str] = list(config["components_path"])
    components_name = config["components"]
    components_paths.append(np.get_include())
    components_paths.append(f"{OPENCV_ROOT}/include/opencv4")
    components_paths.append(f"{CONDA_ENV}/include")

    # ─── Forzar compilador Intel ANTES de setup() ───
    os.environ["CC"]  = f"{INTEL_BIN}/icx"
    os.environ["CXX"] = f"{INTEL_BIN}/icpx"
    sysconfig.get_config_vars()["CC"]  = f"{INTEL_BIN}/icx"
    sysconfig.get_config_vars()["CXX"] = f"{INTEL_BIN}/icpx"

    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[PYX_FILE],
        ),
        Extension(
            name=imge_name,
            sources=[IMAGE_FILE],
            include_dirs=components_paths,
            library_dirs=[
                libs_dir,
                f"{OPENCV_ROOT}/lib",
                f"{CONDA_ENV}/lib",
            ],
            libraries=components_name + [
                "file_handler",
                "payload_container",
                "c_utils",
                "opencv_core", "opencv_imgproc", "opencv_imgcodecs",
                "tbb",
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++20",
                "-mavx2",
                "-mfma",
                f"--gcc-install-dir={GCC_INSTALL_DIR}",
            ],
            extra_link_args=[
                f"-B{GCC_INSTALL_DIR}",
                f"-L{GCC_INSTALL_DIR}",       
                f"-L{CONDA_ENV}/lib",
                f"-L{OPENCV_ROOT}/lib",
                f"-Wl,-rpath,{CONDA_ENV}/lib",
                f"-Wl,-rpath,{OPENCV_ROOT}/lib",
            ],
            runtime_library_dirs=[
                f"{CONDA_ENV}/lib",
                f"{OPENCV_ROOT}/lib",
            ],
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
                include_path=[compiled_services_path]
            ),
            cmdclass={"build_ext": IntelBuildExt},
        )
    finally:
        sys.argv = old_argv

    system_service.set_system_config(PROJECT_ROOT, {})
    system_service.cleanup_project()