import sys
import os
# import sysconfig
from setuptools import setup, Extension
# from distutils.command.build_ext import build_ext
import numpy as np
from Cython.Build import cythonize
from services.system_service import cleanup_project
from typing import Dict, Any

INTEL_BIN = "/opt/intel/oneapi/compiler/2026.0/bin"
CONDA_ENV = "/home/kaiogre05/miniforge3/envs/intel"
GCC_INSTALL_DIR = f"{CONDA_ENV}/lib/gcc/x86_64-conda-linux-gnu/15.2.0"
cxx = f"{INTEL_BIN}/icpx"
cc = f"{INTEL_BIN}/icx"

# class IntelBuildExt(build_ext):
#     def build_extensions(self):
#         self.compiler.set_executable("compiler_so", [cxx])
#         self.compiler.set_executable("compiler_cxx", [cxx])
#         self.compiler.set_executable("linker_so", [cxx])
#         self.compiler.set_executable("compiler", [cc])
#         build_ext.build_extensions(self)

def build_extensions(PROJECT_ROOT: str, config: Dict[str, Any]):
    utils_file = config.get("comp_funcs_file", "")
    image_file = config.get("comp_services_file", "")
    # prune_workspace(PROJECT_ROOT, image_file, utils_file)
    
    comp_utils_name = config.get("comp_funcs_name", "")
    compiled_services_path = config.get("comp_services_path", "")
    imge_name = "utils.compiled_services.image"
    
    libs_dir = config["libs_path"]
    # print(f"LIBS DIR:\n"f"{libs_dir}")
    components_paths = config["components_paths"]
    
    components_paths.append(np.get_include())
    components_paths.append(f"{CONDA_ENV}/include")
    
    components_name = config["components"]
    
    # sysconfig.get_config_vars()["CC"]  = cc
    # sysconfig.get_config_vars()["CXX"] = cxx
    
    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[utils_file],
        ),
        # Extension(
        #     name=imge_name,
        #     sources=[image_file],
        #     include_dirs=components_paths,
        #     library_dirs=libs_dir + [
        #         f"{CONDA_ENV}/lib"
        #     ],
        #     libraries=components_name + [
        #         "opencv_core", "opencv_imgproc", "opencv_imgcodecs",
        #         "tbb"
        #     ],
        #     language="c++",
        #     extra_compile_args=[
        #         "-std=c++20",
        #         "-mavx2",
        #         "-mfma",
        #         f"--gcc-install-dir={GCC_INSTALL_DIR}"
        #     ],
        #     extra_link_args=[
        #         f"-B{GCC_INSTALL_DIR}",  # solo -B, el -L va por library_dirs
        #         f"-L{GCC_INSTALL_DIR}"
        #     ],
        #     runtime_library_dirs=[
        #         f"{CONDA_ENV}/lib",
        #         libs_dir[0]
        #     ]
        # )
    ]

    command = config["compile_command"]
    old_argv = sys.argv
    sys.argv = command
    try:
        setup(
            ext_modules=cythonize(
                extensions,
                compiler_directives={"language_level": "3"},
                # include_path=[compiled_services_path]
            ),
            # cmdclass={"build_ext": IntelBuildExt},
        )
    finally:
        sys.argv = old_argv

    prune_workspace(PROJECT_ROOT, image_file, utils_file)
    
def prune_workspace(PROJECT_ROOT: str, image_file: str, utils_file: str):
    if not image_file:
        target_cpp = None
        if utils_file:
            utils_path = os.path.splitext(utils_file)[0]
            target_c = (utils_path + ".c")
        else:
            target_c = None
            
    elif not utils_file:
        target_c = None
        image_path = os.path.splitext(image_file)[0]
        target_cpp = (image_path + ".cpp")
        
    else:
        utils_path = os.path.splitext(utils_file)[0]
        image_path = os.path.splitext(image_file)[0]
        target_cpp = (image_path + ".cpp")
        target_c = (utils_path + ".c")
        
    specific_files = [target_cpp, target_c]
    aditional_dirs = [os.path.join(PROJECT_ROOT, "build")]
    
    cleanup_project(specific_files=specific_files, aditional_dirs=aditional_dirs)  # type: ignore