# setup.py
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "si_engine",
        [
            "SIPythonBindings.cpp",
            "SIBrain.cpp",
            "SITrainer.cpp",
            "SIBrainPruneTool.cpp",
            "SIBenchmark.cpp",
        ],
        include_dirs=["."],
        libraries=["curl"],
        extra_compile_args=[
            "-std=c++17", "-O3",
            "-static-libstdc++",  # THIS IS THE FIX
            "-static-libgcc"
        ],
        extra_link_args=[
            "-static-libstdc++",  # THIS IS THE FIX
            "-static-libgcc"
        ],
        define_macros=[("NDEBUG", None)],
    ),
]

setup(
    name="si_engine",
    version="2.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
