# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

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
        extra_compile_args=["-std=c++17", "-O3"],
        define_macros=[("NDEBUG", None)],
    ),
]

setup(
    name="si_engine",
    version="2.0.0",
    author="SI Engine Team",
    description="Synthetic Intelligence Engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
)