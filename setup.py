"""pybind11 C++ extension build script."""
import os
import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "engine.cpp.sim_core",
        ["engine/cpp/sim_core.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "engine/cpp",
        ],
        language="c++",
        extra_compile_args=["/std:c++17", "/O2", "/DNDEBUG"] if sys.platform == "win32" else ["-std=c++17", "-O3", "-DNDEBUG"],
    ),
]

setup(
    name="baseball-sim-core",
    version="2.2.0",
    ext_modules=ext_modules,
)
