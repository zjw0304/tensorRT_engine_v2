"""
setup.py - Build and install the trt_engine Python package.

Uses CMake to build the C++ pybind11 extension module, then installs it
as part of the Python package.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension that delegates building to CMake."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build_ext that invokes CMake to build the C++ extension."""

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DTRT_ENGINE_BUILD_PYTHON=ON",
            "-DTRT_ENGINE_BUILD_TESTS=OFF",
            "-DTRT_ENGINE_BUILD_BENCHMARKS=OFF",
        ]

        build_args = ["--config", cfg, "--", "-j"]

        # Use the project root (one level up from python/)
        project_root = os.path.join(ext.sourcedir, "..")
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", project_root] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True,
        )


setup(
    name="trt_engine",
    version="1.0.0",
    author="TRT Engine Team",
    description="TensorRT High-Performance GPU Inference Engine",
    long_description=open(
        os.path.join(os.path.dirname(__file__), "..", "docs", "implementation_plan.md"),
        "r",
        encoding="utf-8",
    ).read()
    if os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "docs", "implementation_plan.md")
    )
    else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[CMakeExtension("trt_engine_python", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "Pillow>=9.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
