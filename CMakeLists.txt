cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(trt_engine LANGUAGES CXX CUDA)

# ── C++ and CUDA standards ──────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architectures ─────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# ── Build type defaults ─────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# ── Options ─────────────────────────────────────────────────────────────────
option(TRT_ENGINE_BUILD_PYTHON   "Build Python bindings (requires pybind11)" OFF)
option(TRT_ENGINE_BUILD_TESTS    "Build unit tests (requires GTest)"         OFF)
option(TRT_ENGINE_BUILD_BENCHMARKS "Build benchmarks"                        OFF)

# ── CMake module path ──────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── Find dependencies ──────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

# ── Compiler flags ──────────────────────────────────────────────────────────
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    endif()
endif()

# ── Library sources ─────────────────────────────────────────────────────────
set(TRT_ENGINE_SOURCES
    src/logger.cpp
    src/memory.cpp
    src/cuda_utils.cpp
    src/model_converter.cpp
    src/calibrator.cpp
    src/builder.cpp
    src/engine.cpp
    src/cuda_graph.cpp
    src/multi_stream.cpp
    src/batcher.cpp
    src/multi_gpu.cpp
    src/profiler.cpp
)

add_library(trt_engine SHARED ${TRT_ENGINE_SOURCES})

target_include_directories(trt_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(trt_engine
    PUBLIC
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        CUDA::cudart
    PRIVATE
        ${CMAKE_DL_LIBS}
)

# Link NVML if available
find_library(NVML_LIBRARY
    NAMES nvidia-ml nvml
    PATHS
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64/stubs
        /usr/lib64
)
if(NVML_LIBRARY)
    target_link_libraries(trt_engine PUBLIC ${NVML_LIBRARY})
    target_compile_definitions(trt_engine PRIVATE TRT_ENGINE_HAS_NVML=1)
    message(STATUS "Found NVML: ${NVML_LIBRARY}")
else()
    message(STATUS "NVML not found - GPU monitoring features disabled")
endif()

# ── Set library version / SOVERSION ─────────────────────────────────────────
set_target_properties(trt_engine PROPERTIES
    VERSION   1.0.0
    SOVERSION 1
    OUTPUT_NAME trt_engine
)

# ── Python bindings ─────────────────────────────────────────────────────────
if(TRT_ENGINE_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        add_subdirectory(python)
        message(STATUS "Python bindings enabled")
    else()
        message(WARNING "pybind11 not found - Python bindings disabled")
    endif()
endif()

# ── Tests ───────────────────────────────────────────────────────────────────
if(TRT_ENGINE_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
    message(STATUS "Tests enabled")
endif()

# ── Benchmarks ──────────────────────────────────────────────────────────────
if(TRT_ENGINE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
    message(STATUS "Benchmarks enabled")
endif()

# ── Install targets ─────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS trt_engine
    EXPORT trt_engine-targets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/trt_engine
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT trt_engine-targets
    FILE        trt_engine-targets.cmake
    NAMESPACE   trt_engine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trt_engine
)
