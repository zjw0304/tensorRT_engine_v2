# FindTensorRT.cmake
# ------------------
# Finds the NVIDIA TensorRT library.
#
# Imported Targets:
#   TensorRT::nvinfer       - The nvinfer library
#   TensorRT::nvonnxparser  - The nvonnxparser library
#
# Result Variables:
#   TensorRT_FOUND          - True if TensorRT was found
#   TensorRT_INCLUDE_DIRS   - Include directories
#   TensorRT_LIBRARIES      - Libraries to link
#   TensorRT_VERSION        - Version string (e.g. "8.6.1")

# Search paths
set(_TRT_SEARCH_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/TensorRT
    /usr/local/TensorRT/lib
    /usr/local/cuda/lib64
    /usr/lib64
    $ENV{TENSORRT_ROOT}/lib
    $ENV{TENSORRT_ROOT}
)

set(_TRT_INCLUDE_SEARCH_PATHS
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
    /usr/local/TensorRT/include
    /usr/local/cuda/include
    $ENV{TENSORRT_ROOT}/include
)

# Find NvInfer.h
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${_TRT_INCLUDE_SEARCH_PATHS}
    DOC "TensorRT include directory"
)

# Find nvinfer library
find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    PATHS ${_TRT_SEARCH_PATHS}
    DOC "TensorRT nvinfer library"
)

# Find nvonnxparser library
find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    PATHS ${_TRT_SEARCH_PATHS}
    DOC "TensorRT nvonnxparser library"
)

# Extract version from NvInfer.h
if(TensorRT_INCLUDE_DIR)
    # Try NvInferVersion.h first (TRT 8+), then fall back to NvInfer.h
    if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
        set(_TRT_VERSION_HEADER "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    elseif(EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
        set(_TRT_VERSION_HEADER "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    endif()

    if(_TRT_VERSION_HEADER)
        file(READ "${_TRT_VERSION_HEADER}" _TRT_VERSION_CONTENT)

        string(REGEX MATCH "NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENT}")
        set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "NV_TENSORRT_MINOR[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENT}")
        set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "NV_TENSORRT_PATCH[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENT}")
        set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")

        if(TensorRT_VERSION_MAJOR AND TensorRT_VERSION_MINOR AND TensorRT_VERSION_PATCH)
            set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
        endif()
    endif()
endif()

# Standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
        TensorRT_INCLUDE_DIR
    VERSION_VAR
        TensorRT_VERSION
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES
        ${TensorRT_nvinfer_LIBRARY}
        ${TensorRT_nvonnxparser_LIBRARY}
    )

    # Create imported target TensorRT::nvinfer
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()

    # Create imported target TensorRT::nvonnxparser
    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found TensorRT ${TensorRT_VERSION}")
    message(STATUS "  Include: ${TensorRT_INCLUDE_DIR}")
    message(STATUS "  nvinfer: ${TensorRT_nvinfer_LIBRARY}")
    message(STATUS "  nvonnxparser: ${TensorRT_nvonnxparser_LIBRARY}")
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
)
