# Python bindings CMakeLists.txt
pybind11_add_module(trt_engine_python bindings.cpp)
target_link_libraries(trt_engine_python PRIVATE trt_engine)
set_target_properties(trt_engine_python PROPERTIES
    OUTPUT_NAME trt_engine
)
