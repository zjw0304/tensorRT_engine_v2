# Benchmarks CMakeLists.txt
add_executable(benchmark_throughput benchmark_throughput.cpp)
target_link_libraries(benchmark_throughput
    PRIVATE
        trt_engine
        CUDA::cudart
)
target_include_directories(benchmark_throughput
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(benchmark_latency benchmark_latency.cpp)
target_link_libraries(benchmark_latency
    PRIVATE
        trt_engine
        CUDA::cudart
)
target_include_directories(benchmark_latency
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(benchmark_nlp benchmark_nlp.cpp)
target_link_libraries(benchmark_nlp
    PRIVATE
        trt_engine
        CUDA::cudart
)
target_include_directories(benchmark_nlp
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
