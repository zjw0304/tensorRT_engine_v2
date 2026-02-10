# Tests CMakeLists.txt

# Main test executable with all test files
add_executable(trt_engine_tests
    test_logger.cpp
    test_memory.cpp
    test_builder.cpp
    test_engine.cpp
    test_cuda_graph.cpp
    test_batcher.cpp
    test_multi_gpu.cpp
)

if(TARGET GTest::GTest)
    target_link_libraries(trt_engine_tests
        PRIVATE
            trt_engine
            GTest::GTest
            GTest::Main
            CUDA::cudart
    )
else()
    target_link_libraries(trt_engine_tests
        PRIVATE
            trt_engine
            gtest
            gtest_main
            CUDA::cudart
    )
endif()

target_include_directories(trt_engine_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(trt_engine_tests)
