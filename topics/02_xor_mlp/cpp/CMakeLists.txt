# XOR multilayer perceptron
target_sources(ml_scratch_cpp PRIVATE
    xor_mlp.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(cpp_xor_mlp main.cpp)
target_link_libraries(cpp_xor_mlp PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_xor_mlp)

if(BUILD_TESTING)
    add_ml_test(cpp_logic_models_test tests/logic_models_test.cpp ml_scratch_utils)
endif()
