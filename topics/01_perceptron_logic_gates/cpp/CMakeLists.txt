# Perceptron and logic gates
target_sources(ml_scratch_cpp PRIVATE
    perceptron.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/evaluation.cpp
    utils/logic_gates.cpp
)

add_executable(cpp_perceptron_logic_gates main.cpp)
target_link_libraries(cpp_perceptron_logic_gates PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_perceptron_logic_gates)
