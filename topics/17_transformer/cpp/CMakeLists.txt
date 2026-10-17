# Transformer
target_sources(ml_scratch_cpp PRIVATE
    transformer.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/transformer_checkpoint.cpp
    utils/transformer_diagnostics.cpp
)
target_include_directories(ml_scratch_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")
target_include_directories(ml_scratch_utils PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")

add_executable(cpp_transformer main.cpp)
target_link_libraries(cpp_transformer PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_transformer)

if(BUILD_TESTING)
    add_ml_test(cpp_transformer_test tests/transformer_test.cpp ml_scratch_utils)
endif()
