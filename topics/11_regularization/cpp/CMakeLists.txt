# Regularization and normalization

add_executable(cpp_regularization main.cpp)
target_link_libraries(cpp_regularization PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_regularization)

if(BUILD_TESTING)
    add_ml_test(cpp_regularization_test tests/regularization_test.cpp ml_scratch_utils)
endif()
