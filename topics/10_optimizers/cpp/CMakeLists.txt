# Optimizers
target_sources(ml_scratch_cpp PRIVATE
    optimizer.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/reporting.cpp
)

add_executable(cpp_optimizers main.cpp)
target_link_libraries(cpp_optimizers PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_optimizers)

if(BUILD_TESTING)
    add_ml_test(cpp_optimizer_test tests/optimizer_test.cpp ml_scratch_utils)
endif()
