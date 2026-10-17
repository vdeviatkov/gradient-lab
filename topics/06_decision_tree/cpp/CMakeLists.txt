# Decision trees
target_sources(ml_scratch_cpp PRIVATE
    decision_tree.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/tree_diagnostics.cpp
)

add_executable(cpp_decision_tree main.cpp)
target_link_libraries(cpp_decision_tree PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_decision_tree)

if(BUILD_TESTING)
    add_ml_test(cpp_decision_tree_test tests/decision_tree_test.cpp ml_scratch_utils)
endif()
