# Snake reinforcement learning
target_sources(ml_scratch_cpp PRIVATE
    reinforcement.cpp
    snake.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/reinforcement_checkpoint.cpp
    utils/snake_render.cpp
)

add_executable(cpp_snake_rl main.cpp)
target_link_libraries(cpp_snake_rl PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_snake_rl)

if(BUILD_TESTING)
    add_ml_test(cpp_reinforcement_test tests/reinforcement_test.cpp ml_scratch_utils)
    add_ml_test(cpp_snake_test tests/snake_test.cpp ml_scratch_utils)
endif()
