# MLP on MNIST

add_executable(cpp_mlp_mnist main.cpp)
target_link_libraries(cpp_mlp_mnist PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_mlp_mnist)
