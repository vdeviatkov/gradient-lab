# Residual networks

add_executable(cpp_resnet_cifar10 main.cpp)
target_link_libraries(cpp_resnet_cifar10 PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_resnet_cifar10)

if(BUILD_TESTING)
    add_ml_test(cpp_residual_test tests/residual_test.cpp ml_scratch_utils)
endif()
