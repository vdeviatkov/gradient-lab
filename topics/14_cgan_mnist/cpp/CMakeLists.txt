# Conditional GAN
target_sources(ml_scratch_cpp PRIVATE
    gan.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/gan_checkpoint.cpp
    utils/reporting.cpp
)

add_executable(cpp_cgan_mnist main.cpp)
target_link_libraries(cpp_cgan_mnist PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_cgan_mnist)

if(BUILD_TESTING)
    add_ml_test(cpp_gan_test tests/gan_test.cpp ml_scratch_utils)
endif()
