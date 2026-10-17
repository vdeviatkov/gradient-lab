# PCA and k-means
target_sources(ml_scratch_cpp PRIVATE
    kmeans.cpp
    pca.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/evaluation.cpp
)

add_executable(cpp_pca_kmeans main.cpp)
target_link_libraries(cpp_pca_kmeans PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_pca_kmeans)

if(BUILD_TESTING)
    add_ml_test(cpp_kmeans_test tests/kmeans_test.cpp ml_scratch_utils)
    add_ml_test(cpp_pca_test tests/pca_test.cpp ml_scratch_utils)
endif()
