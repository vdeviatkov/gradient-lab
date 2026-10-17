# Character RNN
target_sources(ml_scratch_cpp PRIVATE
    char_ngram.cpp
    rnn.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/reporting.cpp
    utils/rnn_checkpoint.cpp
    utils/rnn_diagnostics.cpp
)

add_executable(cpp_char_rnn main.cpp)
target_link_libraries(cpp_char_rnn PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_char_rnn)

if(BUILD_TESTING)
    add_ml_test(cpp_rnn_test tests/rnn_test.cpp ml_scratch_utils)
endif()
