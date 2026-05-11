# Common test utilities + custom test main (initializes GlobalEnv before RUN_ALL_TESTS).
add_library(test_main OBJECT test_main.cc)
target_include_directories(test_main PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_main PUBLIC GTest::gtest)
