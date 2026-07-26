# ==========================================================================
# Generator tests
# ==========================================================================

file(GLOB GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
set(GENERATOR_DDP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test_generator_ddp.cc)
list(REMOVE_ITEM GENERATOR_SOURCES ${GENERATOR_DDP_SOURCE})

infini_train_add_test_suite(test_generator
  SOURCES ${GENERATOR_SOURCES}
  TEST_TIMEOUT 40
)

if(USE_CUDA)
  add_executable(test_generator_ddp ${GENERATOR_DDP_SOURCE})
  target_compile_options(test_generator_ddp PRIVATE -Wno-error)
  target_link_libraries(test_generator_ddp PRIVATE GTest::gtest)
  target_include_directories(test_generator_ddp PRIVATE ${glog_SOURCE_DIR}/src)
  link_infini_train_exe(test_generator_ddp)

  add_test(NAME test_generator_ddp COMMAND $<TARGET_FILE:test_generator_ddp>)
  set_tests_properties(test_generator_ddp PROPERTIES LABELS "cuda;distributed" TIMEOUT 40)
endif()
