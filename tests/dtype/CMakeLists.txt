# ==========================================================================
# Dtype dispatch tests
# ==========================================================================

file(GLOB DTYPE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
list(FILTER DTYPE_SOURCES EXCLUDE REGEX ".*compile_fail.*")

infini_train_add_test_suite(test_dtype
  SOURCES ${DTYPE_SOURCES}
)

# Negative compile test: missing dtype registration must fail at compile time.
set(DTYPE_DISPATCH_COMPILE_FAIL_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_dtype_dispatch_compile_fail.cc)

try_compile(DTYPE_DISPATCH_COMPILE_UNEXPECTEDLY_SUCCEEDED
  ${CMAKE_BINARY_DIR}/CMakeFiles/try_compile_dtype_dispatch_missing_map
  SOURCES ${DTYPE_DISPATCH_COMPILE_FAIL_SOURCE}
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DCMAKE_CXX_FLAGS=-I${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE DTYPE_DISPATCH_TRY_COMPILE_OUTPUT
)

if(DTYPE_DISPATCH_COMPILE_UNEXPECTEDLY_SUCCEEDED)
  message(FATAL_ERROR
    "dtype dispatch compile-fail test unexpectedly succeeded.\n"
    "Source: ${DTYPE_DISPATCH_COMPILE_FAIL_SOURCE}\n"
    "Output:\n${DTYPE_DISPATCH_TRY_COMPILE_OUTPUT}")
endif()

add_custom_target(test_dtype_dispatch_compile_fail
  COMMAND ${CMAKE_COMMAND} -E echo
    "dtype dispatch compile-fail check passed (missing dtype registration correctly fails to compile)."
  VERBATIM
)
