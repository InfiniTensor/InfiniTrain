cmake_minimum_required(VERSION 3.20)

project(InfiniTrainTidy LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

if(NOT LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 18 OR
   LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 19)
  message(FATAL_ERROR "InfiniTrainTidy requires LLVM/Clang 18")
endif()

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

add_library(InfiniTrainTidy MODULE
  InfiniTrainTidyModule.cpp
  NoMemberDefinitionsInHeadersCheck.cpp
)

target_compile_features(InfiniTrainTidy PRIVATE cxx_std_17)
target_include_directories(InfiniTrainTidy SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)
target_compile_options(InfiniTrainTidy PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(InfiniTrainTidy PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
)

set_target_properties(InfiniTrainTidy PROPERTIES
  PREFIX ""
)
