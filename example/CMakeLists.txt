# Keep the established build/<executable> paths when examples use subdirectories.
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/gpt_examples.cmake)

# Interface sources are compiled directly into each executable, preserving model
# registration objects and each consumer's external-backend compile definitions.
add_library(infini_train_gpt_sources INTERFACE)
target_sources(infini_train_gpt_sources INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/pretrain_gpt.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/tiny_shakespeare_dataset.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/tokenizer.cc"
)
infini_train_add_gpt_executable(pretrain_gpt)

# Discover only directory manifests; each manifest explicitly declares its files.
# CONFIGURE_DEPENDS notices newly added/removed model directories on the next build.
file(GLOB example_manifests CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt"
)
list(SORT example_manifests)
foreach(manifest IN LISTS example_manifests)
  get_filename_component(example_dir "${manifest}" DIRECTORY)
  add_subdirectory("${example_dir}")
endforeach()
