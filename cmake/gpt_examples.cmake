include_guard(GLOBAL)

function(infini_train_add_gpt_executable target_name)
  add_executable(${target_name})
  target_link_libraries(${target_name} PRIVATE infini_train_gpt_sources InfiniTrain::training_cli)
  # Keep the framework archives as direct link entries in their whole-archive
  # group; transitive de-duplication can otherwise move the core outside it.
  link_infini_train_exe(${target_name})
endfunction()

# Called by example/<model>/CMakeLists.txt. Each model contributes source files
# to the shared GPT program and declares a preset-loading executable.
function(infini_train_add_gpt_model model_name)
  cmake_parse_arguments(MODEL "" "FLAGFILE" "SOURCES" ${ARGN})
  if(MODEL_UNPARSED_ARGUMENTS OR MODEL_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "infini_train_add_gpt_model(${model_name}): invalid or incomplete arguments")
  endif()
  if(NOT MODEL_SOURCES OR NOT MODEL_FLAGFILE)
    message(FATAL_ERROR
      "infini_train_add_gpt_model(${model_name}): SOURCES and FLAGFILE are required")
  endif()
  if(NOT TARGET infini_train_gpt_sources)
    message(FATAL_ERROR "GPT models must be declared from the example build")
  endif()

  # Absolute source paths retain the declaring directory's meaning when the
  # interface sources are consumed by executables in other model directories.
  foreach(source IN LISTS MODEL_SOURCES)
    get_filename_component(source_path "${source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    target_sources(infini_train_gpt_sources INTERFACE "${source_path}")
  endforeach()
  get_filename_component(flagfile "${MODEL_FLAGFILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${flagfile}" OR IS_DIRECTORY "${flagfile}")
    message(FATAL_ERROR "Training preset not found: ${flagfile}")
  endif()

  infini_train_add_gpt_executable(${model_name})
  target_compile_definitions(${model_name} PRIVATE
    INFINITRAIN_GPT_DEFAULT_FLAGFILE="configs/${model_name}/train.flags")

  # Target-file directories also cover multi-config generators and custom output
  # locations. Preset edits are copied on the next build without recompilation.
  add_custom_target(${model_name}_training_config
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${model_name}>/configs/${model_name}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${flagfile}" "$<TARGET_FILE_DIR:${model_name}>/configs/${model_name}/train.flags"
    VERBATIM
  )
  add_dependencies(${model_name} ${model_name}_training_config)
endfunction()
