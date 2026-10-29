# ==========================================================================
# Parallel topology tests
# ==========================================================================

infini_train_add_test(test_parallel_topology
  SOURCES test_global_env.cc
  TEST_MAIN_SOURCE test_global_env_main.cc
  LABELS cpu
)

infini_train_add_test(test_context_parallel
  SOURCES test_context_parallel.cc
  TEST_MAIN_SOURCE test_context_parallel_main.cc
  TEST_FILTER "ContextParallelLayoutTest.*"
  BINARY_REGISTRATION
  LABELS cpu
  TEST_TIMEOUT 30
)

add_test(NAME ContextParallelLayout.Contiguous
  COMMAND test_context_parallel --attention-cp-layout=contiguous
    --gtest_filter=ContextParallelLayoutTest.*)
add_test(NAME ContextParallelLayout.SingleRank
  COMMAND test_context_parallel --context-parallel=1
    --gtest_filter=ContextParallelLayoutTest.*)
set_tests_properties(ContextParallelLayout.Contiguous ContextParallelLayout.SingleRank
  PROPERTIES LABELS cpu TIMEOUT 30)

if(USE_CUDA AND USE_NCCL)
  function(add_cp_attention_test name cp_size tp_size mode hierarchy layout)
    add_test(NAME ContextParallelCUDA.${name}
      COMMAND test_context_parallel --gtest_filter=ContextParallelAttentionTest.*
        --context-parallel=${cp_size} --tensor-parallel=${tp_size}
        --cp-comm-type=${mode} --hierarchical-context-parallel-sizes=${hierarchy}
        --attention-cp-layout=${layout})
    set_tests_properties(ContextParallelCUDA.${name} PROPERTIES
      LABELS "cuda;accelerator;hardware" RUN_SERIAL TRUE TIMEOUT 120
      ENVIRONMENT "NVIDIA_TF32_OVERRIDE=0;OMP_NUM_THREADS=2")
  endfunction()

  foreach(layout IN ITEMS contiguous zigzag)
    foreach(mode IN ITEMS all_gather p2p a2a a2a+p2p)
      set(hierarchy "")
      if(mode STREQUAL "a2a+p2p")
        set(hierarchy "2,2")
      endif()
      add_cp_attention_test(${layout}.${mode} 4 1 ${mode} "${hierarchy}" ${layout})
    endforeach()
  endforeach()
  add_cp_attention_test(zigzag.p2p_cp2 2 1 p2p "" zigzag)
  add_cp_attention_test(zigzag.hierarchy24 8 1 a2a+p2p "2,4" zigzag)
  add_cp_attention_test(zigzag.hierarchy42 8 1 a2a+p2p "4,2" zigzag)
  add_cp_attention_test(zigzag.tp2_p2p 4 2 p2p "" zigzag)
  add_cp_attention_test(zigzag.tp2_hierarchy22 4 2 a2a+p2p "2,2" zigzag)
  add_cp_attention_test(zigzag.hierarchy14 4 1 a2a+p2p "1,4" zigzag)
  add_cp_attention_test(zigzag.hierarchy41 4 1 a2a+p2p "4,1" zigzag)
endif()
