set(basic_usage_targets_cc
  basic_usage_cc
  basic_usage_autotune_cc
)

add_executable(basic_usage_cc)
  target_sources(basic_usage_cc
  PRIVATE
  basic_usage.cu
)

add_executable(basic_usage_autotune_cc)
target_sources(basic_usage_autotune_cc
  PRIVATE
  basic_usage_autotune.cu
)

foreach(tgt ${basic_usage_targets_cc})
  string(REPLACE "_cc" "" bin_name ${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${bin_name})
  if (CMAKE_VERSION VERSION_LESS 3.18)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_CC_FLAGS}>)
  else()
    set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "${CUDA_CC_LIST}")
  endif()
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE cudecomp)
endforeach()

install(
  TARGETS ${basic_usage_targets_cc}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cc/basic_usage
)
