set(basic_usage_targets_cc
  basic_usage_cc
  basic_usage_autotune_cc
)

add_executable(basic_usage_cc)
  target_sources(basic_usage_cc
  PRIVATE
  basic_usage.cc
)

add_executable(basic_usage_autotune_cc)
target_sources(basic_usage_autotune_cc
  PRIVATE
  basic_usage_autotune.cc
)

foreach(tgt ${basic_usage_targets_cc})
  string(REPLACE "_cc" "" bin_name ${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${bin_name})
  set_target_properties(${tgt} PROPERTIES HIP_ARCHITECTURES "${HIPDECOMP_HIP_ARCH_LIST}")
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE hipdecomp)
  set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE CXX)
endforeach()

install(
  TARGETS ${basic_usage_targets_cc}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cc/basic_usage
)
