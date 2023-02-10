set(transpose_test_targets_cc
  transpose_test_R32_cc
  transpose_test_R64_cc
  transpose_test_C32_cc
  transpose_test_C64_cc
)

set(halo_test_targets_cc
  halo_test_R32_cc
  halo_test_R64_cc
  halo_test_C32_cc
  halo_test_C64_cc
)

foreach(tgt ${transpose_test_targets_cc})
  add_executable(${tgt})
  target_sources(${tgt}
    PRIVATE
    transpose_test.cc
  )
endforeach()

foreach(tgt ${halo_test_targets_cc})
  add_executable(${tgt})
  target_sources(${tgt}
    PRIVATE
    halo_test.cc
  )
endforeach()

foreach(tgt ${transpose_test_targets_cc} ${halo_test_targets_cc})
  string(REPLACE "_cc" "" bin_name ${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${bin_name})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${MPI_CXX_INCLUDE_DIRS}
    ${NVHPC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE NVHPC::CUDART)
  target_link_libraries(${tgt} PUBLIC cudecomp)
endforeach()

target_compile_definitions(transpose_test_R32_cc PRIVATE R32)
target_compile_definitions(transpose_test_R64_cc PRIVATE R64)
target_compile_definitions(transpose_test_C32_cc PRIVATE C32)
target_compile_definitions(transpose_test_C64_cc PRIVATE C64)

target_compile_definitions(halo_test_R32_cc PRIVATE R32)
target_compile_definitions(halo_test_R64_cc PRIVATE R64)
target_compile_definitions(halo_test_C32_cc PRIVATE C32)
target_compile_definitions(halo_test_C64_cc PRIVATE C64)

install(
  TARGETS ${transpose_test_targets_cc} ${halo_test_targets_cc}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/cc
)
