set(transpose_test_targets_f
  transpose_test_R32_f
  transpose_test_R64_f
  transpose_test_C32_f
  transpose_test_C64_f
)

set(halo_test_targets_f
  halo_test_R32_f
  halo_test_R64_f
  halo_test_C32_f
  halo_test_C64_f
)

foreach(tgt ${transpose_test_targets_f})
  add_executable(${tgt})
  target_sources(${tgt}
    PRIVATE
    transpose_test.f90
  )
endforeach()

foreach(tgt ${halo_test_targets_f})
  add_executable(${tgt})
  target_sources(${tgt}
    PRIVATE
    halo_test.f90
  )
endforeach()

foreach(tgt ${transpose_test_targets_f} ${halo_test_targets_f})
  string(REPLACE "_f" "" bin_name ${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${bin_name})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PUBLIC cudecomp)
  target_link_libraries(${tgt} PUBLIC cudecomp_fort)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
endforeach()

target_compile_definitions(transpose_test_R32_f PUBLIC R32)
target_compile_definitions(transpose_test_R64_f PUBLIC R64)
target_compile_definitions(transpose_test_C32_f PUBLIC C32)
target_compile_definitions(transpose_test_C64_f PUBLIC C64)

target_compile_definitions(halo_test_R32_f PUBLIC R32)
target_compile_definitions(halo_test_R64_f PUBLIC R64)
target_compile_definitions(halo_test_C32_f PUBLIC C32)
target_compile_definitions(halo_test_C64_f PUBLIC C64)

install(
  TARGETS ${transpose_test_targets_f} ${halo_test_targets_f}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/fortran
)
