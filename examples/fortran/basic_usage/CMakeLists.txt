set(basic_usage_targets_f
  basic_usage_f
  basic_usage_autotune_f
)

add_executable(basic_usage_f)
target_sources(basic_usage_f
  PRIVATE
  basic_usage.f90
)

add_executable(basic_usage_autotune_f)
target_sources(basic_usage_autotune_f
  PRIVATE
  basic_usage_autotune.f90
)

foreach(tgt ${basic_usage_targets_f})
  string(REPLACE "_f" "" bin_name ${tgt})
  set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${bin_name})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PRIVATE cudecomp)
  target_link_libraries(${tgt} PRIVATE cudecomp_fort)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
endforeach()

install(
  TARGETS ${basic_usage_targets_f}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/basic_usage
)
