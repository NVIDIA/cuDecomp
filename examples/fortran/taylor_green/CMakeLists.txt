add_executable(tg_fortran)
target_sources(tg_fortran
  PRIVATE
  tg.f90
)
set_target_properties(tg_fortran
                      PROPERTIES OUTPUT_NAME tg)
target_include_directories(tg_fortran
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
  ${NVHPC_CUFFT_INCLUDE_DIR}
)
target_link_libraries(tg_fortran PRIVATE MPI::MPI_Fortran)
target_link_libraries(tg_fortran PRIVATE NVHPC::CUFFT)
target_link_libraries(tg_fortran PRIVATE cudecomp)
target_link_libraries(tg_fortran PRIVATE cudecomp_fort)
target_compile_options(tg_fortran PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -acc -gpu=${CUF_GPU_ARG}>)
target_link_options(tg_fortran PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -acc -gpu=${CUF_GPU_ARG}>)
install(
  TARGETS tg_fortran
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/taylor_green
)
