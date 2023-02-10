add_executable(poisson)
target_sources(poisson
  PRIVATE
  poisson.f90
)

target_include_directories(poisson
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
  ${NVHPC_CUFFT_INCLUDE_DIR}
)
target_link_libraries(poisson PRIVATE MPI::MPI_Fortran)
target_link_libraries(poisson PRIVATE NVHPC::CUFFT)
target_link_libraries(poisson PRIVATE cudecomp)
target_link_libraries(poisson PRIVATE cudecomp_fort)
target_compile_options(poisson PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
target_link_options(poisson PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)

install(
  TARGETS poisson
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/poisson
)
