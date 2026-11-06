if (CUDECOMP_BUILD_CPU_ONLY)
  foreach(kind api transpose halo)
    if (kind STREQUAL "api")
      set(source fortran_cpu_api_test.f90)
    else()
      set(source fortran_${kind}_tests.f90)
    endif()
    add_executable(cudecomp_test_fortran_cpu_${kind} ${source})
    target_compile_definitions(cudecomp_test_fortran_cpu_${kind} PRIVATE CUDECOMP_BUILD_CPU_ONLY)
    set_target_properties(cudecomp_test_fortran_cpu_${kind} PROPERTIES Fortran_PREPROCESS ON)
    target_compile_options(cudecomp_test_fortran_cpu_${kind} PRIVATE
      $<$<Fortran_COMPILER_ID:GNU>:-ffree-line-length-none>)
    target_include_directories(cudecomp_test_fortran_cpu_${kind} PRIVATE ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(cudecomp_test_fortran_cpu_${kind} PRIVATE cudecomp_fort cudecomp)
    add_test(NAME cudecomp_fortran_cpu_${kind}
      COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
              ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cudecomp_test_fortran_cpu_${kind}>)
    set_tests_properties(cudecomp_fortran_cpu_${kind} PROPERTIES
      TIMEOUT ${CUDECOMP_TEST_TIMEOUT} LABELS "cpu;fortran;mpi")
  endforeach()
  return()
endif()

add_executable(cudecomp_test_fortran_api)
target_sources(cudecomp_test_fortran_api
  PRIVATE
  fortran_api_test.f90
)
target_include_directories(cudecomp_test_fortran_api
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
)
target_link_libraries(cudecomp_test_fortran_api
  PRIVATE
  MPI::MPI_Fortran
  cudecomp
  cudecomp_fort
)
target_compile_options(cudecomp_test_fortran_api PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_link_options(cudecomp_test_fortran_api PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_compile_options(cudecomp_test_fortran_api PRIVATE -O3)
set_target_properties(cudecomp_test_fortran_api PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

add_executable(cudecomp_test_fortran_transpose)
target_sources(cudecomp_test_fortran_transpose
  PRIVATE
  fortran_transpose_tests.f90
)
target_include_directories(cudecomp_test_fortran_transpose
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
)
target_link_libraries(cudecomp_test_fortran_transpose
  PRIVATE
  MPI::MPI_Fortran
  cudecomp
  cudecomp_fort
)
target_compile_options(cudecomp_test_fortran_transpose PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_link_options(cudecomp_test_fortran_transpose PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_compile_options(cudecomp_test_fortran_transpose PRIVATE -O3)
set_target_properties(cudecomp_test_fortran_transpose PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

add_executable(cudecomp_test_fortran_halo)
target_sources(cudecomp_test_fortran_halo
  PRIVATE
  fortran_halo_tests.f90
)
target_include_directories(cudecomp_test_fortran_halo
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
)
target_link_libraries(cudecomp_test_fortran_halo
  PRIVATE
  MPI::MPI_Fortran
  cudecomp
  cudecomp_fort
)
target_compile_options(cudecomp_test_fortran_halo PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_link_options(cudecomp_test_fortran_halo PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>
)
target_compile_options(cudecomp_test_fortran_halo PRIVATE -O3)
set_target_properties(cudecomp_test_fortran_halo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

set(CUDECOMP_TEST_FORTRAN_API_MPI_RANKS 4)
set(CUDECOMP_TEST_FORTRAN_TRANSPOSE_MPI_RANKS 4)
set(CUDECOMP_TEST_FORTRAN_HALO_MPI_RANKS 4)

function(add_cudecomp_fortran_test test_name target labels mpi_ranks)
  add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
            ${MPIEXEC_NUMPROC_FLAG} ${mpi_ranks}
            $<TARGET_FILE:${target}>
            ${ARGN}
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "${labels}"
    TIMEOUT ${CUDECOMP_TEST_TIMEOUT}
  )
endfunction()

add_cudecomp_fortran_test(cudecomp_fortran_api cudecomp_test_fortran_api "fortran;api;mpi"
                          ${CUDECOMP_TEST_FORTRAN_API_MPI_RANKS})
add_cudecomp_fortran_test(cudecomp_fortran_transpose cudecomp_test_fortran_transpose "fortran;transpose;mpi"
                          ${CUDECOMP_TEST_FORTRAN_TRANSPOSE_MPI_RANKS})
add_cudecomp_fortran_test(cudecomp_fortran_halo cudecomp_test_fortran_halo "fortran;halo;mpi"
                          ${CUDECOMP_TEST_FORTRAN_HALO_MPI_RANKS})
