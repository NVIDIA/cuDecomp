find_package(GTest CONFIG QUIET)
if (NOT GTest_FOUND)
  find_package(GTest MODULE QUIET)
endif()

if (NOT GTest_FOUND AND CUDECOMP_TEST_FETCH_GTEST)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if (TARGET GTest::gtest)
  set(CUDECOMP_TEST_FRAMEWORK_TARGET GTest::gtest)
elseif (TARGET GTest::GTest)
  set(CUDECOMP_TEST_FRAMEWORK_TARGET GTest::GTest)
endif()

if (NOT CUDECOMP_TEST_FRAMEWORK_TARGET)
  message(FATAL_ERROR
    "GoogleTest was not found. Install a system GoogleTest package or configure with "
    "-DCUDECOMP_TEST_FETCH_GTEST=ON to fetch the pinned test dependency."
  )
endif()

set(CUDECOMP_TEST_MPIEXEC_PREFLAGS_DEFAULT "")
if (MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI")
  set(CUDECOMP_TEST_MPIEXEC_PREFLAGS_DEFAULT "--oversubscribe")
endif()

set(CUDECOMP_TEST_MPIEXEC_PREFLAGS "${CUDECOMP_TEST_MPIEXEC_PREFLAGS_DEFAULT}" CACHE STRING
    "Extra MPI launcher flags for cuDecomp CTest tests.")
separate_arguments(CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST UNIX_COMMAND "${CUDECOMP_TEST_MPIEXEC_PREFLAGS}")

set(CUDECOMP_TEST_TIMEOUT 90)

set(CUDECOMP_TEST_ENABLE_NVSHMEM ${CUDECOMP_ENABLE_NVSHMEM})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/backend_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/backend_config.h
)

add_library(cudecomp_test_support STATIC)
target_sources(cudecomp_test_support
  PRIVATE
  backend_test_context.cc
  backend_utils.cc
  gpu_test_utils.cc
  mpi_test_utils.cc
  test_utils.cc
)
target_include_directories(cudecomp_test_support
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${NVHPC_CUDA_INCLUDE_DIR}
  ${NCCL_INCLUDE_DIR}
)
target_link_libraries(cudecomp_test_support
  PUBLIC
  MPI::MPI_CXX
  NVHPC::CUDART
  ${NCCL_LIBRARY}
  ${CUDECOMP_TEST_FRAMEWORK_TARGET}
)

add_executable(cudecomp_test_api)
target_sources(cudecomp_test_api
  PRIVATE
  api_tests.cc
  mpi_test_main.cc
)
target_link_libraries(cudecomp_test_api
  PRIVATE
  cudecomp
  cudecomp_test_support
)
set_target_properties(cudecomp_test_api PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

set(CUDECOMP_TEST_API_MPI_RANKS 4)
add_test(
  NAME cudecomp_api
  COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
          ${MPIEXEC_NUMPROC_FLAG} ${CUDECOMP_TEST_API_MPI_RANKS}
          $<TARGET_FILE:cudecomp_test_api>
)
set_tests_properties(cudecomp_api PROPERTIES
  LABELS "api;mpi"
  TIMEOUT ${CUDECOMP_TEST_TIMEOUT}
)

add_executable(cudecomp_test_transpose)
target_sources(cudecomp_test_transpose
  PRIVATE
  mpi_test_main.cc
  transpose_tests.cc
)
target_link_libraries(cudecomp_test_transpose
  PRIVATE
  cudecomp
  cudecomp_test_support
  NVHPC::CUTENSOR
)
set_target_properties(cudecomp_test_transpose PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

set(CUDECOMP_TEST_TRANSPOSE_MPI_RANKS 4)
function(add_cudecomp_transpose_test test_name filter labels)
  add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
            ${MPIEXEC_NUMPROC_FLAG} ${CUDECOMP_TEST_TRANSPOSE_MPI_RANKS}
            $<TARGET_FILE:cudecomp_test_transpose>
            --gtest_filter=${filter}
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "${labels}"
    TIMEOUT ${CUDECOMP_TEST_TIMEOUT}
  )
endfunction()

add_cudecomp_transpose_test(cudecomp_transpose_mpi "MpiBackends/*" "transpose;mpi")
add_cudecomp_transpose_test(cudecomp_transpose_nccl "NcclBackends/*" "transpose;nccl")
set_tests_properties(cudecomp_transpose_nccl PROPERTIES
  ENVIRONMENT "CUDECOMP_TEST_KEEPALIVE_BACKEND=nccl"
)

if (CUDECOMP_ENABLE_NVSHMEM)
  add_cudecomp_transpose_test(cudecomp_transpose_nvshmem "NvshmemBackends/*" "transpose;nvshmem")
  set_tests_properties(cudecomp_transpose_nvshmem PROPERTIES
    ENVIRONMENT "CUDECOMP_TEST_KEEPALIVE_BACKEND=nvshmem;NVSHMEM_DISABLE_NCCL=1"
  )
endif()

add_executable(cudecomp_test_halo)
target_sources(cudecomp_test_halo
  PRIVATE
  mpi_test_main.cc
  halo_tests.cc
)
target_link_libraries(cudecomp_test_halo
  PRIVATE
  cudecomp
  cudecomp_test_support
)
set_target_properties(cudecomp_test_halo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ctest
)

set(CUDECOMP_TEST_HALO_MPI_RANKS 4)
function(add_cudecomp_halo_test test_name filter labels)
  add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
            ${MPIEXEC_NUMPROC_FLAG} ${CUDECOMP_TEST_HALO_MPI_RANKS}
            $<TARGET_FILE:cudecomp_test_halo>
            --gtest_filter=${filter}
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "${labels}"
    TIMEOUT ${CUDECOMP_TEST_TIMEOUT}
  )
endfunction()

add_cudecomp_halo_test(cudecomp_halo_mpi "MpiBackends/*" "halo;mpi")
add_cudecomp_halo_test(cudecomp_halo_nccl "NcclBackends/*" "halo;nccl")
set_tests_properties(cudecomp_halo_nccl PROPERTIES
  ENVIRONMENT "CUDECOMP_TEST_KEEPALIVE_BACKEND=nccl"
)

if (CUDECOMP_ENABLE_NVSHMEM)
  add_cudecomp_halo_test(cudecomp_halo_nvshmem "NvshmemBackends/*" "halo;nvshmem")
  set_tests_properties(cudecomp_halo_nvshmem PROPERTIES
    ENVIRONMENT "CUDECOMP_TEST_KEEPALIVE_BACKEND=nvshmem;NVSHMEM_DISABLE_NCCL=1"
  )
endif()

add_cudecomp_transpose_test(cudecomp_cuda_graphs "CudaGraphMpiBackends/*" "cuda_graphs;mpi")
set_tests_properties(cudecomp_cuda_graphs PROPERTIES
  ENVIRONMENT "CUDECOMP_ENABLE_CUDA_GRAPHS=1"
)
add_cudecomp_transpose_test(cudecomp_nccl_user_buffer_registration "NcclUserBufferRegistration/*"
                            "nccl;nccl_ubr")
set_tests_properties(cudecomp_nccl_user_buffer_registration PROPERTIES
  ENVIRONMENT "CUDECOMP_ENABLE_NCCL_UBR=1;CUDECOMP_TEST_KEEPALIVE_BACKEND=nccl"
)

if (CUDECOMP_BUILD_FORTRAN)
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
endif()
