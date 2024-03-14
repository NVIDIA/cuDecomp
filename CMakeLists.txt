cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# User-defined build options
option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" ON)
option(CUDECOMP_ENABLE_NVTX "Enable NVTX ranges" ON)
option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)
set(CUDECOMP_CUDA_CC_LIST "70;80;90" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
set(CUDECOMP_NCCL_HOME CACHE STRING "Path to search for NCCL installation. Use to override NVHPC provided NCCL version.")
set(CUDECOMP_NVSHMEM_HOME CACHE STRING "Path to search for NVSHMEM installation. Use to override NVHPC provided NVSHMEM version.")

# Use NVHPC compilers by default
find_program(NVHPC_CXX_BIN "nvc++" REQUIRED)
set(CMAKE_CXX_COMPILER ${NVHPC_CXX_BIN})

if (CUDECOMP_BUILD_FORTRAN)
  find_program(NVHPC_Fortran_BIN "nvfortran" REQUIRED)
  set(CMAKE_Fortran_COMPILER ${NVHPC_Fortran_BIN})
endif()

# Locate and use NVHPC CMake configuration
string(REPLACE "compilers/bin/nvc++" "cmake" NVHPC_CMAKE_DIR ${NVHPC_CXX_BIN})
set(CMAKE_PREFIX_PATH ${NVHPC_CMAKE_DIR})

# Limit optimization levels of CPU code compilation
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")
set(CMAKE_Fortran_FLAGS_RELEASE "-O1 -DNDEBUG")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")

if (CUDECOMP_BUILD_FORTRAN)
  set(LANGS CXX CUDA Fortran)
else()
  set(LANGS CXX CUDA)
endif()

project(cudecomp LANGUAGES ${LANGS})

# Detect if Cray compiler wrappers are available to assess if in Cray environment.
# We do not use the Cray compiler wrappers directly for greater flexibility.
find_program(CRAY_CC_BIN "CC")

if (CRAY_CC_BIN)
  message(STATUS "Found Cray CC wrapper. Compiling for Cray programming environment.")
endif()

# MPI
find_package(MPI REQUIRED)

if (CRAY_CC_BIN)
  # FindMPI does not include Cray GTL (e.g. CUDA-aware) libs
  # automatically in Cray environment. Locate it to include in linking.
  string(REPLACE ":" ";" CRAY_LIB_PATHS $ENV{CRAY_LD_LIBRARY_PATH})
  find_library(CRAY_MPI_GTL_CUDA_LIBRARY REQUIRED
    NAMES mpi_gtl_cuda
    HINTS ${CRAY_LIB_PATHS}
  )

  # Cray GTL libs benefit from linking against gdrcopy, so also
  # locating that library.
  find_library(GDRCOPY_LIBRARY REQUIRED
    NAMES gdrapi
  )

  message(STATUS "Found Cray GTL library: " ${CRAY_MPI_GTL_CUDA_LIBRARY})
  message(STATUS "Found GDRCopy library: " ${GDRCOPY_LIBRARY})
endif()

# TODO: Check for MPICH to define `-DMPICH` flag

# HPC SDK
find_package(NVHPC REQUIRED COMPONENTS CUDA MATH)

# Set up required include directory flags, NVHPC CMake config only defined library directories
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUFFT_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUTENSOR_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})

# Get NCCL library (with optional override)
if (CUDECOMP_NCCL_HOME)
  find_path(NCCL_INCLUDE_DIR REQUIRED
    NAMES nccl.h
    HINTS ${CUDECOMP_NCCL_HOME}/include
  )

  find_library(NCCL_LIBRARY REQUIRED
    NAMES nccl
    HINTS ${CUDECOMP_NCCL_HOME}/lib
  )
else()
  find_package(NVHPC REQUIRED COMPONENTS NCCL)
  find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NVHPC_NCCL_LIBRARY_DIR}
  )
  string(REPLACE "/lib" "/include" NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
endif()

message(STATUS "Using NCCL library: ${NCCL_LIBRARY}")

if (CUDECOMP_ENABLE_NVSHMEM)
  # Get NVSHMEM library (with optional override)
  if (CUDECOMP_NVSHMEM_HOME)
    find_path(NVSHMEM_INCLUDE_DIR REQUIRED
      NAMES nvshmem.h
      HINTS ${CUDECOMP_NVSHMEM_HOME}/include
    )

    find_path(NVSHMEM_LIBRARY_DIR REQUIRED
      NAMES libnvshmem.a
      HINTS ${CUDECOMP_NVSHMEM_HOME}/lib
    )
  else()
    find_package(NVHPC REQUIRED COMPONENTS NVSHMEM)
    set(NVSHMEM_LIBRARY_DIR ${NVHPC_NVSHMEM_LIBRARY_DIR})
    string(REPLACE "/lib" "/include" NVSHMEM_INCLUDE_DIR ${NVHPC_NVSHMEM_LIBRARY_DIR})
  endif()

  message(STATUS "Using NVSHMEM installation at: ${NVSHMEM_LIBRARY_DIR}")

endif()

# Building cuDecomp shared lib
add_library(cudecomp SHARED)
set_target_properties(cudecomp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set NVCC flags for requested compute capability
if (CMAKE_VERSION VERSION_LESS 3.18)
  foreach(CUDA_CC ${CUDECOMP_CUDA_CC_LIST})
    list(APPEND CUDA_CC_FLAGS -gencode=arch=compute_${CUDA_CC},code=sm_${CUDA_CC})
  endforeach()
  target_compile_options(cudecomp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_CC_FLAGS}>)
else()
  set_target_properties(cudecomp PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")
endif()

target_compile_options(cudecomp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v >)

target_sources(cudecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autotune.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels_rdc.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp.cc
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels_rdc.cu PROPERTIES COMPILE_FLAGS -rdc=true)
set_target_properties(cudecomp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(cudecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${NVHPC_CUDA_INCLUDE_DIR}
  ${NVHPC_CUTENSOR_INCLUDE_DIR}
  ${NCCL_INCLUDE_DIR}
)

target_link_libraries(cudecomp PUBLIC NVHPC::CUDART)
target_link_libraries(cudecomp PUBLIC MPI::MPI_CXX)
target_link_libraries(cudecomp PRIVATE NVHPC::CUTENSOR)
target_link_libraries(cudecomp PRIVATE ${NCCL_LIBRARY})
if (CRAY_CC_BIN)
  # In Cray environments, add links to GTL and GDRCopy libs for CUDA-aware support
  target_link_libraries(cudecomp PRIVATE ${CRAY_MPI_GTL_CUDA_LIBRARY})
  target_link_libraries(cudecomp PRIVATE ${GDRCOPY_LIBRARY})
endif()

if (CUDECOMP_ENABLE_NVTX)
  target_compile_definitions(cudecomp PRIVATE ENABLE_NVTX)
endif()

if (CUDECOMP_ENABLE_NVSHMEM)
  target_compile_definitions(cudecomp PRIVATE ENABLE_NVSHMEM)
  target_include_directories(cudecomp
    PRIVATE
    ${NVSHMEM_INCLUDE_DIR}
  )

  # Get NVSHMEM version from header
  if (EXISTS ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h)
    file(READ ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h NVSHMEM_VERSION_RAW)
  else()
    file(READ ${NVSHMEM_INCLUDE_DIR}/common/nvshmem_version.h NVSHMEM_VERSION_RAW)
  endif()
  string(REGEX MATCH "NVSHMEM_VENDOR_MAJOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
  list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
  string(REGEX MATCH "NVSHMEM_VENDOR_MINOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
  list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
  list(JOIN NVSHMEM_VERSION "." NVSHMEM_VERSION)

  if (NVSHMEM_VERSION VERSION_LESS "2.7")
    # NVSHMEM versions before 2.7 will  export NCCL symbols erroneously, need to define this flag
    target_compile_definitions(cudecomp PRIVATE NVSHMEM_USE_NCCL)
  endif()

  if (NVSHMEM_VERSION VERSION_LESS "2.5")
    target_link_libraries(cudecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem.a)
  else()
    target_link_libraries(cudecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem_host.so)
    target_link_libraries(cudecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem_device.a)
    target_link_libraries(cudecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lnvidia-ml)
  endif()
  target_link_libraries(cudecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lcuda)
endif()

set_target_properties(cudecomp PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h ${CMAKE_BINARY_DIR}/include/cudecomp.h)

install(
  TARGETS cudecomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# Building Fortran shared lib and module
if (CUDECOMP_BUILD_FORTRAN)
  # Creating -gpu argument string for Fortran files
  foreach(CUDA_CC ${CUDECOMP_CUDA_CC_LIST})
    list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
  endforeach()
  list(APPEND CUF_GPU_ARG "cuda${NVHPC_CUDA_VERSION}")
  list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

  add_library(cudecomp_fort SHARED)
  set_target_properties(cudecomp_fort PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set_target_properties(cudecomp_fort PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
  set_target_properties(cudecomp_fort PROPERTIES LINKER_LANGUAGE Fortran)
  target_compile_options(cudecomp_fort PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
  target_sources(
    cudecomp_fort
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_m.cuf
  )
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_m.cuf PROPERTIES LANGUAGE Fortran)

  target_link_libraries(cudecomp_fort PUBLIC MPI::MPI_Fortran)

  # Test for MPI_Comm_f2c/c2f
  try_compile(
    TEST_F2C_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_mpi_f2c.f90
    LINK_LIBRARIES MPI::MPI_Fortran
  )
  if (NOT TEST_F2C_RESULT)
    message(STATUS "Could not link MPI_Comm_f2c in Fortran module. Setting -DMPICH flag during module compilation.")
    target_compile_definitions(cudecomp_fort PRIVATE MPICH)
  endif()

  install(
    TARGETS cudecomp_fort
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
   )
   # Install cuDecomp module
   install(FILES ${CMAKE_BINARY_DIR}/include/cudecomp.mod DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

if (CUDECOMP_BUILD_EXTRAS)
  add_subdirectory(benchmark)

  add_subdirectory(tests/cc)
  add_subdirectory(examples/cc/basic_usage)
  add_subdirectory(examples/cc/taylor_green)

  if (CUDECOMP_BUILD_FORTRAN)
    add_subdirectory(tests/fortran)
    add_subdirectory(examples/fortran/basic_usage)
    add_subdirectory(examples/fortran/poisson)
  endif()
endif()

