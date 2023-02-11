cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# User-defined build options
option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" ON)
option(CUDECOMP_ENABLE_NVTX "Enable NVTX ranges" ON)
option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)
set(CUDECOMP_CUDA_CC_LIST "60;70;80" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")

if (CUDECOMP_BUILD_FORTRAN)
  set(LANGS CXX CUDA Fortran)
else()
  set(LANGS CXX CUDA)
endif()

project(cudecomp LANGUAGES ${LANGS})

# MPI
find_package(MPI REQUIRED)
# TODO: Check for MPICH to define `-DMPICH` flag

# HPC SDK
if (CUDECOMP_ENABLE_NVSHMEM)
  find_package(NVHPC REQUIRED COMPONENTS CUDA MATH NCCL NVSHMEM)
else()
  find_package(NVHPC REQUIRED COMPONENTS CUDA MATH NCCL)
endif()

# Set up required include directory flags
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUFFT_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUTENSOR_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
if (CUDECOMP_ENABLE_NVSHMEM)
  string(REPLACE "/lib" "/include" NVHPC_NVSHMEM_INCLUDE_DIR ${NVHPC_NVSHMEM_LIBRARY_DIR})
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
target_sources(cudecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autotune.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp.cc
)

target_include_directories(cudecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${NVHPC_CUDA_INCLUDE_DIR}
  ${NVHPC_CUTENSOR_INCLUDE_DIR}
  ${NVHPC_NCCL_INCLUDE_DIR}
)

target_link_libraries(cudecomp PUBLIC NVHPC::CUDART)
target_link_libraries(cudecomp PUBLIC MPI::MPI_CXX)
target_link_libraries(cudecomp PRIVATE NVHPC::CUTENSOR)
target_link_libraries(cudecomp PRIVATE NVHPC::NCCL)

if (CUDECOMP_ENABLE_NVTX)
  target_compile_definitions(cudecomp PRIVATE ENABLE_NVTX)
endif()

if (CUDECOMP_ENABLE_NVSHMEM)
  target_compile_definitions(cudecomp PRIVATE ENABLE_NVSHMEM)

  # Get NVSHMEM version from header
  file(READ ${NVHPC_NVSHMEM_INCLUDE_DIR}/nvshmem_version.h NVSHMEM_VERSION_RAW)
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
    target_link_libraries(cudecomp PRIVATE ${NVHPC_NVSHMEM_LIBRARY_DIR}/libnvshmem.a)
  else()
    target_link_libraries(cudecomp PRIVATE -L${NVHPC_NVSHMEM_LIBRARY_DIR} -lnvshmem_host)
    target_link_libraries(cudecomp PRIVATE ${NVHPC_NVSHMEM_LIBRARY_DIR}/libnvshmem_device.a)
    target_link_libraries(cudecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lnvidia-ml)
  endif()
  target_link_libraries(cudecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lcuda)
  set_target_properties(cudecomp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(cudecomp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
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

  #set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules)

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

