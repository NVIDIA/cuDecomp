cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# User-defined build options
option(HIPDECOMP_BUILD_FORTRAN "Build Fortran bindings" ON)
#option(HIPDECOMP_ENABLE_NVTX "Enable NVTX ranges" ON)
#option(HIPDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
#option(HIPDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)
#set(HIPDECOMP_NCCL_HOME CACHE STRING "Path to search for NCCL installation. Use to override NVHPC provided NCCL version.")
#set(HIPDECOMP_NVSHMEM_HOME CACHE STRING "Path to search for NVSHMEM installation. Use to override NVHPC provided NVSHMEM version.")

# Use AMD Clang compilers by default
find_program(CMAKE_CXX_COMPILER "amdclang++" REQUIRED)

# if (HIPDECOMP_BUILD_FORTRAN)
#   find_program(NVHPC_Fortran_BIN "nvfortran" REQUIRED)
#   set(CMAKE_Fortran_COMPILER ${NVHPC_Fortran_BIN})
# endif()

# Locate and use HIP CMake configuration
find_program(HIPCC "hipcc" REQUIRED)
file(REAL_PATH ${HIPCC} HIPCC)
string(REPLACE "bin/hipcc" "lib/cmake" HIP_CMAKE_DIR ${HIPCC})
set(CMAKE_PREFIX_PATH ${HIP_CMAKE_DIR})

# Limit optimization levels of CPU code compilation
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")
set(CMAKE_Fortran_FLAGS_RELEASE "-O1 -DNDEBUG")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")

if (HIPDECOMP_BUILD_FORTRAN)
  set(LANGS CXX HIP Fortran)
else()
  set(LANGS CXX HIP)
endif()

project(HIPDECOMP LANGUAGES ${LANGS})

# HIP
find_package(HIP REQUIRED)

# Set up HIP compute capabilities. Users can override defaults with HIPDECOMP_HIP_ARCH_LIST
if (NOT HIPDECOMP_HIP_ARCH_LIST)
  if (${HIP_VERSION} VERSION_GREATER_EQUAL 7.3.0)
    set(HIPDECOMP_HIP_ARCH_LIST_DEFAULTS "gfx1100")
  else()
    set(HIPDECOMP_HIP_ARCH_LIST_DEFAULTS "gfx1100")
  endif()

  set(HIPDECOMP_HIP_ARCH_LIST ${HIPDECOMP_HIP_ARCH_LIST_DEFAULTS} CACHE STRING "List of CUDA compute capabilities to build HIPDECOMP for.")
endif()

# Detect if Cray compiler wrappers are available to assess if in Cray environment.
# We do not use the Cray compiler wrappers directly for greater flexibility.
# find_program(CRAY_CC_BIN "CC")

# if (CRAY_CC_BIN)
#   message(STATUS "Found Cray CC wrapper. Compiling for Cray programming environment.")
# endif()

# MPI
find_package(MPI REQUIRED)

# if (CRAY_CC_BIN)
#   # FindMPI does not include Cray GTL (e.g. CUDA-aware) libs
#   # automatically in Cray environment. Locate it to include in linking.
#   string(REPLACE ":" ";" CRAY_LIB_PATHS $ENV{CRAY_LD_LIBRARY_PATH})
#   find_library(CRAY_MPI_GTL_CUDA_LIBRARY REQUIRED
#     NAMES mpi_gtl_cuda
#     HINTS ${CRAY_LIB_PATHS}
#   )

#   # Cray GTL libs benefit from linking against gdrcopy, so also
#   # locating that library.
#   find_library(GDRCOPY_LIBRARY REQUIRED
#     NAMES gdrapi
#   )

#   message(STATUS "Found Cray GTL library: " ${CRAY_MPI_GTL_CUDA_LIBRARY})
#   message(STATUS "Found GDRCopy library: " ${GDRCOPY_LIBRARY})
# endif()

# Get RCCL library
find_package(RCCL REQUIRED)

if (HIPDECOMP_ENABLE_ROCSHMEM)
  # Get ROCSHMEM library
  find_package(ROCSHMEM REQUIRED)
endif()

# Building HIPDECOMP shared lib
add_library(hipdecomp SHARED)
set_target_properties(hipdecomp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set compiler flags for requested compute capability
set_target_properties(hipdecomp PROPERTIES HIP_ARCHITECTURES "${HIPDECOMP_HIP_ARCH_LIST}")

target_sources(hipdecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autotune.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_wrap.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels_rdc.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nvml_wrap.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/performance.cc
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/hipdecomp_kernels_rdc.cu PROPERTIES COMPILE_FLAGS -fgpu-rdc)

target_include_directories(hipdecomp
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(hipdecomp PUBLIC hip::host hip::device)
target_link_libraries(hipdecomp PUBLIC MPI::MPI_CXX)
target_link_libraries(hipdecomp PRIVATE hiptensor)
target_link_libraries(hipdecomp PRIVATE rccl)
# if (CRAY_CC_BIN)
#   # In Cray environments, add links to GTL and GDRCopy libs for CUDA-aware support
#   target_link_libraries(hipdecomp PRIVATE ${CRAY_MPI_GTL_CUDA_LIBRARY})
#   target_link_libraries(hipdecomp PRIVATE ${GDRCOPY_LIBRARY})
# endif()

# Test for std::filesystem linking
try_compile(
  TEST_STD_FS_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_std_fs.cc
)
if (NOT TEST_STD_FS_RESULT)
  message(STATUS "GCC toolchain version does not link std::filesystem symbols by default. Adding explicit -lstdc++fs link.")
  target_link_libraries(HIPDECOMP PRIVATE -lstdc++fs)
endif()

# if (HIPDECOMP_ENABLE_NVTX)
#   target_compile_definitions(hipdecomp PRIVATE ENABLE_NVTX)
# endif()

# if (HIPDECOMP_ENABLE_NVSHMEM)
#   target_compile_definitions(hipdecomp PRIVATE ENABLE_NVSHMEM)
#   target_include_directories(hipdecomp
#     PRIVATE
#     ${NVSHMEM_INCLUDE_DIR}
#   )

#   # Get NVSHMEM version from header
#   if (EXISTS ${NVSHMEM_INCLUDE_DIR}/non_abi/nvshmem_version.h)
#     file(READ ${NVSHMEM_INCLUDE_DIR}/non_abi/nvshmem_version.h NVSHMEM_VERSION_RAW)
#   elseif (EXISTS ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h)
#     file(READ ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h NVSHMEM_VERSION_RAW)
#   else()
#     file(READ ${NVSHMEM_INCLUDE_DIR}/common/nvshmem_version.h NVSHMEM_VERSION_RAW)
#   endif()
#   string(REGEX MATCH "NVSHMEM_VENDOR_MAJOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
#   list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
#   string(REGEX MATCH "NVSHMEM_VENDOR_MINOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
#   list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
#   list(JOIN NVSHMEM_VERSION "." NVSHMEM_VERSION)

#   if (NVSHMEM_VERSION VERSION_LESS "2.5")
#     target_link_libraries(hipdecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem.a)
#   else()
#     target_link_libraries(hipdecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem_host.so)
#     target_link_libraries(hipdecomp PRIVATE ${NVSHMEM_LIBRARY_DIR}/libnvshmem_device.a)
#   endif()

#   if (NVSHMEM_VERSION VERSION_LESS "2.7")
#     # NVSHMEM versions before 2.7 will export NCCL symbols erroneously, need to define this flag
#     target_compile_definitions(hipdecomp PRIVATE NVSHMEM_USE_NCCL)
#     target_link_libraries(hipdecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lcuda)
#   endif()

#   if (NVSHMEM_VERSION VERSION_LESS "2.11")
#     target_link_libraries(hipdecomp PUBLIC -L${NVHPC_CUDA_LIBRARY_DIR}/stubs -lnvidia-ml)
#   endif()
# endif()

set_target_properties(hipdecomp PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h ${CMAKE_BINARY_DIR}/include/cudecomp.h)

install(
  TARGETS hipdecomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# Building Fortran shared lib and module
# if (HIPDECOMP_BUILD_FORTRAN)
#   # Creating -gpu argument string for Fortran files
#   foreach(CUDA_CC ${HIPDECOMP_CUDA_CC_LIST})
#     list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
#   endforeach()
#   list(APPEND CUF_GPU_ARG "cuda${NVHPC_CUDA_VERSION}")
#   list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

#   add_library(hipdecomp_fort SHARED)
#   set_target_properties(hipdecomp_fort PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#   set_target_properties(hipdecomp_fort PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
#   set_target_properties(hipdecomp_fort PROPERTIES LINKER_LANGUAGE Fortran)
#   target_compile_options(hipdecomp_fort PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
#   target_sources(
#     hipdecomp_fort
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/hipdecomp_m.cuf
#   )
#   set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/hipdecomp_m.cuf PROPERTIES LANGUAGE Fortran)

#   target_link_libraries(hipdecomp_fort PUBLIC MPI::MPI_Fortran)

#   install(
#     TARGETS hipdecomp_fort
#     LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#    )
#    # Install hipDecomp module
#    install(FILES ${CMAKE_BINARY_DIR}/include/hipdecomp.mod DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
# endif()

# if (HIPDECOMP_BUILD_EXTRAS)
#   add_subdirectory(benchmark)

#   add_subdirectory(tests/cc)
#   add_subdirectory(examples/cc/basic_usage)
#   add_subdirectory(examples/cc/taylor_green)

#   if (HIPDECOMP_BUILD_FORTRAN)
#     add_subdirectory(tests/fortran)
#     add_subdirectory(examples/fortran/basic_usage)
#     add_subdirectory(examples/fortran/poisson)
#     add_subdirectory(examples/fortran/taylor_green)
#   endif()
# endif()
