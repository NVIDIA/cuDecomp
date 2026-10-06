cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# User-defined build options
option(CUDECOMP_BUILD_CPU_ONLY "Build the toolkit-free, MPI-only CPU debug/test variant" OFF)
set(CUDECOMP_NVTX_DEFAULT ON)
if (CUDECOMP_BUILD_CPU_ONLY)
  set(CUDECOMP_NVTX_DEFAULT OFF)
endif()
option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" ON)
option(CUDECOMP_ENABLE_NVTX "Enable NVTX ranges" ${CUDECOMP_NVTX_DEFAULT})
option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
option(CUDECOMP_BUILD_TESTS "Build tests" OFF)
option(CUDECOMP_TEST_FETCH_GTEST "Fetch GoogleTest if a system installation is unavailable" OFF)
option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)
set(CUDECOMP_CUTENSOR_HOME CACHE PATH "CUTENSOR installation root. Overrides CUTENSOR_ROOT when set.")
set(CUDECOMP_NCCL_HOME CACHE PATH "NCCL installation root. Overrides NCCL_ROOT when set.")
set(CUDECOMP_NVSHMEM_HOME CACHE PATH "NVSHMEM installation root. Overrides NVSHMEM_ROOT when set.")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CUDECOMP_CUTENSOR_HOME)
  set(CUTENSOR_ROOT "${CUDECOMP_CUTENSOR_HOME}")
endif()

if (CUDECOMP_NCCL_HOME)
  set(NCCL_ROOT "${CUDECOMP_NCCL_HOME}")
endif()

if (CUDECOMP_NVSHMEM_HOME)
  set(NVSHMEM_ROOT "${CUDECOMP_NVSHMEM_HOME}")
endif()

if (CUDECOMP_BUILD_CPU_ONLY AND (CUDECOMP_ENABLE_NVTX OR CUDECOMP_ENABLE_NVSHMEM))
  message(FATAL_ERROR "CPU-only builds do not support NVTX or NVSHMEM")
endif()

if (NOT CUDECOMP_BUILD_CPU_ONLY)
  # Prefer nvc++ when available unless the user has selected a C++ compiler
  # through CMake, the environment, or a toolchain file.
  if (NOT CMAKE_CXX_COMPILER AND NOT CMAKE_TOOLCHAIN_FILE AND "$ENV{CXX}" STREQUAL "")
    find_program(NVHPC_CXX_BIN "nvc++")
    if (NVHPC_CXX_BIN)
      set(CMAKE_CXX_COMPILER "${NVHPC_CXX_BIN}")
    endif()
  endif()

  # GPU Fortran bindings require nvfortran. Select it when the user has not
  # already chosen a Fortran compiler through CMake or the environment.
  if (CUDECOMP_BUILD_FORTRAN AND NOT CMAKE_Fortran_COMPILER AND NOT CMAKE_TOOLCHAIN_FILE AND "$ENV{FC}" STREQUAL "")
    find_program(NVHPC_Fortran_BIN "nvfortran")
    if (NOT NVHPC_Fortran_BIN)
      message(FATAL_ERROR "nvfortran was not found. Load an NVHPC environment or set CMAKE_Fortran_COMPILER.")
    endif()
    set(CMAKE_Fortran_COMPILER "${NVHPC_Fortran_BIN}")
  endif()

endif()

set(LANGS CXX)
if (NOT CUDECOMP_BUILD_CPU_ONLY)
  list(APPEND LANGS CUDA)
endif()
if (CUDECOMP_BUILD_FORTRAN)
  list(APPEND LANGS Fortran)
endif()

set(CUDECOMP_VERSION_COMPONENTS)
foreach(COMPONENT MAJOR MINOR PATCH)
  file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp_version.h CUDECOMP_VERSION_LINE
       REGEX "^#define CUDECOMP_${COMPONENT} [0-9]+$")
  if (NOT CUDECOMP_VERSION_LINE)
    message(FATAL_ERROR "Failed to read CUDECOMP_${COMPONENT} from include/cudecomp_version.h")
  endif()
  string(REGEX MATCH "[0-9]+$" CUDECOMP_VERSION_COMPONENT ${CUDECOMP_VERSION_LINE})
  list(APPEND CUDECOMP_VERSION_COMPONENTS ${CUDECOMP_VERSION_COMPONENT})
endforeach()
list(JOIN CUDECOMP_VERSION_COMPONENTS "." CUDECOMP_VERSION)

# Increment only when a release breaks binary compatibility.
set(CUDECOMP_SOVERSION 0)

project(cudecomp VERSION ${CUDECOMP_VERSION} LANGUAGES ${LANGS})
include(GNUInstallDirs)

# Locate a standard NVHPC SDK root without requiring its optional environment
# variables. Explicit dependency roots still take precedence in the find modules.
foreach(CUDECOMP_COMPILER IN ITEMS
        "${CMAKE_CXX_COMPILER}"
        "${CMAKE_CUDA_COMPILER}"
        "${CMAKE_Fortran_COMPILER}")
  if (CUDECOMP_COMPILER)
    get_filename_component(CUDECOMP_COMPILER_REALPATH "${CUDECOMP_COMPILER}" REALPATH)
    if (CUDECOMP_COMPILER_REALPATH MATCHES "^(.+)/compilers/bin/[^/]+$")
      set(CUDECOMP_NVHPC_SDK_ROOT "${CMAKE_MATCH_1}")
      break()
    endif()
  endif()
endforeach()
unset(CUDECOMP_COMPILER)
unset(CUDECOMP_COMPILER_REALPATH)

if (CUDECOMP_BUILD_FORTRAN AND NOT CUDECOMP_BUILD_CPU_ONLY AND
    NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  message(FATAL_ERROR "The cuDecomp Fortran bindings require nvfortran.")
endif()

if (CUDECOMP_BUILD_FORTRAN AND NOT CUDECOMP_BUILD_CPU_ONLY)
  # Remove debug information to enable mixed Fortran compilation with cc100 and older compute capabilities.
  string(REGEX REPLACE "(^| )-g(opt)?($| )" "\\3" CMAKE_Fortran_FLAGS_RELWITHDEBINFO
                       "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO}")
endif()

if (NOT CUDECOMP_BUILD_CPU_ONLY)
  # Set up CUDA compute capabilities by CUDA version. Users can override defaults with CUDECOMP_CUDA_CC_LIST
  if (NOT CUDECOMP_CUDA_CC_LIST)
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      set(CUDECOMP_CUDA_CC_LIST_DEFAULTS "75;80;90;100")
    elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set(CUDECOMP_CUDA_CC_LIST_DEFAULTS "70;80;90;100")
    elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 11.8)
      set(CUDECOMP_CUDA_CC_LIST_DEFAULTS "70;80;90")
    else()
      set(CUDECOMP_CUDA_CC_LIST_DEFAULTS "70;80")
    endif()

    set(CUDECOMP_CUDA_CC_LIST ${CUDECOMP_CUDA_CC_LIST_DEFAULTS} CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
  endif()

  # Detect if Cray compiler wrappers are available to assess if in Cray environment.
  # We do not use the Cray compiler wrappers directly for greater flexibility.
  find_program(CRAY_CC_BIN "CC")

  if (CRAY_CC_BIN)
    message(STATUS "Found Cray CC wrapper. Compiling for Cray programming environment.")
  endif()

endif()

# MPI
if (CUDECOMP_BUILD_TESTS)
  # The CTest launcher defaults use MPI_CXX_LIBRARY_VERSION_STRING to identify Open MPI.
  set(MPI_DETERMINE_LIBRARY_VERSION ON)
endif()
find_package(MPI REQUIRED)

if (NOT CUDECOMP_BUILD_CPU_ONLY)
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

# Resolve every CUDA dependency independently of the compiler installation.
# This supports both an unmodified NVHPC SDK and installations assembled from
# components under separate prefixes.
find_package(CUDAToolkit REQUIRED)
find_package(CUTENSOR REQUIRED)

set(CUDECOMP_CUDART_TARGET CUDA::cudart)
set(CUDECOMP_CUTENSOR_TARGET CUTENSOR::CUTENSOR)
set(CUDECOMP_CUFFT_TARGET CUDA::cufft)
get_target_property(CUDECOMP_CUFFT_LIBRARY ${CUDECOMP_CUFFT_TARGET} IMPORTED_LOCATION)
get_filename_component(CUDECOMP_CUFFT_LIBRARY_DIR "${CUDECOMP_CUFFT_LIBRARY}" DIRECTORY)
get_filename_component(CUDECOMP_CUFFT_ROOT "${CUDECOMP_CUFFT_LIBRARY_DIR}" DIRECTORY)
find_path(CUDECOMP_CUFFT_INCLUDE_DIR
  NAMES cufft.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} "${CUDECOMP_CUFFT_ROOT}/include"
  PATHS ENV CPATH
  REQUIRED
)
unset(CUDECOMP_CUFFT_LIBRARY_DIR)
unset(CUDECOMP_CUFFT_ROOT)
set(CUDECOMP_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
set(CUDECOMP_CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
foreach(CUDECOMP_CUDA_INCLUDE_ROOT IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if (EXISTS "${CUDECOMP_CUDA_INCLUDE_ROOT}/cccl")
    list(APPEND CUDECOMP_CUDA_INCLUDE_DIR "${CUDECOMP_CUDA_INCLUDE_ROOT}/cccl")
  endif()
endforeach()
set(CUDECOMP_CUTENSOR_INCLUDE_DIR "${CUTENSOR_INCLUDE_DIR}")
set(CUDECOMP_CUDA_STUBS_DIR "${CUDAToolkit_LIBRARY_DIR}/stubs")

if (CUDECOMP_BUILD_FORTRAN)
  if (NOT "$ENV{NVHPC_CUDA_HOME}" STREQUAL "" AND
      NOT "$ENV{NVCOMPILER_CUDA_HOME}" STREQUAL "")
    get_filename_component(CUDECOMP_NVHPC_CUDA_HOME_REAL "$ENV{NVHPC_CUDA_HOME}" REALPATH)
    get_filename_component(CUDECOMP_NVCOMPILER_CUDA_HOME_REAL "$ENV{NVCOMPILER_CUDA_HOME}" REALPATH)
    if (NOT CUDECOMP_NVHPC_CUDA_HOME_REAL STREQUAL CUDECOMP_NVCOMPILER_CUDA_HOME_REAL)
      message(FATAL_ERROR
        "NVHPC_CUDA_HOME and NVCOMPILER_CUDA_HOME specify different CUDA Toolkit installations. "
        "Set only one override, or set both to the same installation."
      )
    endif()
    set(CUDECOMP_NVFORTRAN_CUDA_HOME "${CUDECOMP_NVHPC_CUDA_HOME_REAL}")
  elseif (NOT "$ENV{NVHPC_CUDA_HOME}" STREQUAL "")
    set(CUDECOMP_NVFORTRAN_CUDA_HOME "$ENV{NVHPC_CUDA_HOME}")
  elseif (NOT "$ENV{NVCOMPILER_CUDA_HOME}" STREQUAL "")
    set(CUDECOMP_NVFORTRAN_CUDA_HOME "$ENV{NVCOMPILER_CUDA_HOME}")
  endif()

  if (CUDECOMP_NVFORTRAN_CUDA_HOME)
    set(CUDECOMP_NVFORTRAN_EXTERNAL_CUDA_OVERRIDE ON)
    set(CUDECOMP_NVFORTRAN_NVCC "${CUDECOMP_NVFORTRAN_CUDA_HOME}/bin/nvcc")
    if (NOT EXISTS "${CUDECOMP_NVFORTRAN_NVCC}")
      message(FATAL_ERROR
        "The NVHPC CUDA Toolkit override does not contain bin/nvcc: "
        "${CUDECOMP_NVFORTRAN_CUDA_HOME}"
      )
    endif()

    execute_process(
      COMMAND "${CUDECOMP_NVFORTRAN_NVCC}" --version
      RESULT_VARIABLE CUDECOMP_NVFORTRAN_NVCC_RESULT
      OUTPUT_VARIABLE CUDECOMP_NVFORTRAN_NVCC_OUTPUT
      ERROR_VARIABLE CUDECOMP_NVFORTRAN_NVCC_ERROR
    )
    if (NOT CUDECOMP_NVFORTRAN_NVCC_RESULT STREQUAL "0")
      message(FATAL_ERROR
        "Could not determine the CUDA version from ${CUDECOMP_NVFORTRAN_NVCC}: "
        "${CUDECOMP_NVFORTRAN_NVCC_ERROR}"
      )
    endif()

    string(REGEX MATCH "release[ \t]+([0-9]+)\\.([0-9]+)"
           CUDECOMP_NVFORTRAN_NVCC_VERSION_MATCH
           "${CUDECOMP_NVFORTRAN_NVCC_OUTPUT}")
    if (NOT CUDECOMP_NVFORTRAN_NVCC_VERSION_MATCH)
      message(FATAL_ERROR
        "Could not determine the CUDA version from ${CUDECOMP_NVFORTRAN_NVCC}."
      )
    endif()
    set(CUDECOMP_NVFORTRAN_CUDA_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    string(REGEX MATCH "V([0-9]+\\.[0-9]+\\.[0-9]+)"
           CUDECOMP_NVFORTRAN_NVCC_FULL_VERSION_MATCH
           "${CUDECOMP_NVFORTRAN_NVCC_OUTPUT}")
    if (CUDECOMP_NVFORTRAN_NVCC_FULL_VERSION_MATCH)
      set(CUDECOMP_NVFORTRAN_CUDA_FULL_VERSION "${CMAKE_MATCH_1}")
    else()
      set(CUDECOMP_NVFORTRAN_CUDA_FULL_VERSION "${CUDECOMP_NVFORTRAN_CUDA_VERSION}")
    endif()

    if (NOT CUDECOMP_NVFORTRAN_CUDA_VERSION STREQUAL CUDECOMP_CUDA_VERSION)
      message(FATAL_ERROR
        "The selected CUDA compiler uses CUDA ${CUDECOMP_CUDA_VERSION}, but the NVHPC CUDA "
        "Toolkit override uses CUDA ${CUDECOMP_NVFORTRAN_CUDA_VERSION}. Set CUDACXX or "
        "CMAKE_CUDA_COMPILER and the NVHPC CUDA override to equivalent Toolkit versions."
      )
    endif()
  else()
    set(CUDECOMP_NVFORTRAN_CUDA_VERSION_ARG "cuda${CUDECOMP_CUDA_VERSION}")

    include(CheckFortranSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-cuda -gpu=${CUDECOMP_NVFORTRAN_CUDA_VERSION_ARG}")
    check_fortran_source_compiles(
      "      program test_nvfortran_cuda
      use cudafor
      implicit none
      end"
      CUDECOMP_NVFORTRAN_CUDA_VERSION_SUPPORTED
    )
    cmake_pop_check_state()

    if (NOT CUDECOMP_NVFORTRAN_CUDA_VERSION_SUPPORTED)
      message(FATAL_ERROR
        "The selected CUDA compiler uses CUDA ${CUDECOMP_CUDA_VERSION}, but nvfortran "
        "${CMAKE_Fortran_COMPILER_VERSION} cannot use that CUDA toolchain. Select a CUDA compiler "
        "matching a CUDA version installed with NVHPC, or set NVHPC_CUDA_HOME or "
        "NVCOMPILER_CUDA_HOME to the selected external CUDA Toolkit."
      )
    endif()
  endif()
endif()

find_package(NCCL MODULE REQUIRED)
set(CUDECOMP_NCCL_TARGET NCCL::NCCL)

if (CUDECOMP_ENABLE_NVSHMEM)
  find_package(NVSHMEM REQUIRED)
  set(CUDECOMP_NVSHMEM_HOST_TARGET NVSHMEM::host)
  set(CUDECOMP_NVSHMEM_DEVICE_TARGET NVSHMEM::device)
  get_filename_component(NVSHMEM_LIBRARY_DIR "${NVSHMEM_HOST_LIBRARY}" DIRECTORY)
endif()

  message(STATUS "cuDecomp dependency summary:")
  message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
  message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
  if (CUDECOMP_BUILD_FORTRAN)
    message(STATUS "  Fortran compiler: ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION})")
    if (CUDECOMP_NVFORTRAN_EXTERNAL_CUDA_OVERRIDE)
      message(STATUS "  NVHPC CUDA Toolkit: ${CUDECOMP_NVFORTRAN_CUDA_HOME} (${CUDECOMP_NVFORTRAN_CUDA_FULL_VERSION})")
    endif()
  endif()
  message(STATUS "  CUDA Toolkit: ${CUDAToolkit_LIBRARY_ROOT} (${CUDAToolkit_VERSION})")
  message(STATUS "  CUTENSOR: ${CUTENSOR_LIBRARY} (${CUTENSOR_VERSION})")
  message(STATUS "  NCCL: ${NCCL_LIBRARY} (${NCCL_VERSION})")
  if (CUDECOMP_ENABLE_NVSHMEM)
    message(STATUS "  NVSHMEM host library: ${NVSHMEM_HOST_LIBRARY}")
    message(STATUS "  NVSHMEM device library: ${NVSHMEM_DEVICE_LIBRARY}")
  endif()
endif()

# Building cuDecomp shared lib
add_library(cudecomp SHARED)
set_target_properties(
  cudecomp
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
             VERSION ${PROJECT_VERSION}
             SOVERSION ${CUDECOMP_SOVERSION}
)

target_sources(cudecomp PRIVATE src/cudecomp.cc)
target_include_directories(cudecomp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(cudecomp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cudecomp PUBLIC MPI::MPI_CXX)
if (CUDECOMP_BUILD_CPU_ONLY)
  target_sources(cudecomp PRIVATE src/cudecomp_kernels_cpu.cc)
else()
  set_target_properties(cudecomp PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")

  target_compile_options(cudecomp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets >)

  target_sources(cudecomp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/autotune.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_wrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels_rdc.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvml_wrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/performance.cc
  )

  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/cudecomp_kernels_rdc.cu PROPERTIES COMPILE_FLAGS -rdc=true)
  set_target_properties(cudecomp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(cudecomp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDECOMP_CUDA_INCLUDE_DIR}
    ${CUDECOMP_CUFFT_INCLUDE_DIR}
    ${CUDECOMP_CUTENSOR_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR}
  )

  target_link_libraries(cudecomp PUBLIC ${CUDECOMP_CUDART_TARGET})
  target_link_libraries(cudecomp PRIVATE ${CUDECOMP_CUTENSOR_TARGET})
  target_link_libraries(cudecomp PRIVATE ${CUDECOMP_NCCL_TARGET})
  if (CRAY_CC_BIN)
    # In Cray environments, add links to GTL and GDRCopy libs for CUDA-aware support
    target_link_libraries(cudecomp PRIVATE ${CRAY_MPI_GTL_CUDA_LIBRARY})
    target_link_libraries(cudecomp PRIVATE ${GDRCOPY_LIBRARY})
  endif()

  # Test for std::filesystem linking
  try_compile(
    TEST_STD_FS_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_std_fs.cc
  )
  if (NOT TEST_STD_FS_RESULT)
    message(STATUS "GCC toolchain version does not link std::filesystem symbols by default. Adding explicit -lstdc++fs link.")
    target_link_libraries(cudecomp PRIVATE -lstdc++fs)
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
    if (EXISTS ${NVSHMEM_INCLUDE_DIR}/non_abi/nvshmem_version.h)
      file(READ ${NVSHMEM_INCLUDE_DIR}/non_abi/nvshmem_version.h NVSHMEM_VERSION_RAW)
    elseif (EXISTS ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h)
      file(READ ${NVSHMEM_INCLUDE_DIR}/nvshmem_version.h NVSHMEM_VERSION_RAW)
    else()
      file(READ ${NVSHMEM_INCLUDE_DIR}/common/nvshmem_version.h NVSHMEM_VERSION_RAW)
    endif()
    string(REGEX MATCH "NVSHMEM_VENDOR_MAJOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
    list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
    string(REGEX MATCH "NVSHMEM_VENDOR_MINOR_VERSION ([0-9]*)" _ ${NVSHMEM_VERSION_RAW})
    list(APPEND NVSHMEM_VERSION ${CMAKE_MATCH_1})
    list(JOIN NVSHMEM_VERSION "." NVSHMEM_VERSION)

    if (NVSHMEM_VERSION VERSION_LESS "2.6")
      message(FATAL_ERROR "NVSHMEM versions earlier than 2.6.0 are not supported by cuDecomp.")
    else()
      target_link_libraries(cudecomp PRIVATE ${CUDECOMP_NVSHMEM_HOST_TARGET})
      target_link_libraries(cudecomp PRIVATE ${CUDECOMP_NVSHMEM_DEVICE_TARGET})
    endif()

    if (NVSHMEM_VERSION VERSION_LESS "2.7")
      # NVSHMEM versions before 2.7 will export NCCL symbols erroneously, need to define this flag
      target_compile_definitions(cudecomp PRIVATE NVSHMEM_USE_NCCL)
      target_link_libraries(cudecomp PUBLIC -L${CUDECOMP_CUDA_STUBS_DIR} -lcuda)
    endif()

    if (NVSHMEM_VERSION VERSION_LESS "2.11")
      target_link_libraries(cudecomp PUBLIC -L${CUDECOMP_CUDA_STUBS_DIR} -lnvidia-ml)
    endif()
  endif()

endif()

configure_file(include/cudecomp_config.h.in ${CMAKE_BINARY_DIR}/include/cudecomp_config.h)
set_target_properties(
  cudecomp
  PROPERTIES PUBLIC_HEADER
             "${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h;${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp_version.h;${CMAKE_BINARY_DIR}/include/cudecomp_config.h"
)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp.h ${CMAKE_BINARY_DIR}/include/cudecomp.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/cudecomp_version.h ${CMAKE_BINARY_DIR}/include/cudecomp_version.h)

install(
  TARGETS cudecomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Building Fortran shared lib and module
if (CUDECOMP_BUILD_FORTRAN)
  if (NOT CUDECOMP_BUILD_CPU_ONLY)
    # Creating -gpu argument string for Fortran files
    foreach(CUDA_CC ${CUDECOMP_CUDA_CC_LIST})
      list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
    endforeach()
    if (CUDECOMP_NVFORTRAN_CUDA_VERSION_ARG)
      list(APPEND CUF_GPU_ARG "${CUDECOMP_NVFORTRAN_CUDA_VERSION_ARG}")
    endif()
    list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)
  endif()

  add_library(cudecomp_fort SHARED)
  set_target_properties(
    cudecomp_fort
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
               Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include
               LINKER_LANGUAGE Fortran
               VERSION ${PROJECT_VERSION}
               SOVERSION ${CUDECOMP_SOVERSION}
  )
  if (CUDECOMP_BUILD_CPU_ONLY)
    # A standard extension lets GNU Fortran preprocess the shared binding source.
    configure_file(src/cudecomp_m.cuf ${CMAKE_CURRENT_BINARY_DIR}/cudecomp_m.F90 COPYONLY)
    target_sources(cudecomp_fort PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/cudecomp_m.F90)
    target_compile_definitions(cudecomp_fort PRIVATE CUDECOMP_BUILD_CPU_ONLY=1)
    target_compile_options(cudecomp_fort PRIVATE $<$<Fortran_COMPILER_ID:GNU>:-ffree-line-length-none>)
  else()
    target_compile_options(cudecomp_fort PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)
    target_sources(cudecomp_fort PRIVATE src/cudecomp_m.cuf)
    set_source_files_properties(src/cudecomp_m.cuf PROPERTIES LANGUAGE Fortran)
  endif()

  target_link_libraries(cudecomp_fort PUBLIC MPI::MPI_Fortran)

  install(
    TARGETS cudecomp_fort
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   )
   # Install cuDecomp modules
   install(
     FILES ${CMAKE_BINARY_DIR}/include/cudecomp.mod
           ${CMAKE_BINARY_DIR}/include/cudecomp_internal.mod
     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
   )
endif()

if (CUDECOMP_BUILD_TESTS OR CUDECOMP_BUILD_EXTRAS)
  enable_testing()
  if (NOT CUDECOMP_BUILD_CPU_ONLY)
    add_subdirectory(tests/cc)

    if (CUDECOMP_BUILD_FORTRAN)
      add_subdirectory(tests/fortran)
    endif()

  endif()

  if (CUDECOMP_BUILD_TESTS)
    add_subdirectory(tests/ctest)
  endif()
endif()

if (CUDECOMP_BUILD_EXTRAS AND NOT CUDECOMP_BUILD_CPU_ONLY)
  add_subdirectory(examples/cc/basic_usage)
  add_subdirectory(benchmark)
  add_subdirectory(examples/cc/taylor_green)

  if (CUDECOMP_BUILD_FORTRAN)
    add_subdirectory(examples/fortran/basic_usage)
    add_subdirectory(examples/fortran/poisson)
    add_subdirectory(examples/fortran/taylor_green)
  endif()
endif()
