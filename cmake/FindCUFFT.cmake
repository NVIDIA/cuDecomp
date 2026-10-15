# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

include(FindPackageHandleStandardArgs)

# Older FindCUDAToolkit versions can find the CUDA runtime in NVHPC without
# finding cuFFT in the separate, versioned math_libs directory.
if (TARGET CUDA::cufft)
  get_target_property(CUFFT_LIBRARY CUDA::cufft IMPORTED_LOCATION)
  # Older FindCUDAToolkit versions use an interface target without a location.
  if (NOT CUFFT_LIBRARY)
    set(CUFFT_LIBRARY "${CUDA_cufft_LIBRARY}")
  endif()
else()
  set(_CUFFT_NVHPC_ROOT
      "${CUDECOMP_NVHPC_SDK_ROOT}/math_libs/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  if (CUDECOMP_NVHPC_SDK_ROOT AND IS_DIRECTORY "${_CUFFT_NVHPC_ROOT}")
    list(APPEND _CUFFT_PATHS "${_CUFFT_NVHPC_ROOT}")
  endif()
  find_library(CUFFT_LIBRARY
    NAMES cufft
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS ${_CUFFT_PATHS} ENV NVCOMPILER_MATH_LIBS_HOME ENV LIBRARY_PATH ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib64 lib
  )
endif()

if (CUFFT_LIBRARY)
  get_filename_component(_CUFFT_LIBRARY_DIR "${CUFFT_LIBRARY}" DIRECTORY)
  get_filename_component(_CUFFT_ROOT "${_CUFFT_LIBRARY_DIR}" DIRECTORY)
endif()
find_path(CUFFT_INCLUDE_DIR
  NAMES cufft.h
  HINTS "${_CUFFT_ROOT}/include" ${CUDAToolkit_INCLUDE_DIRS}
  PATHS ${_CUFFT_PATHS} ENV NVCOMPILER_MATH_LIBS_HOME ENV CPATH
  PATH_SUFFIXES include
)

find_package_handle_standard_args(CUFFT REQUIRED_VARS CUFFT_LIBRARY CUFFT_INCLUDE_DIR)

if (CUFFT_FOUND AND NOT TARGET CUFFT::CUFFT)
  if (TARGET CUDA::cufft)
    add_library(CUFFT::CUFFT INTERFACE IMPORTED)
    set_target_properties(CUFFT::CUFFT PROPERTIES INTERFACE_LINK_LIBRARIES CUDA::cufft)
  else()
    add_library(CUFFT::CUFFT UNKNOWN IMPORTED)
    set_target_properties(CUFFT::CUFFT PROPERTIES IMPORTED_LOCATION "${CUFFT_LIBRARY}")
  endif()
  set_target_properties(CUFFT::CUFFT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUFFT_INCLUDE_DIR};${CUDAToolkit_INCLUDE_DIRS}"
  )
endif()

mark_as_advanced(CUFFT_LIBRARY CUFFT_INCLUDE_DIR)
unset(_CUFFT_NVHPC_ROOT)
unset(_CUFFT_PATHS)
unset(_CUFFT_LIBRARY_DIR)
unset(_CUFFT_ROOT)
