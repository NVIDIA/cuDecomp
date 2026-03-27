/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 The Authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HIPDECOMP_CHECKS_H
#define HIPDECOMP_CHECKS_H

#include <cstdio>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
// ToDo: select hiptensor header through CMake
#if HIP_VERSION_MAJOR < 7
#include <hiptensor/hiptensor.hpp>
#else
#include <hiptensor/hiptensor.h>
#endif
#include <mpi.h>
#include <rccl/rccl.h>

#include "internal/exceptions.h"

// Checks with exception throwing (internal usage)

#define CHECK_HIPDECOMP(call)                                                                                          \
  do {                                                                                                                 \
    hipdecompResult_t err = call;                                                                                      \
    if (HIPDECOMP_RESULT_SUCCESS != err) {                                                                             \
      std::ostringstream os;                                                                                           \
      os << "error code " << err;                                                                                      \
      throw hipdecomp::InternalError(__FILE__, __LINE__, os.str().c_str());                                            \
    }                                                                                                                  \
  } while (false)

#define CHECK_HIP(call)                                                                                                \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (hipSuccess != err) { throw hipdecomp::CudaError(__FILE__, __LINE__, hipGetErrorString(err)); }                 \
  } while (false)

#define CHECK_HIP_DRV(call)                                                                                            \
  do {                                                                                                                 \
    hipError_t err = cuFnTable.pfn_##call;                                                                             \
    if (hipSuccess != err) {                                                                                           \
      const char* error_str;                                                                                           \
      cuFnTable.pfn_cuGetErrorString(err, &error_str);                                                                 \
      throw hipdecomp::CudaError(__FILE__, __LINE__, error_str);                                                       \
    }                                                                                                                  \
  } while (false)

#define CHECK_HIP_LAUNCH()                                                                                             \
  do {                                                                                                                 \
    hipError_t err = hipGetLastError();                                                                                \
    if (hipSuccess != err) { throw hipdecomp::CudaError(__FILE__, __LINE__, hipGetErrorString(err)); }                 \
  } while (false)

#define CHECK_CUTENSOR(call)                                                                                           \
  do {                                                                                                                 \
    hiptensorStatus_t err = call;                                                                                      \
    if (HIPTENSOR_STATUS_SUCCESS != err) {                                                                             \
      throw hipdecomp::CutensorError(__FILE__, __LINE__, hiptensorGetErrorString(err));                                \
    }                                                                                                                  \
  } while (false)

#define CHECK_NCCL(call)                                                                                               \
  do {                                                                                                                 \
    ncclResult_t err = call;                                                                                           \
    if (ncclSuccess != err) { throw hipdecomp::NcclError(__FILE__, __LINE__, ncclGetErrorString(err)); }               \
  } while (false)

#define CHECK_MPI(call)                                                                                                \
  do {                                                                                                                 \
    int err = call;                                                                                                    \
    if (0 != err) {                                                                                                    \
      char error_str[MPI_MAX_ERROR_STRING];                                                                            \
      int len;                                                                                                         \
      MPI_Error_string(err, error_str, &len);                                                                          \
      if (error_str) {                                                                                                 \
        throw hipdecomp::MpiError(__FILE__, __LINE__, error_str);                                                      \
      } else {                                                                                                         \
        std::ostringstream os;                                                                                         \
        os << "error code " << err;                                                                                    \
        throw hipdecomp::MpiError(__FILE__, __LINE__, os.str().c_str());                                               \
      }                                                                                                                \
    }                                                                                                                  \
  } while (false)

// Checks with exit (test usage)
#define CHECK_HIPDECOMP_EXIT(call)                                                                                     \
  do {                                                                                                                 \
    hipdecompResult_t err = call;                                                                                      \
    if (HIPDECOMP_RESULT_SUCCESS != err) {                                                                             \
      fprintf(stderr, "%s:%d HIPDECOMP error. (error code %d)\n", __FILE__, __LINE__, err);                            \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_HIP_EXIT(call)                                                                                           \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (hipSuccess != err) {                                                                                           \
      fprintf(stderr, "%s:%d CUDA error. (%s)\n", __FILE__, __LINE__, hipGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_HIP_LAUNCH_EXIT()                                                                                        \
  do {                                                                                                                 \
    hipError_t err = hipGetLastError();                                                                                \
    if (hipSuccess != err) {                                                                                           \
      fprintf(stderr, "%s:%d CUDA error. (%s)\n", __FILE__, __LINE__, hipGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_MPI_EXIT(call)                                                                                           \
  do {                                                                                                                 \
    int err = call;                                                                                                    \
    if (0 != err) {                                                                                                    \
      char error_str[MPI_MAX_ERROR_STRING];                                                                            \
      int len;                                                                                                         \
      MPI_Error_string(err, error_str, &len);                                                                          \
      if (error_str) {                                                                                                 \
        fprintf(stderr, "%s:%d MPI error. (%s)\n", __FILE__, __LINE__, error_str);                                     \
      } else {                                                                                                         \
        fprintf(stderr, "%s:%d MPI error. (error code %d)\n", __FILE__, __LINE__, err);                                \
      }                                                                                                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_HIPFFT_EXIT(call)                                                                                        \
  do {                                                                                                                 \
    hipfftResult_t err = call;                                                                                         \
    if (HIPFFT_SUCCESS != err) {                                                                                       \
      fprintf(stderr, "%s:%d CUFFT error. (error code %d)\n", __FILE__, __LINE__, err);                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#endif // HIPDECOMP_CHECKS_H
