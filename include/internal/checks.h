/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDECOMP_CHECKS_H
#define CUDECOMP_CHECKS_H

#include <cstdio>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cutensor.h>
#include <mpi.h>
#include <nccl.h>
#include <nvml.h>

#include "internal/exceptions.h"

// Checks with exception throwing (internal usage)

#define CHECK_CUDECOMP(call)                                                                                           \
  do {                                                                                                                 \
    cudecompResult_t err = call;                                                                                       \
    if (CUDECOMP_RESULT_SUCCESS != err) {                                                                              \
      std::ostringstream os;                                                                                           \
      os << "error code " << err;                                                                                      \
      throw cudecomp::InternalError(__FILE__, __LINE__, os.str().c_str());                                             \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (cudaSuccess != err) { throw cudecomp::CudaError(__FILE__, __LINE__, cudaGetErrorString(err)); }                \
  } while (false)

#define CHECK_CUDA_DRV(call)                                                                                           \
  do {                                                                                                                 \
    CUresult err = cuFnTable.pfn_##call;                                                                               \
    if (CUDA_SUCCESS != err) {                                                                                         \
      const char* error_str;                                                                                           \
      cuFnTable.pfn_cuGetErrorString(err, &error_str);                                                                 \
      throw cudecomp::CudaError(__FILE__, __LINE__, error_str);                                                        \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA_LAUNCH()                                                                                            \
  do {                                                                                                                 \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (cudaSuccess != err) { throw cudecomp::CudaError(__FILE__, __LINE__, cudaGetErrorString(err)); }                \
  } while (false)

#define CHECK_CUTENSOR(call)                                                                                           \
  do {                                                                                                                 \
    cutensorStatus_t err = call;                                                                                       \
    if (CUTENSOR_STATUS_SUCCESS != err) {                                                                              \
      throw cudecomp::CutensorError(__FILE__, __LINE__, cutensorGetErrorString(err));                                  \
    }                                                                                                                  \
  } while (false)

#define CHECK_NCCL(call)                                                                                               \
  do {                                                                                                                 \
    ncclResult_t err = call;                                                                                           \
    if (ncclSuccess != err) {                                                                                          \
      std::ostringstream os;                                                                                           \
      os << "error code " << err;                                                                                      \
      throw cudecomp::NcclError(__FILE__, __LINE__, os.str().c_str());                                                 \
    }                                                                                                                  \
  } while (false)

#define CHECK_MPI(call)                                                                                                \
  do {                                                                                                                 \
    int err = call;                                                                                                    \
    if (0 != err) {                                                                                                    \
      char error_str[MPI_MAX_ERROR_STRING];                                                                            \
      int len;                                                                                                         \
      MPI_Error_string(err, error_str, &len);                                                                          \
      if (error_str) {                                                                                                 \
        throw cudecomp::MpiError(__FILE__, __LINE__, error_str);                                                       \
      } else {                                                                                                         \
        std::ostringstream os;                                                                                         \
        os << "error code " << err;                                                                                    \
        throw cudecomp::MpiError(__FILE__, __LINE__, os.str().c_str());                                                \
      }                                                                                                                \
    }                                                                                                                  \
  } while (false)

#define CHECK_NVML(call)                                                                                               \
  do {                                                                                                                 \
    nvmlReturn_t err = nvmlFnTable.pfn_##call;                                                                         \
    if (NVML_SUCCESS != err) { throw cudecomp::NvmlError(__FILE__, __LINE__, nvmlFnTable.pfn_nvmlErrorString(err)); }  \
  } while (false)

// Checks with exit (test usage)
#define CHECK_CUDECOMP_EXIT(call)                                                                                      \
  do {                                                                                                                 \
    cudecompResult_t err = call;                                                                                       \
    if (CUDECOMP_RESULT_SUCCESS != err) {                                                                              \
      fprintf(stderr, "%s:%d CUDECOMP error. (error code %d)\n", __FILE__, __LINE__, err);                             \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA_EXIT(call)                                                                                          \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (cudaSuccess != err) {                                                                                          \
      fprintf(stderr, "%s:%d CUDA error. (%s)\n", __FILE__, __LINE__, cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA_LAUNCH_EXIT()                                                                                       \
  do {                                                                                                                 \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (cudaSuccess != err) {                                                                                          \
      fprintf(stderr, "%s:%d CUDA error. (%s)\n", __FILE__, __LINE__, cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_MPI_EXIT(call)                                                                                           \
  {                                                                                                                    \
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
  }                                                                                                                    \
  while (false)

#define CHECK_CUFFT_EXIT(call)                                                                                         \
  do {                                                                                                                 \
    cufftResult_t err = call;                                                                                          \
    if (CUFFT_SUCCESS != err) {                                                                                        \
      fprintf(stderr, "%s:%d CUFFT error. (error code %d)\n", __FILE__, __LINE__, err);                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#endif // CUDECOMP_CHECKS_H
