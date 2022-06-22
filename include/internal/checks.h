/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDECOMP_CHECKS_H
#define CUDECOMP_CHECKS_H

#include <cstdio>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cutensor.h>
#include <mpi.h>
#include <nccl.h>

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
