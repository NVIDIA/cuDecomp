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

#include <array>
#include <complex>
#include <cstdio>
#include <numeric>
#include <vector>

#include <getopt.h>

#include <mpi.h>

#include <cuda_runtime.h>

#include "cudecomp.h"

// Error checking macros
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

// CUDA kernel to demonstrate pencil data access on device.
__global__ void initialize_pencil(double* data, cudecompPencilInfo_t pinfo) {

  int64_t l = blockIdx.x * blockDim.x + threadIdx.x;

  if (l > pinfo.size) return;

  int i = l % pinfo.shape[0];
  int j = l / pinfo.shape[0] % pinfo.shape[1];
  int k = l / (pinfo.shape[0] * pinfo.shape[1]);

  int gx[3];
  gx[pinfo.order[0]] = i + pinfo.lo[0];
  gx[pinfo.order[1]] = j + pinfo.lo[1];
  gx[pinfo.order[2]] = k + pinfo.lo[2];

  gx[pinfo.order[0]] -= pinfo.halo_extents[pinfo.order[0]];
  gx[pinfo.order[1]] -= pinfo.halo_extents[pinfo.order[1]];
  gx[pinfo.order[2]] -= pinfo.halo_extents[pinfo.order[2]];

  data[i] = gx[0] + gx[1] + gx[2];
}

int main(int argc, char** argv) {

  // Initialize MPI and start up cuDecomp
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  int rank, nranks;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank));

  cudecompHandle_t handle;
  CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));

  // Create cuDecomp grid descriptor (with autotuning enabled)
  cudecompGridDescConfig_t config;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));

  // Set pdims entries to 0 to enable process grid autotuning
  config.pdims[0] = 0; // P_rows
  config.pdims[1] = 0; // P_cols

  config.gdims[0] = 64; // X
  config.gdims[1] = 64; // Y
  config.gdims[2] = 64; // Z

  config.transpose_axis_contiguous[0] = true;
  config.transpose_axis_contiguous[1] = true;
  config.transpose_axis_contiguous[2] = true;

  // Set up autotune options structure
  cudecompGridDescAutotuneOptions_t options;
  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));

  // General options
  options.dtype = CUDECOMP_DOUBLE;
  options.disable_nccl_backends = false;
  options.disable_nvshmem_backends = false;

  // Process grid autotuning options
  options.grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;
  options.allow_uneven_decompositions = true;

  // Transpose communication backend autotuning options
  options.autotune_transpose_backend = true;
  options.transpose_use_inplace_buffers = true;

  // Halo communication backend autotuning options
  options.autotune_halo_backend = true;

  options.halo_axis = 0;

  options.halo_extents[0] = 1;
  options.halo_extents[1] = 1;
  options.halo_extents[2] = 1;

  options.halo_periods[0] = true;
  options.halo_periods[1] = true;
  options.halo_periods[2] = true;

  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  // Print information on configuration (updated by autotuner)
  if (rank == 0) {
    printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
    printf("running using %s transpose backend...\n",
           cudecompTransposeCommBackendToString(config.transpose_comm_backend));
    printf("running using %s halo backend...\n", cudecompHaloCommBackendToString(config.halo_comm_backend));
  }

  // Allocating pencil memory

  // Get X-pencil information (with halo elements).
  cudecompPencilInfo_t pinfo_x;
  int32_t halo_extents_x[3]{1, 1, 1};
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, halo_extents_x));

  // Get Y-pencil information
  cudecompPencilInfo_t pinfo_y;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, nullptr));

  // Get Z-pencil information
  cudecompPencilInfo_t pinfo_z;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, nullptr));

  // Allocate pencil memory
  int64_t data_num_elements = std::max(std::max(pinfo_x.size, pinfo_y.size), pinfo_z.size);

  // Allocate device buffer
  double* data_d;
  CHECK_CUDA_EXIT(cudaMalloc(&data_d, data_num_elements * sizeof(*data_d)));

  // Allocate host buffer
  double* data = reinterpret_cast<double*>(malloc(data_num_elements * sizeof(*data)));

  // Initializing pencil data (device version using CUDA kernel)
  int threads_per_block = 256;
  int nblocks = (pinfo_x.size + threads_per_block - 1) / threads_per_block;
  initialize_pencil<<<nblocks, threads_per_block>>>(data_d, pinfo_x);

  // Allocating cuDecomp workspace

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t halo_work_num_elements;
  CHECK_CUDECOMP_EXIT(
      cudecompGetHaloWorkspaceSize(handle, grid_desc, 0, pinfo_x.halo_extents, &halo_work_num_elements));

  // Allocate using cudecompMalloc
  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_DOUBLE, &dtype_size));

  double* transpose_work_d;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                     transpose_work_num_elements * dtype_size));

  double* halo_work_d;
  CHECK_CUDECOMP_EXIT(
      cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&halo_work_d), halo_work_num_elements * dtype_size));

  // Transposing data

  // Transpose from X-pencils to Y-pencils.
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE,
                                            pinfo_x.halo_extents, nullptr, 0));

  // Transpose from Y-pencils to Z-pencils.
  CHECK_CUDECOMP_EXIT(
      cudecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE, nullptr, nullptr, 0));

  // Transpose from Z-pencils to Y-pencils.
  CHECK_CUDECOMP_EXIT(
      cudecompTransposeZToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE, nullptr, nullptr, 0));

  // Transpose from Y-pencils to X-pencils.
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE,
                                            nullptr, pinfo_x.halo_extents, 0));

  // Updating halos

  // Setting for periodic halos in all directions
  bool halo_periods[3]{true, true, true};

  // Update X-pencil halos in X direction
  CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE,
                                           pinfo_x.halo_extents, halo_periods, 0, 0));

  // Update X-pencil halos in Y direction
  CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE,
                                           pinfo_x.halo_extents, halo_periods, 1, 0));

  // Update X-pencil halos in Z direction
  CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE,
                                           pinfo_x.halo_extents, halo_periods, 2, 0));

  // Cleanup resources
  free(data);
  CHECK_CUDA_EXIT(cudaFree(data_d));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, halo_work_d));
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));

  CHECK_MPI_EXIT(MPI_Finalize());
}
