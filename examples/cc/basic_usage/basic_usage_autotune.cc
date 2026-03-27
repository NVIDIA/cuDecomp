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

#include <array>
#include <complex>
#include <cstdio>
#include <numeric>
#include <vector>

#include <getopt.h>

#include <mpi.h>

#include <hip/hip_runtime.h>

#include "hipdecomp.h"

// Error checking macros
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

// CUDA kernel to demonstrate pencil data access on device.
__global__ void initialize_pencil(double* data, hipdecompPencilInfo_t pinfo) {

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

  // Initialize MPI and start up hipDecomp
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  int rank, nranks;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);

  int device_count;
  CHECK_HIP_EXIT(hipGetDeviceCount(&device_count));
  CHECK_HIP_EXIT(hipSetDevice(local_rank % device_count));

  hipdecompHandle_t handle;
  CHECK_HIPDECOMP_EXIT(hipdecompInit(&handle, MPI_COMM_WORLD));

  // Create hipDecomp grid descriptor (with autotuning enabled)
  hipdecompGridDescConfig_t config;
  CHECK_HIPDECOMP_EXIT(hipdecompGridDescConfigSetDefaults(&config));

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
  hipdecompGridDescAutotuneOptions_t options;
  CHECK_HIPDECOMP_EXIT(hipdecompGridDescAutotuneOptionsSetDefaults(&options));

  // General options
  options.n_warmup_trials = 3;
  options.n_trials = 5;
  options.dtype = HIPDECOMP_DOUBLE;
  options.disable_nccl_backends = false;
  options.disable_nvshmem_backends = false;
  options.skip_threshold = 0.0;

  // Process grid autotuning options
  options.grid_mode = HIPDECOMP_AUTOTUNE_GRID_TRANSPOSE;
  options.allow_uneven_decompositions = true;

  // Transpose communication backend autotuning options
  options.autotune_transpose_backend = true;
  options.transpose_use_inplace_buffers[0] = true; // use in-place buffers for X-to-Y transpose
  options.transpose_use_inplace_buffers[1] = true; // use in-place buffers for Y-to-Z transpose
  options.transpose_use_inplace_buffers[2] = true; // use in-place buffers for Z-to-Y transpose
  options.transpose_use_inplace_buffers[3] = true; // use in-place buffers for Y-to-X transpose
  options.transpose_op_weights[0] = 1.0;           // apply 1.0 multiplier to X-to-Y transpose timings
  options.transpose_op_weights[1] = 1.0;           // apply 1.0 multiplier to Y-to-Z transpose timings
  options.transpose_op_weights[2] = 1.0;           // apply 1.0 multiplier to Z-to-Y transpose timings
  options.transpose_op_weights[3] = 1.0;           // apply 1.0 multiplier to Y-to-X transpose timings
  options.transpose_input_halo_extents[0][0] = 1;  // set input_halo_extent to [1, 1, 1] for X-to-Y transpose
  options.transpose_input_halo_extents[0][1] = 1;
  options.transpose_input_halo_extents[0][2] = 1;
  options.transpose_output_halo_extents[3][0] = 1; // set output_halo_extent to [1, 1, 1] for Y-to-X transpose
  options.transpose_output_halo_extents[3][1] = 1;
  options.transpose_output_halo_extents[3][2] = 1;

  // Halo communication backend autotuning options
  options.autotune_halo_backend = true;

  options.halo_axis = 0;

  options.halo_extents[0] = 1;
  options.halo_extents[1] = 1;
  options.halo_extents[2] = 1;

  options.halo_periods[0] = true;
  options.halo_periods[1] = true;
  options.halo_periods[2] = true;

  hipdecompGridDesc_t grid_desc;
  CHECK_HIPDECOMP_EXIT(hipdecompGridDescCreate(handle, &grid_desc, &config, &options));

  // Print information on configuration (updated by autotuner)
  if (rank == 0) {
    printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
    printf("running using %s transpose backend...\n",
           hipdecompTransposeCommBackendToString(config.transpose_comm_backend));
    printf("running using %s halo backend...\n", hipdecompHaloCommBackendToString(config.halo_comm_backend));
  }

  // Allocating pencil memory

  // Get X-pencil information (with halo elements).
  hipdecompPencilInfo_t pinfo_x;
  int32_t halo_extents_x[3]{1, 1, 1};
  CHECK_HIPDECOMP_EXIT(hipdecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, halo_extents_x, nullptr));

  // Get Y-pencil information
  hipdecompPencilInfo_t pinfo_y;
  CHECK_HIPDECOMP_EXIT(hipdecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, nullptr, nullptr));

  // Get Z-pencil information
  hipdecompPencilInfo_t pinfo_z;
  CHECK_HIPDECOMP_EXIT(hipdecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, nullptr, nullptr));

  // Allocate pencil memory
  int64_t data_num_elements = std::max(std::max(pinfo_x.size, pinfo_y.size), pinfo_z.size);

  // Allocate device buffer
  double* data_d;
  CHECK_HIP_EXIT(hipMalloc(&data_d, data_num_elements * sizeof(*data_d)));

  // Allocate host buffer
  double* data = reinterpret_cast<double*>(malloc(data_num_elements * sizeof(*data)));

  // Initializing pencil data (device version using CUDA kernel)
  int threads_per_block = 256;
  int nblocks = (pinfo_x.size + threads_per_block - 1) / threads_per_block;
  initialize_pencil<<<nblocks, threads_per_block>>>(data_d, pinfo_x);

  // Allocating hipDecomp workspace

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_HIPDECOMP_EXIT(hipdecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t halo_work_num_elements;
  CHECK_HIPDECOMP_EXIT(
      hipdecompGetHaloWorkspaceSize(handle, grid_desc, 0, pinfo_x.halo_extents, &halo_work_num_elements));

  // Allocate using hipdecompMalloc
  int64_t dtype_size;
  CHECK_HIPDECOMP_EXIT(hipdecompGetDataTypeSize(HIPDECOMP_DOUBLE, &dtype_size));

  double* transpose_work_d;
  CHECK_HIPDECOMP_EXIT(hipdecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                       transpose_work_num_elements * dtype_size));

  double* halo_work_d;
  CHECK_HIPDECOMP_EXIT(
      hipdecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&halo_work_d), halo_work_num_elements * dtype_size));

  // Transposing data

  // Transpose from X-pencils to Y-pencils.
  CHECK_HIPDECOMP_EXIT(hipdecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE,
                                              pinfo_x.halo_extents, nullptr, nullptr, nullptr, 0));

  // Transpose from Y-pencils to Z-pencils.
  CHECK_HIPDECOMP_EXIT(hipdecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE,
                                              nullptr, nullptr, nullptr, nullptr, 0));

  // Transpose from Z-pencils to Y-pencils.
  CHECK_HIPDECOMP_EXIT(hipdecompTransposeZToY(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE,
                                              nullptr, nullptr, nullptr, nullptr, 0));

  // Transpose from Y-pencils to X-pencils.
  CHECK_HIPDECOMP_EXIT(hipdecompTransposeYToX(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE,
                                              nullptr, pinfo_x.halo_extents, nullptr, nullptr, 0));

  // Updating halos

  // Setting for periodic halos in all directions
  bool halo_periods[3]{true, true, true};

  // Update X-pencil halos in X direction
  CHECK_HIPDECOMP_EXIT(hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE,
                                             pinfo_x.halo_extents, halo_periods, 0, nullptr, 0));

  // Update X-pencil halos in Y direction
  CHECK_HIPDECOMP_EXIT(hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE,
                                             pinfo_x.halo_extents, halo_periods, 1, nullptr, 0));

  // Update X-pencil halos in Z direction
  CHECK_HIPDECOMP_EXIT(hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE,
                                             pinfo_x.halo_extents, halo_periods, 2, nullptr, 0));

  // Cleanup resources
  free(data);
  CHECK_HIP_EXIT(hipFree(data_d));
  CHECK_HIPDECOMP_EXIT(hipdecompFree(handle, grid_desc, transpose_work_d));
  CHECK_HIPDECOMP_EXIT(hipdecompFree(handle, grid_desc, halo_work_d));
  CHECK_HIPDECOMP_EXIT(hipdecompGridDescDestroy(handle, grid_desc));
  CHECK_HIPDECOMP_EXIT(hipdecompFinalize(handle));

  CHECK_MPI_EXIT(MPI_Finalize());
}
