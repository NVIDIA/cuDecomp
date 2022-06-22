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
#include "internal/checks.h"

#if defined(R32)
using real_t = float;
static cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
#elif defined(R64)
using real_t = double;
static cudecompDataType_t get_cudecomp_datatype(double) { return CUDECOMP_DOUBLE; }
#elif defined(C32)
using real_t = std::complex<float>;
static cudecompDataType_t get_cudecomp_datatype(std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
#elif defined(C64)
using real_t = std::complex<double>;
static cudecompDataType_t get_cudecomp_datatype(std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }
#else
using real_t = float;
static cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
#endif

static bool compare_pencils(const std::vector<real_t>& ref, const std::vector<real_t>& res,
                            const cudecompPencilInfo_t& pinfo) {
  int64_t lx[3];
  for (int64_t i = 0; i < pinfo.size; ++i) {
    // Compute pencil local coordinate
    lx[0] = i % pinfo.shape[0];
    lx[1] = i / pinfo.shape[0] % pinfo.shape[1];
    lx[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    // Only compare values inside internal region
    if (lx[0] >= pinfo.halo_extents[pinfo.order[0]] && lx[0] < (pinfo.shape[0] - pinfo.halo_extents[pinfo.order[0]]) &&
        lx[1] >= pinfo.halo_extents[pinfo.order[1]] && lx[1] < (pinfo.shape[1] - pinfo.halo_extents[pinfo.order[1]]) &&
        lx[2] >= pinfo.halo_extents[pinfo.order[2]] && lx[2] < (pinfo.shape[2] - pinfo.halo_extents[pinfo.order[2]])) {
      if (std::abs(ref[i] - res[i]) != 0) return false;
    }
  }
  return true;
}

static void initialize_pencil(std::vector<real_t>& ref, const cudecompPencilInfo_t& pinfo,
                              const std::array<int32_t, 3>& gdims) {
  int64_t lx[3];
  int64_t gx[3];
  for (int64_t i = 0; i < pinfo.size; ++i) {
    // Compute pencil local coordinate
    lx[0] = i % pinfo.shape[0];
    lx[1] = i / pinfo.shape[0] % pinfo.shape[1];
    lx[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    // Compute global grid coordinate
    gx[pinfo.order[0]] = lx[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]];
    gx[pinfo.order[1]] = lx[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]];
    gx[pinfo.order[2]] = lx[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]];
    int64_t gi = gx[0] + gdims[0] * (gx[1] + gx[2] * gdims[1]);

    // Only set values inside internal region
    if (lx[0] >= pinfo.halo_extents[pinfo.order[0]] && lx[0] < (pinfo.shape[0] - pinfo.halo_extents[pinfo.order[0]]) &&
        lx[1] >= pinfo.halo_extents[pinfo.order[1]] && lx[1] < (pinfo.shape[1] - pinfo.halo_extents[pinfo.order[1]]) &&
        lx[2] >= pinfo.halo_extents[pinfo.order[2]] && lx[2] < (pinfo.shape[2] - pinfo.halo_extents[pinfo.order[2]])) {
      ref[i] = gi;
    } else {
      ref[i] = -1;
    }
  }
}

static void usage(const char* pname) {

  const char* bname = rindex(pname, '/');
  if (!bname) {
    bname = pname;
  } else {
    bname++;
  }

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t--gx\n"
          "\t\tX-dimension of grid. (default: 256) \n"
          "\t--gy\n"
          "\t\tY-dimension of grid. (default: 256) \n"
          "\t--gz\n"
          "\t\tZ-dimension of grid. (default: 256) \n"
          "\t--pr\n"
          "\t\tRow dimension of process grid. (default: 0, autotune) \n"
          "\t--pc\n"
          "\t\tColumn dimension of process grid. (default: 0, autotune) \n"
          "\t--backend"
          "\t\tTranspose communication backend (default: 0, autotune) \n"
          "\t--acx\n"
          "\t\tX-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t--acy\n"
          "\t\tY-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t--acz\n"
          "\t\tZ-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t--gdx\n"
          "\t\tX-dimension gdim_dist setting, set to gx - gdx. (default: 0) \n"
          "\t--gdy\n"
          "\t\tY-dimension gdim_dist setting, set to gy - gdy. (default: 0) \n"
          "\t--gdz\n"
          "\t\tZ-dimension gdim_dist setting, set to gz - gdz. (default: 0) \n"
          "\t--hex\n"
          "\t\tX-dimension halo_extents setting. (default: 0) \n"
          "\t--hey\n"
          "\t\tY-dimension halo_extents setting. (default: 0) \n"
          "\t--hez\n"
          "\t\tZ-dimension halo_extents setting. (default: 0) \n"
          "\t-m|--use-managed-memory\n"
          "\t\tFlag to test operation with managed memory.\n"
          "\t-o|--out-of-place\n"
          "\t\tFlag to test out of place operation.\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  int rank, nranks;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank));

  // Parse command-line arguments
  int gx = 256;
  int gy = 256;
  int gz = 256;
  int pr = 0;
  int pc = 0;
  cudecompTransposeCommBackend_t comm_backend = static_cast<cudecompTransposeCommBackend_t>(0);
  std::array<bool, 3> axis_contiguous{};
  std::array<int, 3> gdims_dist{};
  std::array<int, 3> halo_extents{};
  bool out_of_place = false;
  bool use_managed_memory = false;

  while (1) {
    static struct option long_options[] = {{"gx", required_argument, 0, 'x'},
                                           {"gy", required_argument, 0, 'y'},
                                           {"gz", required_argument, 0, 'z'},
                                           {"backend", required_argument, 0, 'b'},
                                           {"pr", required_argument, 0, 'r'},
                                           {"pc", required_argument, 0, 'c'},
                                           {"acx", required_argument, 0, '1'},
                                           {"acy", required_argument, 0, '2'},
                                           {"acz", required_argument, 0, '3'},
                                           {"gdx", required_argument, 0, '4'},
                                           {"gdy", required_argument, 0, '5'},
                                           {"gdz", required_argument, 0, '6'},
                                           {"hex", required_argument, 0, '7'},
                                           {"hey", required_argument, 0, '8'},
                                           {"hez", required_argument, 0, '9'},
                                           {"out-of-place", no_argument, 0, 'o'},
                                           {"use-managed-memory", no_argument, 0, 'm'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:z:b:r:c:1:2:3:4:5:6:7:8:9:omh", long_options, &option_index);
    if (ch == -1) break;

    switch (ch) {
    case 0: break;
    case 'x': gx = atoi(optarg); break;
    case 'y': gy = atoi(optarg); break;
    case 'z': gz = atoi(optarg); break;
    case 'b': comm_backend = static_cast<cudecompTransposeCommBackend_t>(atoi(optarg)); break;
    case 'r': pr = atoi(optarg); break;
    case 'c': pc = atoi(optarg); break;
    case '1': axis_contiguous[0] = atoi(optarg); break;
    case '2': axis_contiguous[1] = atoi(optarg); break;
    case '3': axis_contiguous[2] = atoi(optarg); break;
    case '4': gdims_dist[0] = atoi(optarg); break;
    case '5': gdims_dist[1] = atoi(optarg); break;
    case '6': gdims_dist[2] = atoi(optarg); break;
    case '7': halo_extents[0] = atoi(optarg); break;
    case '8': halo_extents[1] = atoi(optarg); break;
    case '9': halo_extents[2] = atoi(optarg); break;
    case 'o': out_of_place = true; break;
    case 'm': use_managed_memory = true; break;
    case 'h': usage(argv[0]); break;
    case '?': exit(EXIT_FAILURE);
    default: fprintf(stderr, "unknown option: %c\n", ch); exit(EXIT_FAILURE);
    }
  }

  // Finish setting up gdim_dist
  gdims_dist[0] = gx - gdims_dist[0];
  gdims_dist[1] = gy - gdims_dist[1];
  gdims_dist[2] = gz - gdims_dist[2];

  std::array<int, 2> pdims;
  pdims[0] = pr;
  pdims[1] = pc;

  if (rank == 0) printf("running on %d x %d x %d spatial grid...\n", gx, gy, gz);

  // Initialize cuDecomp
  cudecompHandle_t handle;
  CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));

  // Setup grid descriptor
  std::array<int32_t, 3> gdims{gx, gy, gz};
  cudecompGridDesc_t grid_desc;
  cudecompGridDescConfig_t config;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));
  config.pdims[0] = pdims[0];
  config.pdims[1] = pdims[1];
  config.gdims[0] = gdims[0];
  config.gdims[1] = gdims[1];
  config.gdims[2] = gdims[2];
  config.gdims_dist[0] = gdims_dist[0];
  config.gdims_dist[1] = gdims_dist[1];
  config.gdims_dist[2] = gdims_dist[2];
  config.transpose_axis_contiguous[0] = axis_contiguous[0];
  config.transpose_axis_contiguous[1] = axis_contiguous[1];
  config.transpose_axis_contiguous[2] = axis_contiguous[2];

  cudecompGridDescAutotuneOptions_t options;
  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));
  options.dtype = get_cudecomp_datatype(real_t(0));
  options.transpose_use_inplace_buffers = !out_of_place;

  if (comm_backend != 0) {
    config.transpose_comm_backend = comm_backend;
  } else {
    options.autotune_transpose_backend = true;
  }

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  if (rank == 0) {
    printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
    printf("running using %s transpose backend...\n",
           cudecompTransposeCommBackendToString(config.transpose_comm_backend));
  }

  // Get x-pencil information
  cudecompPencilInfo_t pinfo_x;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, halo_extents.data()));

  // Get y-pencil information
  cudecompPencilInfo_t pinfo_y;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, halo_extents.data()));

  // Get z-pencil information
  cudecompPencilInfo_t pinfo_z;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, halo_extents.data()));

  // Get workspace size
  int64_t workspace_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &workspace_num_elements));

  // Allocate data arrays
  int64_t data_num_elements = std::max(std::max(pinfo_x.size, pinfo_y.size), pinfo_z.size);

  std::vector<real_t> data(data_num_elements);

  // Create reference data
  std::vector<real_t> xref(pinfo_x.size);
  std::vector<real_t> yref(pinfo_y.size);
  std::vector<real_t> zref(pinfo_z.size);

  initialize_pencil(xref, pinfo_x, gdims);
  initialize_pencil(yref, pinfo_y, gdims);
  initialize_pencil(zref, pinfo_z, gdims);

  real_t *data_d, *work_d;
  if (use_managed_memory) {
    CHECK_CUDA_EXIT(cudaMallocManaged(&data_d, data.size() * sizeof(*data_d)));
  } else {
    CHECK_CUDA_EXIT(cudaMalloc(&data_d, data.size() * sizeof(*data_d)));
  }
  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(get_cudecomp_datatype(real_t(0)), &dtype_size));
  CHECK_CUDECOMP_EXIT(
      cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_d), workspace_num_elements * dtype_size));

  real_t* data_2_d = nullptr;
  if (out_of_place) {
    if (use_managed_memory) {
      CHECK_CUDA_EXIT(cudaMallocManaged(&data_2_d, data.size() * sizeof(*data_2_d)));
    } else {
      CHECK_CUDA_EXIT(cudaMalloc(&data_2_d, data.size() * sizeof(*data_2_d)));
    }
  }

  // Running correctness tests
  if (rank == 0) printf("running correctness tests...\n");

  // Initialize data to reference x-pencil data
  CHECK_CUDA_EXIT(cudaMemcpy(data_d, xref.data(), xref.size() * sizeof(*data_d), cudaMemcpyHostToDevice));

  real_t* input = data_d;
  real_t* output = data_d;
  if (out_of_place) output = data_2_d;

  CHECK_CUDA_EXIT(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                            pinfo_x.halo_extents, pinfo_y.halo_extents, 0));
  CHECK_CUDA_EXIT(cudaMemcpy(data.data(), output, data.size() * sizeof(*output), cudaMemcpyDeviceToHost));
  if (!compare_pencils(yref, data, pinfo_y)) {
    printf("FAILED cudecompTransposeXToY\n");
    exit(EXIT_FAILURE);
  }

  if (out_of_place) std::swap(input, output);

  CHECK_CUDA_EXIT(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                            pinfo_y.halo_extents, pinfo_z.halo_extents, 0));
  CHECK_CUDA_EXIT(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
  if (!compare_pencils(zref, data, pinfo_z)) {
    printf("FAILED cudecompTransposeYToZ\n");
    exit(EXIT_FAILURE);
  }

  if (out_of_place) std::swap(input, output);

  CHECK_CUDA_EXIT(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                            pinfo_z.halo_extents, pinfo_y.halo_extents, 0));
  CHECK_CUDA_EXIT(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
  if (!compare_pencils(yref, data, pinfo_y)) {
    printf("FAILED cudecompTransposeZToY\n");
    exit(EXIT_FAILURE);
  }

  if (out_of_place) std::swap(input, output);

  CHECK_CUDA_EXIT(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                            pinfo_y.halo_extents, pinfo_x.halo_extents, 0));
  CHECK_CUDA_EXIT(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
  if (!compare_pencils(xref, data, pinfo_x)) {
    printf("FAILED cudecompTransposeYToX\n");
    exit(EXIT_FAILURE);
  }

  CHECK_CUDA_EXIT(cudaFree(data_d));
  if (data_2_d) CHECK_CUDA_EXIT(cudaFree(data_2_d));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d));
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));

  CHECK_MPI_EXIT(MPI_Finalize());
}
