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
  for (int64_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != real_t(-1)) {
      if (std::abs(ref[i] - res[i]) > 1e-6) return false;
    }
  }
  return true;
}
static void initialize_pencil(std::vector<real_t>& ref, const cudecompPencilInfo_t& pinfo,
                              const std::array<int32_t, 3>& gdims) {
  for (int64_t i = 0; i < ref.size(); ++i) {
    // Compute pencil local coordinate
    int64_t lx[3];
    lx[0] = i % pinfo.shape[0];
    lx[1] = i / pinfo.shape[0] % pinfo.shape[1];
    lx[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    // Compute global grid coordinate
    int64_t gx[3];
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

static void initialize_reference(std::vector<real_t>& ref, const cudecompPencilInfo_t& pinfo,
                                 const std::array<int32_t, 3>& gdims, const std::array<bool, 3>& halo_periods) {
  for (int64_t i = 0; i < ref.size(); ++i) {
    // Compute pencil local coordinate
    int64_t lx[3];
    lx[0] = i % pinfo.shape[0];
    lx[1] = i / pinfo.shape[0] % pinfo.shape[1];
    lx[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    // Compute global grid coordinate
    int64_t gx[3];
    gx[pinfo.order[0]] = lx[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]];
    gx[pinfo.order[1]] = lx[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]];
    gx[pinfo.order[2]] = lx[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]];

    // Handle halo entries on global boundary and periodicity
    bool unset = false;
    for (int j = 0; j < 3; ++j) {
      if (gx[pinfo.order[j]] < 0 || gx[pinfo.order[j]] >= gdims[pinfo.order[j]]) {
        if (halo_periods[pinfo.order[j]]) {
          // If halo entry is on boundary and periodic, wrap around index
          gx[pinfo.order[j]] = (gx[pinfo.order[j]] + gdims[pinfo.order[j]]) % gdims[pinfo.order[j]];
        } else {
          // If halo entry is on boundary but not periodic, mark entry for unset value (-1)
          unset = true;
        }
      }
    }

    int64_t gi = (unset) ? -1 : gx[0] + gdims[0] * (gx[1] + gx[2] * gdims[1]);
    ref[i] = gi;
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
          "\t\tHalo communication backend (default: 0, autotune) \n"
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
          "\t--hpx\n"
          "\t\tX-dimension halo_periods setting. (default: 0) \n"
          "\t--hpy\n"
          "\t\tY-dimension halo_periods setting. (default: 0) \n"
          "\t--hpz\n"
          "\t\tZ-dimension halo_periods setting. (default: 0) \n"
          "\t--ax\n"
          "\t\t Pencil configuration (by axis) to test. (default: 0) \n"
          "\t-m|--use-managed-memory\n"
          "\t\tFlag to test operation with managed memory. (default: 0) \n",
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
  cudecompHaloCommBackend_t comm_backend = static_cast<cudecompHaloCommBackend_t>(0);
  std::array<bool, 3> axis_contiguous{};
  std::array<int, 3> gdims_dist{};
  std::array<int, 3> halo_extents{1, 1, 1};
  std::array<bool, 3> halo_periods{true, true, true};
  int axis = 0;
  bool use_managed_memory = false;

  while (1) {
    static struct option long_options[] = {
        {"gx", required_argument, 0, 'x'},  {"gy", required_argument, 0, 'y'},
        {"gz", required_argument, 0, 'z'},  {"backend", required_argument, 0, 'b'},
        {"pr", required_argument, 0, 'r'},  {"pc", required_argument, 0, 'c'},
        {"acx", required_argument, 0, '1'}, {"acy", required_argument, 0, '2'},
        {"acz", required_argument, 0, '3'}, {"gdx", required_argument, 0, '4'},
        {"gdy", required_argument, 0, '5'}, {"gdz", required_argument, 0, '6'},
        {"hex", required_argument, 0, '7'}, {"hey", required_argument, 0, '8'},
        {"hez", required_argument, 0, '9'}, {"hpx", required_argument, 0, 'e'},
        {"hpy", required_argument, 0, 'f'}, {"hpz", required_argument, 0, 'g'},
        {"ax", required_argument, 0, 'a'},  {"use-managed-memory", no_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},      {0, 0, 0, 0}};

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:z:b:r:c:1:2:3:4:5:6:7:8:9:e:f:g:a:mh", long_options, &option_index);
    if (ch == -1) break;

    switch (ch) {
    case 0: break;
    case 'x': gx = atoi(optarg); break;
    case 'y': gy = atoi(optarg); break;
    case 'z': gz = atoi(optarg); break;
    case 'b': comm_backend = static_cast<cudecompHaloCommBackend_t>(atoi(optarg)); break;
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
    case 'e': halo_periods[0] = atoi(optarg); break;
    case 'f': halo_periods[1] = atoi(optarg); break;
    case 'g': halo_periods[2] = atoi(optarg); break;
    case 'a': axis = atoi(optarg); break;
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

  // Check for custom process grid
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
  options.halo_extents[0] = halo_extents[0];
  options.halo_extents[1] = halo_extents[1];
  options.halo_extents[2] = halo_extents[2];
  options.halo_periods[0] = halo_periods[0];
  options.halo_periods[1] = halo_periods[1];
  options.halo_periods[2] = halo_periods[2];
  options.halo_axis = axis;
  options.dtype = get_cudecomp_datatype(real_t(0));
  options.grid_mode = CUDECOMP_AUTOTUNE_GRID_HALO;

  if (comm_backend != 0) {
    config.halo_comm_backend = comm_backend;
  } else {
    options.autotune_halo_backend = true;
  }

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  if (rank == 0) {
    printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
    printf("running using %s halo backend...\n", cudecompHaloCommBackendToString(config.halo_comm_backend));
  }

  // Get pencil information
  cudecompPencilInfo_t pinfo;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, halo_extents.data()));

  // Get workspace size
  int64_t workspace_num_elements;
  CHECK_CUDECOMP_EXIT(
      cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents.data(), &workspace_num_elements));

  // Allocate data arrays
  int64_t data_num_elements = pinfo.size;

  std::vector<real_t> data(data_num_elements);

  // Create reference data
  std::vector<real_t> init(pinfo.size);
  std::vector<real_t> ref(pinfo.size);

  initialize_pencil(init, pinfo, gdims);
  initialize_reference(ref, pinfo, gdims, halo_periods);

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

  // Running correctness tests
  if (rank == 0) printf("running correctness tests...\n");
  // Initialize data to initial pencil data
  CHECK_CUDA_EXIT(cudaMemcpy(data_d, init.data(), init.size() * sizeof(*data_d), cudaMemcpyHostToDevice));

  real_t* input = data_d;
  for (int i = 0; i < 3; ++i) {
    switch (axis) {
    case 0:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, input, work_d, get_cudecomp_datatype(real_t(0)),
                                               pinfo.halo_extents, halo_periods.data(), i, 0));
      break;
    case 1:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosY(handle, grid_desc, input, work_d, get_cudecomp_datatype(real_t(0)),
                                               pinfo.halo_extents, halo_periods.data(), i, 0));
      break;
    case 2:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosZ(handle, grid_desc, input, work_d, get_cudecomp_datatype(real_t(0)),
                                               pinfo.halo_extents, halo_periods.data(), i, 0));
      break;
    }
  }

  CHECK_CUDA_EXIT(cudaMemcpy(data.data(), data_d, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
  if (!compare_pencils(ref, data, pinfo)) {
    printf("FAILED cudecompUpdateHalos\n");
    exit(EXIT_FAILURE);
  }

  CHECK_CUDA_EXIT(cudaFree(data_d));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d));
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));

  CHECK_MPI_EXIT(MPI_Finalize());
}
