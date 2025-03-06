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
#include <cstring>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
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

// Helper function to convert string into equivalent argc/argv input
static void string_to_argcv(const std::string& input, int& argc, char**& argv) {
  std::istringstream stream(input);
  std::vector<std::string> tokens;
  std::string token;

  // Set argv[0] to dummy value
  tokens.push_back("");

  while (stream >> token) {
    tokens.push_back(token);
  }

  argc = tokens.size();
  argv = new char*[argc];

  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[tokens[i].size() + 1];
    std::strcpy(argv[i], tokens[i].c_str());
  }
}

static void free_argcv(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
}

static std::vector<std::string> read_testfile(const std::string& filename) {
  std::vector<std::string> cases;
  std::ifstream file(filename);
  std::string line;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      cases.push_back(line);
    }
    file.close();
  } else {
    fprintf(stderr, "unable to open test file: %s\n", filename.c_str());
    exit(EXIT_FAILURE);
  }

  return cases;
}

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
          "\t\tX-pencil halo_extents setting. (default: 0 0 0) \n"
          "\t--hey\n"
          "\t\tY-pencil halo_extents setting. (default: 0 0 0) \n"
          "\t--hez\n"
          "\t\tZ-pencil halo_extents setting. (default: 0 0 0) \n"
          "\t--mem_order\n"
          "\t\ttranspose_mem_order setting. (default: unset) \n"
          "\t-m|--use-managed-memory\n"
          "\t\tFlag to test operation with managed memory.\n"
          "\t-o|--out-of-place\n"
          "\t\tFlag to test out of place operation.\n",
          bname);
  exit(EXIT_SUCCESS);
}

struct transposeTestArgs {
  int gx = 256;
  int gy = 256;
  int gz = 256;
  int pr = 0;
  int pc = 0;
  cudecompTransposeCommBackend_t comm_backend = static_cast<cudecompTransposeCommBackend_t>(0);
  std::array<bool, 3> axis_contiguous{};
  std::array<int, 3> gdims_dist{};
  std::array<int, 3> halo_extents_x{};
  std::array<int, 3> halo_extents_y{};
  std::array<int, 3> halo_extents_z{};
  bool out_of_place = false;
  bool use_managed_memory = false;
  std::array<int, 9> mem_order{-1, -1, -1, -1, -1, -1, -1, -1, -1};
};

static transposeTestArgs parse_arguments(const std::string& arguments) {
  transposeTestArgs args;

  int argc;
  char** argv;
  string_to_argcv(arguments, argc, argv);

  // Parse command-line arguments
  optind = 0;
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
                                           {"mem_order", required_argument, 0, 'q'},
                                           {"out-of-place", no_argument, 0, 'o'},
                                           {"use-managed-memory", no_argument, 0, 'm'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:z:b:r:c:1:2:3:4:5:6:7:8:9:q:omh", long_options, &option_index);
    if (ch == -1) break;

    switch (ch) {
    case 0: break;
    case 'x': args.gx = atoi(optarg); break;
    case 'y': args.gy = atoi(optarg); break;
    case 'z': args.gz = atoi(optarg); break;
    case 'b': args.comm_backend = static_cast<cudecompTransposeCommBackend_t>(atoi(optarg)); break;
    case 'r': args.pr = atoi(optarg); break;
    case 'c': args.pc = atoi(optarg); break;
    case '1': args.axis_contiguous[0] = atoi(optarg); break;
    case '2': args.axis_contiguous[1] = atoi(optarg); break;
    case '3': args.axis_contiguous[2] = atoi(optarg); break;
    case '4': args.gdims_dist[0] = atoi(optarg); break;
    case '5': args.gdims_dist[1] = atoi(optarg); break;
    case '6': args.gdims_dist[2] = atoi(optarg); break;
    case '7':
      optind--;
      for (int i = 0; i < 3; ++i) {
        args.halo_extents_x[i] = atoi(argv[optind]);
        optind++;
      }
      break;
    case '8':
      optind--;
      for (int i = 0; i < 3; ++i) {
        args.halo_extents_y[i] = atoi(argv[optind]);
        optind++;
      }
      break;
    case '9':
      optind--;
      for (int i = 0; i < 3; ++i) {
        args.halo_extents_z[i] = atoi(argv[optind]);
        optind++;
      }
      break;
    case 'o': args.out_of_place = true; break;
    case 'm': args.use_managed_memory = true; break;
    case 'q':
      optind--;
      for (int i = 0; i < 9; ++i) {
        args.mem_order[i] = atoi(argv[optind]);
        optind++;
      }
      break;
    case 'h': usage(argv[0]); break;
    case '?': exit(EXIT_FAILURE);
    default: fprintf(stderr, "unknown option: %c\n", ch); exit(EXIT_FAILURE);
    }
  }

  // Finish setting up gdim_dist
  args.gdims_dist[0] = args.gx - args.gdims_dist[0];
  args.gdims_dist[1] = args.gy - args.gdims_dist[1];
  args.gdims_dist[2] = args.gz - args.gdims_dist[2];

  free_argcv(argc, argv);

  return args;
}

int rank, nranks;
cudecompHandle_t handle;
std::unordered_map<cudecompTransposeCommBackend_t, cudecompGridDesc_t> grid_desc_cache;

// Cache a single grid descriptor per backend type. This keeps NCCL/NVSHMEM initialized between tests for
// better throughput.
static void cache_grid_desc(const cudecompGridDesc_t& grid_desc, cudecompTransposeCommBackend_t backend) {
  if (grid_desc_cache.count(backend) != 0) {
    CHECK_CUDECOMP(cudecompGridDescDestroy(handle, grid_desc_cache[backend]));
  }
  grid_desc_cache[backend] = grid_desc;
}

static int run_test(const std::string& arguments, bool silent) {

  try {
    transposeTestArgs args = parse_arguments(arguments);

    std::array<int, 2> pdims;
    pdims[0] = args.pr;
    pdims[1] = args.pc;

    if (!silent && rank == 0) printf("running on %d x %d x %d spatial grid...\n", args.gx, args.gy, args.gz);

    // Setup grid descriptor
    std::array<int32_t, 3> gdims{args.gx, args.gy, args.gz};
    cudecompGridDesc_t grid_desc;
    cudecompGridDescConfig_t config;
    CHECK_CUDECOMP(cudecompGridDescConfigSetDefaults(&config));
    config.pdims[0] = pdims[0];
    config.pdims[1] = pdims[1];
    config.gdims[0] = gdims[0];
    config.gdims[1] = gdims[1];
    config.gdims[2] = gdims[2];
    config.gdims_dist[0] = args.gdims_dist[0];
    config.gdims_dist[1] = args.gdims_dist[1];
    config.gdims_dist[2] = args.gdims_dist[2];
    config.transpose_axis_contiguous[0] = args.axis_contiguous[0];
    config.transpose_axis_contiguous[1] = args.axis_contiguous[1];
    config.transpose_axis_contiguous[2] = args.axis_contiguous[2];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        config.transpose_mem_order[i][j] = args.mem_order[i * 3 + j];
      }
    }

    cudecompGridDescAutotuneOptions_t options;
    CHECK_CUDECOMP(cudecompGridDescAutotuneOptionsSetDefaults(&options));
    options.dtype = get_cudecomp_datatype(real_t(0));
    for (int i = 0; i < 4; ++i) { options.transpose_use_inplace_buffers[i] = !args.out_of_place; }

    if (args.comm_backend != 0) {
      config.transpose_comm_backend = args.comm_backend;
    } else {
      options.autotune_transpose_backend = true;
    }

    CHECK_CUDECOMP(cudecompGridDescCreate(handle, &grid_desc, &config, &options));
    cache_grid_desc(grid_desc, config.transpose_comm_backend);

    if (!silent && rank == 0) {
      printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
      printf("running using %s transpose backend...\n",
             cudecompTransposeCommBackendToString(config.transpose_comm_backend));
    }

    // Get x-pencil information
    cudecompPencilInfo_t pinfo_x;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, args.halo_extents_x.data()));

    // Get y-pencil information
    cudecompPencilInfo_t pinfo_y;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, args.halo_extents_y.data()));

    // Get z-pencil information
    cudecompPencilInfo_t pinfo_z;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, args.halo_extents_z.data()));

    // Get workspace size
    int64_t workspace_num_elements;
    CHECK_CUDECOMP(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &workspace_num_elements));

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
    if (args.use_managed_memory) {
      CHECK_CUDA(cudaMallocManaged(&data_d, data.size() * sizeof(*data_d)));
    } else {
      CHECK_CUDA(cudaMalloc(&data_d, data.size() * sizeof(*data_d)));
    }
    int64_t dtype_size;
    CHECK_CUDECOMP(cudecompGetDataTypeSize(get_cudecomp_datatype(real_t(0)), &dtype_size));
    CHECK_CUDECOMP(
        cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_d), workspace_num_elements * dtype_size));

    real_t* data_2_d = nullptr;
    if (args.out_of_place) {
      if (args.use_managed_memory) {
        CHECK_CUDA(cudaMallocManaged(&data_2_d, data.size() * sizeof(*data_2_d)));
      } else {
        CHECK_CUDA(cudaMalloc(&data_2_d, data.size() * sizeof(*data_2_d)));
      }
    }

    // Running correctness tests
    if (!silent && rank == 0) printf("running correctness tests...\n");

    // Initialize data to reference x-pencil data
    CHECK_CUDA(cudaMemcpy(data_d, xref.data(), xref.size() * sizeof(*data_d), cudaMemcpyHostToDevice));

    real_t* input = data_d;
    real_t* output = data_d;
    if (args.out_of_place) output = data_2_d;

    CHECK_CUDA(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
    CHECK_CUDECOMP(cudecompTransposeXToY(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                              pinfo_x.halo_extents, pinfo_y.halo_extents, 0));
    CHECK_CUDA(cudaMemcpy(data.data(), output, data.size() * sizeof(*output), cudaMemcpyDeviceToHost));
    if (!compare_pencils(yref, data, pinfo_y)) {
      fprintf(stderr, "FAILED cudecompTransposeXToY\n");
      return 1;
    }

    if (args.out_of_place) std::swap(input, output);

    CHECK_CUDA(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
    CHECK_CUDECOMP(cudecompTransposeYToZ(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                              pinfo_y.halo_extents, pinfo_z.halo_extents, 0));
    CHECK_CUDA(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
    if (!compare_pencils(zref, data, pinfo_z)) {
      fprintf(stderr, "FAILED cudecompTransposeYToZ\n");
      return 1;
    }

    if (args.out_of_place) std::swap(input, output);

    CHECK_CUDA(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
    CHECK_CUDECOMP(cudecompTransposeZToY(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                              pinfo_z.halo_extents, pinfo_y.halo_extents, 0));
    CHECK_CUDA(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
    if (!compare_pencils(yref, data, pinfo_y)) {
      fprintf(stderr, "FAILED cudecompTransposeZToY\n");
      return 1;
    }

    if (args.out_of_place) std::swap(input, output);

    CHECK_CUDA(cudaMemset(work_d, 0, workspace_num_elements * dtype_size));
    CHECK_CUDECOMP(cudecompTransposeYToX(handle, grid_desc, input, output, work_d, get_cudecomp_datatype(real_t(0)),
                                              pinfo_y.halo_extents, pinfo_x.halo_extents, 0));
    CHECK_CUDA(cudaMemcpy(data.data(), output, data.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
    if (!compare_pencils(xref, data, pinfo_x)) {
      fprintf(stderr, "FAILED cudecompTransposeYToX\n");
      return 1;
    }

    CHECK_CUDA(cudaFree(data_d));
    if (data_2_d) CHECK_CUDA(cudaFree(data_2_d));
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_d));
  } catch (const std::exception& e) {
    return 1;
  }

  return 0;
}

int main(int argc, char** argv) {
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank));

  // Check if test file was provided
  std::string testfile;
  bool using_testfile = false;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--testfile") == 0) {
      testfile = std::string(argv[i+1]);
      using_testfile = true;
      break;
    }
  }

  // Initialize cuDecomp
  CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));

  std::vector<std::string> testcases;
  if (!using_testfile) {
    std::string arguments;
    for (int i = 1;  i < argc; ++i) {
      arguments += argv[i];
      if (i < argc - 1) {
        arguments += " ";
      }
    }
    testcases.push_back(arguments);
  } else {
    testcases = read_testfile(testfile);
  }

  std::vector<std::string> failed_cases;
  double t0 = MPI_Wtime();
  if (using_testfile && rank == 0) printf("Running %d tests...\n", static_cast<int>(testcases.size()));
  for (int i = 0; i < testcases.size(); ++i) {
    if (using_testfile && rank == 0) printf("command: %s %s\n", argv[0], testcases[i].c_str());
    int res = run_test(testcases[i], using_testfile);
    CHECK_MPI_EXIT(MPI_Reduce((rank == 0) ? MPI_IN_PLACE: &res, &res, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));
    if (using_testfile && rank == 0) {
      if (res != 0) {
        printf(" FAILED\n");
        failed_cases.push_back(testcases[i]);
      } else {
        printf(" PASSED\n");
      }
    }
    CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
    if (using_testfile && (i + 1) % 10 == 0) {
      if (rank == 0) printf("Completed %d/%d tests, running time %f s\n", i + 1, static_cast<int>(testcases.size()), MPI_Wtime() - t0);
    }
  }

  int retcode = 0;
  if (using_testfile) {
    if (rank == 0) {
      printf("Completed all tests, running time %f s,\n", MPI_Wtime() - t0);
      if (failed_cases.size() == 0) {
        printf("Passed all tests.\n");
      } else {
        printf("Failed %d/%d tests. Failing cases:\n", static_cast<int>(failed_cases.size()), static_cast<int>(testcases.size()));
        for (int i = 0; i < failed_cases.size(); ++i) {
          printf(" %s %s\n", argv[0], failed_cases[i].c_str());
        }
      }
    }
    if (failed_cases.size() != 0) retcode = 1;
  }

  // Finalize
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
  CHECK_MPI_EXIT(MPI_Finalize());

  return retcode;

}

