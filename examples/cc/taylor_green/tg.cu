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
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <getopt.h>

#include <mpi.h>

#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <cufftXt.h>

#include <cub/cub.cuh>

#include <cudecomp.h>

#define PI (std::atan(1.0) * 4.0)

// Error check macros
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

using real_t = double;
using complex_t = cuda::std::complex<real_t>;

static cufftType get_cufft_type_r2c(double) { return CUFFT_D2Z; }
static cufftType get_cufft_type_r2c(float) { return CUFFT_R2C; }
static cufftType get_cufft_type_c2r(double) { return CUFFT_Z2D; }
static cufftType get_cufft_type_c2r(float) { return CUFFT_C2R; }
static cufftType get_cufft_type_c2c(double) { return CUFFT_Z2Z; }
static cufftType get_cufft_type_c2c(float) { return CUFFT_C2C; }

static cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
static cudecompDataType_t get_cudecomp_datatype(double) { return CUDECOMP_DOUBLE; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }

// CUDA kernels
template <typename T, typename TS>
__global__ static void scale(T* U0, T* U1, T* U2, TS scale_factor, cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  U0[i] *= scale_factor;
  U1[i] *= scale_factor;
  U2[i] *= scale_factor;
}

__host__ __device__ static void get_gx(const cudecompPencilInfo_t& info, int64_t i, int64_t gx[3]) {
  int64_t lx[3];
  // Compute pencil local coordinate
  lx[0] = i % info.shape[0];
  lx[1] = i / info.shape[0] % info.shape[1];
  lx[2] = i / (info.shape[0] * info.shape[1]);
  gx[info.order[0]] = lx[0] + info.lo[0];
  gx[info.order[1]] = lx[1] + info.lo[1];
  gx[info.order[2]] = lx[2] + info.lo[2];
}

__global__ static void initialize_U(real_t* U_r0, real_t* U_r1, real_t* U_r2, real_t dx, real_t dy, real_t dz,
                                    cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  int64_t gx[3];
  get_gx(info, i, gx);

  real_t x = gx[0] * dx;
  real_t y = gx[1] * dy;
  real_t z = gx[2] * dz;

  // Taylor Green initial condition
  real_t u = std::sin(x) * std::cos(y) * std::cos(z);
  real_t v = -std::cos(x) * std::sin(y) * std::cos(z);
  real_t w = 0;

  U_r0[i] = u;
  U_r1[i] = v;
  U_r2[i] = w;
}

__global__ static void curl(const complex_t* Uh_c0, const complex_t* Uh_c1, const complex_t* Uh_c2, complex_t* dU_c0,
                            complex_t* dU_c1, complex_t* dU_c2, real_t* K0, real_t* K1, real_t* K2,
                            cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  int64_t gx[3];
  get_gx(info, i, gx);
  real_t kx = K0[gx[0]];
  real_t ky = K1[gx[1]];
  real_t kz = K2[gx[2]];
  complex_t Uhu = Uh_c0[i];
  complex_t Uhv = Uh_c1[i];
  complex_t Uhw = Uh_c2[i];
  dU_c0[i] = complex_t(0, 1) * (ky * Uhw - kz * Uhv);
  dU_c1[i] = complex_t(0, 1) * (kz * Uhu - kx * Uhw);
  dU_c2[i] = complex_t(0, 1) * (kx * Uhv - ky * Uhu);
}

__global__ static void cross(const real_t* U_r0, const real_t* U_r1, const real_t* U_r2, real_t* dU_r0, real_t* dU_r1,
                             real_t* dU_r2, int64_t N, cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;
  real_t Uu = U_r0[i] / (N * N * N); // Need to scale cuFFT inverse results
  real_t Uv = U_r1[i] / (N * N * N);
  real_t Uw = U_r2[i] / (N * N * N);
  real_t dUu = dU_r0[i] / (N * N * N);
  real_t dUv = dU_r1[i] / (N * N * N);
  real_t dUw = dU_r2[i] / (N * N * N);
  dU_r0[i] = (Uv * dUw - Uw * dUv);
  dU_r1[i] = (Uw * dUu - Uu * dUw);
  dU_r2[i] = (Uu * dUv - Uv * dUu);
}

__global__ static void compute_dU(const complex_t* Uh_c0, const complex_t* Uh_c1, const complex_t* Uh_c2,
                                  complex_t* dU_c0, complex_t* dU_c1, complex_t* dU_c2, real_t* K0, real_t* K1,
                                  real_t* K2, real_t kmax, int64_t N, real_t nu, cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  int64_t gx[3];
  get_gx(info, i, gx);
  real_t kx = K0[gx[0]];
  real_t ky = K1[gx[1]];
  real_t kz = K2[gx[2]];

  real_t k2 = kx * kx + ky * ky + kz * kz;

  // Dealiasing flag
  bool alias = (std::abs(kx) >= kmax || std::abs(ky) >= kmax || std::abs(kz) >= kmax);

  complex_t dUu = dU_c0[i];
  complex_t dUv = dU_c1[i];
  complex_t dUw = dU_c2[i];
  complex_t Ph = (kx * dUu + ky * dUv + kz * dUw) / (k2 != 0 ? k2 : 1);
  if (alias) {
    dU_c0[i] = -nu * k2 * Uh_c0[i];
    dU_c1[i] = -nu * k2 * Uh_c1[i];
    dU_c2[i] = -nu * k2 * Uh_c2[i];
  } else {
    dU_c0[i] = dU_c0[i] - Ph * kx - nu * k2 * Uh_c0[i];
    dU_c1[i] = dU_c1[i] - Ph * ky - nu * k2 * Uh_c1[i];
    dU_c2[i] = dU_c2[i] - Ph * kz - nu * k2 * Uh_c2[i];
  }
}

__global__ static void update_Uh(const complex_t* dU_c0, const complex_t* dU_c1, const complex_t* dU_c2,
                                 complex_t* Uh0_c0, complex_t* Uh0_c1, complex_t* Uh0_c2, complex_t* Uh_c0,
                                 complex_t* Uh_c1, complex_t* Uh_c2, real_t dt, cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;
  Uh_c0[i] = Uh0_c0[i] + dt * dU_c0[i];
  Uh_c1[i] = Uh0_c1[i] + dt * dU_c1[i];
  Uh_c2[i] = Uh0_c2[i] + dt * dU_c2[i];
}

__global__ static void sumsq(int N, const real_t* U_r0, const real_t* U_r1, const real_t* U_r2, real_t* sumsq,
                             cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  int64_t gx[3];
  get_gx(info, i, gx);
  if (gx[0] >= N || gx[1] >= N || gx[2] >= N) {
    // Set padded element entries to zero
    sumsq[i] = 0;
    return;
  }

  real_t u = U_r0[i] / (N * N * N); // Scaling cuFFT result
  real_t v = U_r1[i] / (N * N * N);
  real_t w = U_r2[i] / (N * N * N);

  sumsq[i] = (u * u + v * v + w * w);
}

class TGSolver {
public:
  // Timestepping scheme
  enum TimeScheme { RK1, RK4 };

  TGSolver(int N, real_t nu, real_t dt, TimeScheme tscheme = RK1) : N(N), nu(nu), dt(dt), tscheme(tscheme){};
  void finalize() {
    // Free memory
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaFree(U[i]));
      CHECK_CUDA_EXIT(cudaFree(dU[i]));
      free(U_cpu[i]);
    }
    CHECK_CUDA_EXIT(cudaFree(cub_sum));
    CHECK_CUDA_EXIT(cudaFree(cub_work));

    // Cleanup cudecomp resources
    CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work));
    CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_r));
    CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_c));
    CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
  }

  void initialize(MPI_Comm mpi_comm_in = MPI_COMM_WORLD) {

    // Get MPI values and set device
    mpi_comm = mpi_comm_in;
    CHECK_MPI_EXIT(MPI_Comm_rank(mpi_comm, &rank));
    CHECK_MPI_EXIT(MPI_Comm_size(mpi_comm, &nranks));

    CHECK_MPI_EXIT(MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &mpi_local_comm));
    CHECK_MPI_EXIT(MPI_Comm_rank(mpi_local_comm, &local_rank));
    CHECK_CUDA_EXIT(cudaSetDevice(local_rank));

    if (rank == 0) printf("running on %d x %d x %d spatial grid...\n", N, N, N);

    // Initialize cuDecomp
    cudecompInit(&handle, mpi_comm);

    // Setup grid descriptors, autotuning of process grid and backend selection
    cudecompGridDescConfig_t config;
    cudecompGridDescConfigSetDefaults(&config);
    config.pdims[0] = 0;
    config.pdims[1] = 0;
    config.transpose_axis_contiguous[0] = true;
    config.transpose_axis_contiguous[1] = true;
    config.transpose_axis_contiguous[2] = true;

    cudecompGridDescAutotuneOptions_t options;
    cudecompGridDescAutotuneOptionsSetDefaults(&options);
    options.dtype = get_cudecomp_datatype(complex_t(0));
    options.autotune_transpose_backend = true;

    std::array<int, 3> gdim_c{N / 2 + 1, N, N};
    config.gdims[0] = gdim_c[0];
    config.gdims[1] = gdim_c[1];
    config.gdims[2] = gdim_c[2];
    cudecompGridDescCreate(handle, &grid_desc_c, &config, &options);

    std::array<int, 3> gdim_r{(N / 2 + 1) * 2, N, N}; // with padding for in-place operation
    config.gdims[0] = gdim_r[0];
    config.gdims[1] = gdim_r[1];
    config.gdims[2] = gdim_r[2];
    cudecompGridDescCreate(handle, &grid_desc_r, &config, nullptr);

    // Get x-pencil information (real)
    cudecompGetPencilInfo(handle, grid_desc_r, &pinfo_x_r, 0, nullptr);

    // Get x-pencil information (complex)
    cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_x_c, 0, nullptr);

    // Get y-pencil information (complex)
    cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_y_c, 1, nullptr);

    // Get z-pencil information (complex)
    cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_z_c, 2, nullptr);

    // Get workspace size (only complex workspace required)
    int64_t num_elements_work_c;
    cudecompGetTransposeWorkspaceSize(handle, grid_desc_c, &num_elements_work_c);

    // Setup cuFFT
    // x-axis real-to-complex
    CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_r2c_x));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_r2c_x, 0));
    size_t work_sz_r2c_x;
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_r2c_x, N, get_cufft_type_r2c(real_t(0)),
                                     pinfo_x_r.shape[1] * pinfo_x_r.shape[2], &work_sz_r2c_x));

    // x-axis complex-to-real
    CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2r_x));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2r_x, 0));
    size_t work_sz_c2r_x;
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2r_x, N, get_cufft_type_c2r(real_t(0)),
                                     pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2r_x));

    // y-axis complex-to-complex
    CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_y));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_y, 0));
    size_t work_sz_c2c_y;
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_y, N, get_cufft_type_c2c(real_t(0)),
                                     pinfo_y_c.shape[1] * pinfo_y_c.shape[2], &work_sz_c2c_y));

    // z-axis complex-to-complex
    CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_z));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_z, 0));
    size_t work_sz_c2c_z;
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_z, N, get_cufft_type_c2c(real_t(0)),
                                     pinfo_z_c.shape[1] * pinfo_z_c.shape[2], &work_sz_c2c_z));

    // Allocate data arrays
    int64_t data_sz =
        std::max(std::max(std::max(2 * pinfo_x_c.size, 2 * pinfo_y_c.size), 2 * pinfo_z_c.size), pinfo_x_r.size) *
        sizeof(real_t);
    int64_t work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
    int64_t work_sz_cufft = std::max(std::max(work_sz_r2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z)), work_sz_c2r_x);
    int64_t work_sz = std::max(work_sz_decomp, work_sz_cufft);

    // Workspace array
    CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc_c, &work, work_sz));
    work_r = static_cast<real_t*>(work);
    work_c = static_cast<complex_t*>(work);

    // Assign cuFFT work area
    CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_r2c_x, work));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_y, work));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_z, work));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2r_x, work));

    // Data arrays
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaMalloc(&U[i], data_sz));
      CHECK_CUDA_EXIT(cudaMalloc(&dU[i], data_sz));
      U_cpu[i] = malloc(data_sz);
      U_r[i] = static_cast<real_t*>(U[i]);
      U_c[i] = static_cast<complex_t*>(U[i]);
      dU_r[i] = static_cast<real_t*>(dU[i]);
      dU_c[i] = static_cast<complex_t*>(dU[i]);
      U_cpu_r[i] = static_cast<real_t*>(U_cpu[i]);
      U_cpu_c[i] = static_cast<complex_t*>(U_cpu[i]);
    }

    // Set up CUB arrays
    CHECK_CUDA_EXIT(cudaMallocManaged(&cub_sum, sizeof(real_t)));
    CHECK_CUDA_EXIT(cub::DeviceReduce::Sum(cub_work, cub_work_sz, U_r[0], cub_sum, pinfo_x_r.size));
    CHECK_CUDA_EXIT(cudaMalloc(&cub_work, cub_work_sz));

    // Set timestepping variables
    switch (tscheme) {
    case RK1:
      rk_a.resize(1);
      rk_b.resize(1);
      rk_n = 1;
      break;
    case RK4:
      rk_n = 4;
      rk_a = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};
      rk_b = {0.5, 0.5, 1.};
      break;
    }

    Uh.resize(rk_b.size());
    Uh_r.resize(rk_b.size());
    Uh_c.resize(rk_b.size());
    for (int n = 0; n < rk_b.size(); ++n) {
      for (int i = 0; i < 3; ++i) {
        CHECK_CUDA_EXIT(cudaMalloc(&Uh[n][i], data_sz));
        Uh_r[n][i] = static_cast<real_t*>(Uh[n][i]);
        Uh_c[n][i] = static_cast<complex_t*>(Uh[n][i]);
      }
    }

    // Set up wavenumbers
    CHECK_CUDA_EXIT(cudaMallocManaged(&K[0], (N / 2 + 1) * sizeof(real_t)));
    CHECK_CUDA_EXIT(cudaMallocManaged(&K[1], N * sizeof(real_t)));
    CHECK_CUDA_EXIT(cudaMallocManaged(&K[2], N * sizeof(real_t)));
    for (int i = 0; i < N; ++i) {
      K[1][i] = (i < N / 2) ? i : i - N;
      K[2][i] = K[1][i];
    }
    for (int i = 0; i < N / 2 + 1; ++i) { K[0][i] = i; }
    K[0][N / 2] *= -1;

    // Initialize U (physical space)
    real_t dx = 2 * PI / N;
    real_t dy = 2 * PI / N;
    real_t dz = 2 * PI / N;
    initialize_U<<<(pinfo_x_r.size + 256 - 1) / 256, 256>>>(U_r[0], U_r[1], U_r[2], dx, dy, dz, pinfo_x_r);
    CHECK_CUDA_LAUNCH_EXIT();

    // Compute initial Uh (transformed U)
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaMemcpy(Uh_r[0][i], U_r[i], pinfo_x_r.size * sizeof(*U_r[i]), cudaMemcpyDeviceToDevice));
    }
    forward(Uh_r[0], Uh_c[0]);

    CHECK_CUDA_EXIT(cudaDeviceSynchronize());

    // Scale up initial U by N^3 as solver expects this
    scale<<<(pinfo_x_r.size + 256 - 1) / 256, 256>>>(U_r[0], U_r[1], U_r[2], N * N * N, pinfo_x_r);
    CHECK_CUDA_LAUNCH_EXIT();
  }

  void step() {
    switch (tscheme) {
    case RK1: update_rk1(); break;
    case RK4: update_rk4(); break;
    default: std::cerr << "Unknown TimeScheme provided." << std::endl;
    }

    flowtime += dt;
  }

  void print_stats() {
    // Compute enstrophy
    // Recompute curl and transform to physical space (z-pencil -> x-pencil).
    curl<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(Uh_c[0][0], Uh_c[0][1], Uh_c[0][2], dU_c[0], dU_c[1], dU_c[2], K[0],
                                                    K[1], K[2], pinfo_z_c);
    CHECK_CUDA_LAUNCH_EXIT();

    backward(dU_c, dU_r);

    sumsq<<<(pinfo_x_r.size + 256 - 1) / 256, 256>>>(N, dU_r[0], dU_r[1], dU_r[2], dU_r[0], pinfo_x_r);
    CHECK_CUDA_LAUNCH_EXIT();

    CHECK_CUDA_EXIT(cub::DeviceReduce::Sum(cub_work, cub_work_sz, dU_r[0], cub_sum, pinfo_x_r.size));
    CHECK_CUDA_EXIT(cudaDeviceSynchronize());

    double enst = 0.5 * (*cub_sum) / (N * N * N);
    if (nranks > 1) {
      CHECK_MPI_EXIT(MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &enst, &enst, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm));
    }

    // Compute kinetic energy
    sumsq<<<(pinfo_x_r.size + 256 - 1) / 256, 256>>>(N, U_r[0], U_r[1], U_r[2], dU_r[0], pinfo_x_r);
    CHECK_CUDA_LAUNCH_EXIT();

    CHECK_CUDA_EXIT(cub::DeviceReduce::Sum(cub_work, cub_work_sz, dU_r[0], cub_sum, pinfo_x_r.size));
    CHECK_CUDA_EXIT(cudaDeviceSynchronize());

    double ke = 0.5 * (*cub_sum) / (N * N * N);
    if (nranks > 1) {
      CHECK_MPI_EXIT(MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm));
    }

    // Print statistics
    if (rank == 0) {
      std::cout << std::fixed;
      std::cout << "flow time: " << std::setprecision(5) << flowtime << " ";
      std::cout << " ke: " << std::setprecision(14) << ke << " ";
      std::cout << " enstrophy: " << std::setprecision(14) << enst << std::endl;
    }
  }

  // Simple solution write to CSV (readable by ParaView)
  void write_solution(std::string prefix) {
    // Copy solution to host
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaMemcpy(U_cpu_r[i], U_r[i], pinfo_x_r.size * sizeof(*U_r[i]), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA_EXIT(cudaDeviceSynchronize());

    std::string fname;
    fname = prefix + "_" + std::to_string(rank) + ".csv";
    std::ofstream g;
    g.open(fname);
    g << "x, y, z, u, v, w" << std::endl;
    for (int64_t i = 0; i < pinfo_x_r.size; ++i) {
      int64_t gx[3];
      get_gx(pinfo_x_r, i, gx);
      real_t x = gx[0];
      real_t y = gx[1];
      real_t z = gx[2];
      if (gx[0] >= N || gx[1] >= N || gx[2] >= N) continue; // Skip any padded elements
      g << std::setprecision(12) << x << ", ";
      g << std::setprecision(12) << y << ", ";
      g << std::setprecision(12) << z << ", ";
      g << std::setprecision(12) << U_cpu_r[0][i] / (N * N * N) << ", "; // Scale cuFFT result
      g << std::setprecision(12) << U_cpu_r[1][i] / (N * N * N) << ", ";
      g << std::setprecision(12) << U_cpu_r[2][i] / (N * N * N) << std::endl;
    }
    g.close();
  }

private:
  void forward(std::array<real_t*, 3>& U_r, std::array<complex_t*, 3>& U_c) {
    for (int i = 0; i < 3; ++i) {
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_r2c_x, U_r[i], U_c[i], CUFFT_FORWARD));
      cudecompTransposeXToY(handle, grid_desc_c, U_c[i], U_c[i], work_c, get_cudecomp_datatype(complex_t(0)), nullptr,
                            nullptr, 0);
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, U_c[i], U_c[i], CUFFT_FORWARD));
      cudecompTransposeYToZ(handle, grid_desc_c, U_c[i], U_c[i], work_c, get_cudecomp_datatype(complex_t(0)), nullptr,
                            nullptr, 0);
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, U_c[i], U_c[i], CUFFT_FORWARD));
    }
  }

  void backward(std::array<complex_t*, 3>& U_c, std::array<real_t*, 3>& U_r) {
    for (int i = 0; i < 3; ++i) {
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, U_c[i], U_c[i], CUFFT_INVERSE));
      cudecompTransposeZToY(handle, grid_desc_c, U_c[i], U_c[i], work_c, get_cudecomp_datatype(complex_t(0)), nullptr,
                            nullptr, 0);
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, U_c[i], U_c[i], CUFFT_INVERSE));
      cudecompTransposeYToX(handle, grid_desc_c, U_c[i], U_c[i], work_c, get_cudecomp_datatype(complex_t(0)), nullptr,
                            nullptr, 0);
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2r_x, U_c[i], U_r[i], CUFFT_INVERSE));
    }
  }

  void compute_rhs(std::array<complex_t*, 3>& Uh_c) {

    // Compute curl and transform to physical space (z-pencil -> x-pencil)
    curl<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(Uh_c[0], Uh_c[1], Uh_c[2], dU_c[0], dU_c[1], dU_c[2], K[0], K[1],
                                                    K[2], pinfo_z_c);
    CHECK_CUDA_LAUNCH_EXIT();

    backward(dU_c, dU_r);

    // Compute cross and transform to frequency space (x-pencil -> z-pencil)
    cross<<<(pinfo_x_r.size + 256 - 1) / 256, 256>>>(U_r[0], U_r[1], U_r[2], dU_r[0], dU_r[1], dU_r[2], N, pinfo_x_r);
    CHECK_CUDA_LAUNCH_EXIT();

    forward(dU_r, dU_c);

    // Compute dU in frequency space (z-pencil)
    real_t kmax = 2.0 / 3.0 * (N / 2 + 1); // aliasing limit
    compute_dU<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(Uh_c[0], Uh_c[1], Uh_c[2], dU_c[0], dU_c[1], dU_c[2], K[0],
                                                          K[1], K[2], kmax, N, nu, pinfo_z_c);
    CHECK_CUDA_LAUNCH_EXIT();
  }

  // Forward Euler (z-pencil)
  void update_rk1() {
    compute_rhs(Uh_c[0]);
    update_Uh<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(dU_c[0], dU_c[1], dU_c[2], Uh_c[0][0], Uh_c[0][1], Uh_c[0][2],
                                                         Uh_c[0][0], Uh_c[0][1], Uh_c[0][2], dt, pinfo_z_c);
    CHECK_CUDA_LAUNCH_EXIT();

    // Get physical U (z-pencil -> x-pencil)
    // Copy Uh to U (cuFFT C2R clobbers input, need to make copy)
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaMemcpy(U_c[i], Uh_c[0][i], pinfo_z_c.size * sizeof(*U_c[i]), cudaMemcpyDeviceToDevice));
    }
    backward(U_c, U_r);
  }

  // RK4 (z-pencil)
  void update_rk4() {
    // Copy initial Uh
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(
          cudaMemcpy(Uh_c[1][i], Uh_c[0][i], pinfo_z_c.size * sizeof(*Uh_c[0][i]), cudaMemcpyDeviceToDevice));
    }

    // Run RK stages
    for (int i = 0; i < rk_n; ++i) {
      compute_rhs(Uh_c[1]);
      if (i < 3) {
        update_Uh<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(dU_c[0], dU_c[1], dU_c[2], Uh_c[0][0], Uh_c[0][1],
                                                             Uh_c[0][2], Uh_c[1][0], Uh_c[1][1], Uh_c[1][2],
                                                             rk_b[i] * dt, pinfo_z_c);
        CHECK_CUDA_LAUNCH_EXIT();

        // Get physical U (z-pencil -> x-pencil)
        // Copy Uh to U (cuFFT C2R clobbers input, need to make copy)
        for (int i = 0; i < 3; ++i) {
          CHECK_CUDA_EXIT(cudaMemcpy(U_c[i], Uh_c[1][i], pinfo_z_c.size * sizeof(*U_c[i]), cudaMemcpyDeviceToDevice));
        }
        backward(U_c, U_r);
      }

      if (i == 0) {
        // First stage: assign to Uh_c[2]
        update_Uh<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(dU_c[0], dU_c[1], dU_c[2], Uh_c[0][0], Uh_c[0][1],
                                                             Uh_c[0][2], Uh_c[2][0], Uh_c[2][1], Uh_c[2][2],
                                                             rk_a[i] * dt, pinfo_z_c);
        CHECK_CUDA_LAUNCH_EXIT();
      } else if (i == 3) {
        // Last stage: assign to Uh_c[0]
        update_Uh<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(dU_c[0], dU_c[1], dU_c[2], Uh_c[2][0], Uh_c[2][1],
                                                             Uh_c[2][2], Uh_c[0][0], Uh_c[0][1], Uh_c[0][2],
                                                             rk_a[i] * dt, pinfo_z_c);
        CHECK_CUDA_LAUNCH_EXIT();
      } else {
        // Middle stages: accumulate in Uh_c[2]
        update_Uh<<<(pinfo_z_c.size + 256 - 1) / 256, 256>>>(dU_c[0], dU_c[1], dU_c[2], Uh_c[2][0], Uh_c[2][1],
                                                             Uh_c[2][2], Uh_c[2][0], Uh_c[2][1], Uh_c[2][2],
                                                             rk_a[i] * dt, pinfo_z_c);
        CHECK_CUDA_LAUNCH_EXIT();
      }
    }

    // Get physical U (z-pencil -> x-pencil)
    // Copy Uh to U (cuFFT C2R clobbers input, need to make copy)
    for (int i = 0; i < 3; ++i) {
      CHECK_CUDA_EXIT(cudaMemcpy(U_c[i], Uh_c[0][i], pinfo_z_c.size * sizeof(*U_c[i]), cudaMemcpyDeviceToDevice));
    }
    backward(U_c, U_r);
  }

  // Solver settings
  int N;
  real_t nu;
  real_t dt;
  TimeScheme tscheme;
  real_t flowtime = 0;

  // MPI variables
  int rank, local_rank, nranks;
  MPI_Comm mpi_comm, mpi_local_comm;
  bool should_finalize_mpi = false;

  // cuDecomp
  cudecompHandle_t handle;
  cudecompGridDesc_t grid_desc_r; // real grid
  cudecompGridDesc_t grid_desc_c; // complex grid
  cudecompPencilInfo_t pinfo_x_r;
  cudecompPencilInfo_t pinfo_x_c;
  cudecompPencilInfo_t pinfo_y_c;
  cudecompPencilInfo_t pinfo_z_c;

  // cuFFT
  cufftHandle cufft_plan_r2c_x;
  cufftHandle cufft_plan_c2r_x;
  cufftHandle cufft_plan_c2c_y;
  cufftHandle cufft_plan_c2c_z;

  // CUB
  void* cub_work = nullptr;
  size_t cub_work_sz;
  real_t* cub_sum;

  // Workspace
  void* work;
  real_t* work_r;
  complex_t* work_c;

  // Data arrays
  std::array<void*, 3> U, dU;          // Raw buffers
  std::array<real_t*, 3> U_r, dU_r;    // real pointers (aliased);
  std::array<complex_t*, 3> U_c, dU_c; // complex pointers (aliased)
  std::array<real_t*, 3> K;            // wavenumbers

  std::vector<std::array<void*, 3>> Uh;
  std::vector<std::array<real_t*, 3>> Uh_r;
  std::vector<std::array<complex_t*, 3>> Uh_c;

  std::array<void*, 3> U_cpu;
  std::array<real_t*, 3> U_cpu_r;
  std::array<complex_t*, 3> U_cpu_c;

  // Timestepping variables
  int rk_n;
  std::vector<real_t> rk_a, rk_b;
};

static void usage(const char* pname) {

  const char* bname = rindex(pname, '/');
  if (!bname) {
    bname = pname;
  } else {
    bname++;
  }

  fprintf(
      stdout,
      "Usage: %s [options]\n"
      "options:\n"
      "\t-n|--N\n"
      "\t\tDimension of grid. (default: 256) \n"
      "\t-i|--niter (-i)\n"
      "\t\tNumber of iterations to run. (default: 1000) \n"
      "\t-p|--printfreq\n"
      "\t\tFrequency of printing stats. (default: 100) \n"
      "\t-o|--csv_prefix\n"
      "\t\tFile prefix to write final solution to, in CSV format as <csv_prefix>_<rank>.csv. (default: no write) \n",
      bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  int rank;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // Parse command-line arguments
  int N = 256;
  int niter = 1000;
  int printfreq = 100;
  std::string csvfile;

  while (1) {
    static struct option long_options[] = {{"N", required_argument, 0, 'n'},
                                           {"niter", required_argument, 0, 'i'},
                                           {"printfreq", required_argument, 0, 'p'},
                                           {"csvfile", required_argument, 0, 'o'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int ch = getopt_long(argc, argv, "n:i:p:o:h", long_options, &option_index);
    if (ch == -1) break;

    switch (ch) {
    case 0: break;
    case 'n': N = atoi(optarg); break;
    case 'i': niter = atoi(optarg); break;
    case 'p': printfreq = atoi(optarg); break;
    case 'o': csvfile = std::string(optarg); break;
    case 'h': usage(argv[0]); break;
    case '?': exit(EXIT_FAILURE);
    default: fprintf(stderr, "unknown option: %c\n", ch); exit(EXIT_FAILURE);
    }
  }

  // Physical parameters
  real_t nu = 0.000625;
  real_t dt = 0.001;

  // Construct and initialize solver
  TGSolver solver(N, nu, dt, TGSolver::TimeScheme::RK4);
  solver.initialize(MPI_COMM_WORLD);
  solver.print_stats();

  // Run simulation
  CHECK_CUDA_EXIT(cudaDeviceSynchronize());
  CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
  double ts = MPI_Wtime();
  double ts_step = MPI_Wtime();
  int count = 0;
  for (int i = 0; i < niter; ++i) {
    solver.step();
    count++;

    if (i > 0 && (i + 1) % printfreq == 0) {
      CHECK_CUDA_EXIT(cudaDeviceSynchronize());
      CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
      double te_step = MPI_Wtime();
      if (rank == 0) printf("Average iteration time: %f ms\n", (te_step - ts_step) * 1000 / count);
      count = 0;
      solver.print_stats();
      CHECK_CUDA_EXIT(cudaDeviceSynchronize());
      CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
      ts_step = MPI_Wtime();
    }
  }
  CHECK_CUDA_EXIT(cudaDeviceSynchronize());
  CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
  double te = MPI_Wtime();
  if (rank == 0) printf("Simulation time: %f s\n", te - ts);

  // Write solution file, if requested
  if (csvfile.size() != 0) { solver.write_solution(csvfile); }

  solver.finalize();

  CHECK_CUDA_EXIT(cudaDeviceSynchronize());
  CHECK_MPI_EXIT(MPI_Finalize());
}
