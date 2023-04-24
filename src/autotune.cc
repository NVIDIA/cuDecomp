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

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/common.h"
#include "internal/halo.h"
#include "internal/transpose.h"

namespace cudecomp {
namespace {

static constexpr int NWARMUP = 3;
static constexpr int NTRIALS = 5;

static std::vector<int> getFactors(int N) {
  std::vector<int> factors;
  for (int i = 1; i <= std::sqrt(N); ++i) {
    if (N % i == 0) {
      factors.push_back(i);
      if (N / i != i) { factors.push_back(N / i); }
    }
  }
  std::sort(factors.begin(), factors.end());
  return factors;
}

template <typename T> static std::vector<T> processTimings(cudecompHandle_t handle, std::vector<T> times, T scale = 1) {
  std::sort(times.begin(), times.end());
  double t_min = times[0];
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_min, 1, MPI_DOUBLE, MPI_MIN, handle->mpi_comm));
  double t_max = times[times.size() - 1];
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_max, 1, MPI_DOUBLE, MPI_MAX, handle->mpi_comm));

  double t_avg = std::accumulate(times.begin(), times.end(), T(0)) / times.size();
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_avg, 1, MPI_DOUBLE, MPI_SUM, handle->mpi_comm));
  t_avg /= handle->nranks;

  for (auto& t : times) { t = (t - t_avg) * (t - t_avg); }
  double t_var = std::accumulate(times.begin(), times.end(), T(0)) / times.size();
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_var, 1, MPI_DOUBLE, MPI_SUM, handle->mpi_comm));
  t_var /= handle->nranks;
  double t_std = std::sqrt(t_var);

  return {t_min * scale, t_max * scale, t_avg * scale, t_std * scale};
}

} // namespace

void autotuneTransposeBackend(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                              const cudecompGridDescAutotuneOptions_t* options) {
  if (handle->rank == 0) printf("CUDECOMP: Running transpose autotuning...\n");
  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_start = MPI_Wtime();

  // Create cuda_events for intermediate timings
  std::vector<cudaEvent_t> events(5);
  for (auto& event : events) { CHECK_CUDA(cudaEventCreate(&event)); }

  bool autotune_comm = options->autotune_transpose_backend;
  bool autotune_pdims = (grid_desc->config.pdims[0] == 0 && grid_desc->config.pdims[1] == 0);

  std::vector<cudecompTransposeCommBackend_t> comm_backend_list;
  bool need_nvshmem = false;
  if (autotune_comm) {
    comm_backend_list = {CUDECOMP_TRANSPOSE_COMM_MPI_P2P, CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL,
                         CUDECOMP_TRANSPOSE_COMM_MPI_A2A};
    if (!options->disable_nccl_backends) {
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NCCL);
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NCCL_PL);
    }
#ifdef ENABLE_NVSHMEM
    if (!options->disable_nvshmem_backends) {
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NVSHMEM);
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL);
      need_nvshmem = true;
    }
#endif
  } else {
    comm_backend_list = {grid_desc->config.transpose_comm_backend};
#ifdef ENABLE_NVSHMEM
    if (transposeBackendRequiresNvshmem(comm_backend_list[0])) { need_nvshmem = true; }
#endif
  }

  std::vector<int> pdim0_list;
  if (autotune_pdims) {
    pdim0_list = getFactors(handle->nranks);
  } else {
    pdim0_list = {grid_desc->config.pdims[0]};
  }

  int32_t pdims_best[2]{grid_desc->config.pdims[0], grid_desc->config.pdims[1]};
  auto comm_backend_best = grid_desc->config.transpose_comm_backend;
  double t_best = 1e12;

  void* data = nullptr;
  void* data2 = nullptr;
  void* work = nullptr;
  void* work_nvshmem = nullptr;

  int64_t data_sz = 0;
  int64_t work_sz = 0;
  for (auto& pdim0 : pdim0_list) {
    grid_desc->config.pdims[0] = pdim0;
    grid_desc->config.pdims[1] = handle->nranks / pdim0;
    grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
    grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];

    cudecompPencilInfo_t pinfo_x, pinfo_y, pinfo_z;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, nullptr));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, nullptr));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, nullptr));

    // Skip any decompositions with empty pencils
    if (grid_desc->config.pdims[0] > std::min(grid_desc->config.gdims_dist[0], grid_desc->config.gdims_dist[1]) ||
        grid_desc->config.pdims[1] > std::min(grid_desc->config.gdims_dist[1], grid_desc->config.gdims_dist[2])) {
      continue;
    }

    // Skip any uneven decompositions, if disabled
    if (!options->allow_uneven_decompositions && (grid_desc->config.gdims_dist[0] % grid_desc->config.pdims[0] != 0 ||
                                                  grid_desc->config.gdims_dist[1] % grid_desc->config.pdims[0] != 0 ||
                                                  grid_desc->config.gdims_dist[1] % grid_desc->config.pdims[1] != 0 ||
                                                  grid_desc->config.gdims_dist[2] % grid_desc->config.pdims[1] != 0)) {
      continue;
    }

    // Allocate test data
    int64_t num_elements_work;
    CHECK_CUDECOMP(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &num_elements_work));
    int64_t dtype_size;
    CHECK_CUDECOMP(cudecompGetDataTypeSize(options->dtype, &dtype_size));
    int64_t data_sz_new = std::max(std::max(pinfo_x.size, pinfo_y.size), pinfo_z.size) * dtype_size;
    int64_t work_sz_new = num_elements_work * dtype_size;
    if (data_sz_new > data_sz) {
      data_sz = data_sz_new;
      if (data) CHECK_CUDA(cudaFree(data));
      CHECK_CUDA(cudaMalloc(&data, data_sz));
      if (!options->transpose_use_inplace_buffers) {
        if (data2) CHECK_CUDA(cudaFree(data2));
        CHECK_CUDA(cudaMalloc(&data2, data_sz));
      }
    }

    // For nvshmem, buffers must be the same size. Find global maximums.
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &work_sz_new, 1, MPI_LONG_LONG_INT, MPI_MAX, handle->mpi_comm));

    if (work_sz_new > work_sz) {
      work_sz = work_sz_new;
      if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
        if (work && work != work_nvshmem) CHECK_CUDA(cudaFree(work));
        // Temporarily set backend to force nvshmem_malloc patch in cudecompMalloc/Free
        auto tmp = grid_desc->config.transpose_comm_backend;
        grid_desc->config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NVSHMEM;
        if (work_nvshmem) CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_nvshmem), work_sz));
        grid_desc->config.transpose_comm_backend = tmp;

        auto ret = cudaMalloc(&work, work_sz);
        if (ret == cudaErrorMemoryAllocation) {
          if (handle->rank == 0) {
            printf("CUDECOMP:WARN: Cannot allocate separate workspace for non-NVSHMEM backends during "
                   "autotuning. Using NVSHMEM allocated workspace for all backends, which may cause issues "
                   "for some MPI implementations. See documentation for more details and suggested workarounds.\n");
          }
          work = work_nvshmem;
          cudaGetLastError(); // Reset CUDA error state
        } else {
          CHECK_CUDA(ret);
        }
#endif
      } else {
        if (work) CHECK_CUDA(cudaFree(work));
        CHECK_CUDA(cudaMalloc(&work, work_sz));
      }
    }

    // Create test row and column communicators
    int color_row = grid_desc->pidx[0];
    MPI_Comm row_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_row, handle->rank, &row_comm));
    setCommInfo(handle, grid_desc, row_comm, CUDECOMP_COMM_ROW);

    int color_col = grid_desc->pidx[1];
    MPI_Comm col_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_col, handle->rank, &col_comm));
    setCommInfo(handle, grid_desc, col_comm, CUDECOMP_COMM_COL);
    if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
      nvshmem_team_config_t tmp;
      nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, grid_desc->config.pdims[1], &tmp, 0,
                            &grid_desc->row_comm_info.nvshmem_team, &tmp, 0, &grid_desc->col_comm_info.nvshmem_team);
#endif
    }

    for (auto& comm : comm_backend_list) {
      grid_desc->config.transpose_comm_backend = comm;
      void* din = data;
      void* dout = data2;
      if (options->transpose_use_inplace_buffers) { dout = data; }
      void* w = work;
#ifdef ENABLE_NVSHMEM
      if (transposeBackendRequiresNvshmem(comm)) { w = work_nvshmem; }
#endif

      // Warmup
      for (int i = 0; i < NWARMUP; ++i) {
        CHECK_CUDECOMP(cudecompTransposeXToY(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDECOMP(cudecompTransposeYToZ(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDECOMP(cudecompTransposeZToY(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDECOMP(cudecompTransposeYToX(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
      }

      // Trials
      std::vector<double> trial_times(NTRIALS);
      std::vector<float> trial_xy_times(NTRIALS);
      std::vector<float> trial_yz_times(NTRIALS);
      std::vector<float> trial_zy_times(NTRIALS);
      std::vector<float> trial_yx_times(NTRIALS);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_MPI(MPI_Barrier(handle->mpi_comm));
      double ts = MPI_Wtime();
      for (int i = 0; i < NTRIALS; ++i) {
        CHECK_CUDA(cudaEventRecord(events[0], 0));
        CHECK_CUDECOMP(cudecompTransposeXToY(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDA(cudaEventRecord(events[1], 0));
        CHECK_CUDECOMP(cudecompTransposeYToZ(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDA(cudaEventRecord(events[2], 0));
        CHECK_CUDECOMP(cudecompTransposeZToY(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDA(cudaEventRecord(events[3], 0));
        CHECK_CUDECOMP(cudecompTransposeYToX(handle, grid_desc, din, dout, w, options->dtype, nullptr, nullptr, 0));
        CHECK_CUDA(cudaEventRecord(events[4], 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(handle->mpi_comm));
        double te = MPI_Wtime();
        trial_times[i] = te - ts;
        CHECK_CUDA(cudaEventElapsedTime(&trial_xy_times[i], events[0], events[1]));
        CHECK_CUDA(cudaEventElapsedTime(&trial_yz_times[i], events[1], events[2]));
        CHECK_CUDA(cudaEventElapsedTime(&trial_zy_times[i], events[2], events[3]));
        CHECK_CUDA(cudaEventElapsedTime(&trial_yx_times[i], events[3], events[4]));
        ts = te;
      }
      auto times = processTimings(handle, trial_times, 1000.);
      auto xy_times = processTimings(handle, trial_xy_times);
      auto yz_times = processTimings(handle, trial_yz_times);
      auto zy_times = processTimings(handle, trial_zy_times);
      auto yx_times = processTimings(handle, trial_yx_times);

      if (handle->rank == 0) {
        printf("CUDECOMP:\tgrid: %d x %d, backend: %s \n"
               "CUDECOMP:\tTotal time min/max/avg/std [ms]: %f/%f/%f/%f\n"
               "CUDECOMP:\tTransposeXY time min/max/avg/std [ms]: %f/%f/%f/%f\n"
               "CUDECOMP:\tTransposeYZ time min/max/avg/std [ms]: %f/%f/%f/%f\n"
               "CUDECOMP:\tTransposeZY time min/max/avg/std [ms]: %f/%f/%f/%f\n"
               "CUDECOMP:\tTransposeYX time min/max/avg/std [ms]: %f/%f/%f/%f\n",
               grid_desc->config.pdims[0], grid_desc->config.pdims[1],
               cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend), times[0], times[1],
               times[2], times[3], xy_times[0], xy_times[1], xy_times[2], xy_times[3], yz_times[0], yz_times[1],
               yz_times[2], yz_times[3], zy_times[0], zy_times[1], zy_times[2], zy_times[3], yx_times[0], yx_times[1],
               yx_times[2], yx_times[3]);
      }

      if (times[2] < t_best) {
        pdims_best[0] = grid_desc->config.pdims[0];
        pdims_best[1] = grid_desc->config.pdims[1];
        comm_backend_best = grid_desc->config.transpose_comm_backend;
        t_best = times[2];
      }
    }

    // Destroy test communicators
    CHECK_MPI(MPI_Comm_free(&grid_desc->row_comm_info.mpi_comm));
    CHECK_MPI(MPI_Comm_free(&grid_desc->col_comm_info.mpi_comm));
    if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
      nvshmem_team_destroy(grid_desc->row_comm_info.nvshmem_team);
      nvshmem_team_destroy(grid_desc->col_comm_info.nvshmem_team);
      grid_desc->row_comm_info.nvshmem_team = NVSHMEM_TEAM_INVALID;
      grid_desc->col_comm_info.nvshmem_team = NVSHMEM_TEAM_INVALID;
#endif
    }
  }

  // Free test data and workspace
  if (need_nvshmem) {
    if (work != work_nvshmem) { CHECK_CUDA(cudaFree(work)); }
#ifdef ENABLE_NVSHMEM
    // Temporarily set backend to force nvshmem_malloc patch in cudecompMalloc/Free
    auto tmp = grid_desc->config.transpose_comm_backend;
    grid_desc->config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NVSHMEM;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
    grid_desc->config.transpose_comm_backend = tmp;
#endif
  } else {
    CHECK_CUDA(cudaFree(work));
  }

  CHECK_CUDA(cudaFree(data));
  if (!options->transpose_use_inplace_buffers) { CHECK_CUDA(cudaFree(data2)); }

  // Delete cuda events
  for (auto& event : events) { CHECK_CUDA(cudaEventDestroy(event)); }

  // Set handle to best option (broadcast from rank 0 for consistency)
  CHECK_MPI(MPI_Bcast(&comm_backend_best, sizeof(cudecompTransposeCommBackend_t), MPI_CHAR, 0, handle->mpi_comm));
  CHECK_MPI(MPI_Bcast(pdims_best, 2 * sizeof(int), MPI_INT, 0, handle->mpi_comm));

  grid_desc->config.transpose_comm_backend = comm_backend_best;
  grid_desc->config.pdims[0] = pdims_best[0];
  grid_desc->config.pdims[1] = pdims_best[1];

  if (handle->rank == 0) {
    printf("CUDECOMP: SELECTED: grid: %d x %d, backend: %s, Avg. time %f\n", grid_desc->config.pdims[0],
           grid_desc->config.pdims[1], cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend),
           t_best);
  }

  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_end = MPI_Wtime();
  if (handle->rank == 0) printf("CUDECOMP: transpose autotuning time [s]: %f\n", t_end - t_start);
}

void autotuneHaloBackend(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                         const cudecompGridDescAutotuneOptions_t* options) {
  if (handle->rank == 0) {
    printf("CUDECOMP: Running halo autotuning...\n");
    printf("CUDECOMP: Autotune halo axis: %s\n", options->halo_axis == 0 ? "x" : (options->halo_axis == 1 ? "y" : "z"));
  }
  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_start = MPI_Wtime();

  bool autotune_comm = options->autotune_halo_backend;
  bool autotune_pdims = (grid_desc->config.pdims[0] == 0 && grid_desc->config.pdims[1] == 0);

  std::vector<cudecompHaloCommBackend_t> comm_backend_list;
  bool need_nvshmem = false;
  if (autotune_comm) {
    comm_backend_list = {CUDECOMP_HALO_COMM_MPI, CUDECOMP_HALO_COMM_MPI_BLOCKING};
    if (!options->disable_nccl_backends) { comm_backend_list.push_back(CUDECOMP_HALO_COMM_NCCL); }
#ifdef ENABLE_NVSHMEM
    if (!options->disable_nvshmem_backends) {
      comm_backend_list.push_back(CUDECOMP_HALO_COMM_NVSHMEM);
      comm_backend_list.push_back(CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING);
      need_nvshmem = true;
    }
#endif
  } else {
    comm_backend_list = {grid_desc->config.halo_comm_backend};
#ifdef ENABLE_NVSHMEM
    if (haloBackendRequiresNvshmem(comm_backend_list[0])) { need_nvshmem = true; }
#endif
  }

  std::vector<int> pdim0_list;
  if (autotune_pdims) {
    pdim0_list = getFactors(handle->nranks);
  } else {
    pdim0_list = {grid_desc->config.pdims[0]};
  }

  int32_t pdims_best[2]{grid_desc->config.pdims[0], grid_desc->config.pdims[1]};
  auto comm_backend_best = grid_desc->config.halo_comm_backend;
  double t_best = 1e12;

  void* data = nullptr;
  void* work = nullptr;
  void* work_nvshmem = nullptr;

  int64_t data_sz = 0;
  int64_t work_sz = 0;
  for (auto& pdim0 : pdim0_list) {
    grid_desc->config.pdims[0] = pdim0;
    grid_desc->config.pdims[1] = handle->nranks / pdim0;
    grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
    grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];

    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, options->halo_axis, options->halo_extents));

    // Skip any decompositions with empty pencils
    if (std::max(grid_desc->config.pdims[0], grid_desc->config.pdims[1]) >
        std::min(grid_desc->config.gdims[1], grid_desc->config.gdims[2])) {
      continue;
    }

    // Skip any uneven decompositions, if disabled
    if (!options->allow_uneven_decompositions && (grid_desc->config.gdims_dist[0] % grid_desc->config.pdims[0] != 0 ||
                                                  grid_desc->config.gdims_dist[1] % grid_desc->config.pdims[0] != 0 ||
                                                  grid_desc->config.gdims_dist[1] % grid_desc->config.pdims[1] != 0 ||
                                                  grid_desc->config.gdims_dist[2] % grid_desc->config.pdims[1] != 0)) {
      continue;
    }

    // Allocate test data
    int64_t num_elements_work;
    CHECK_CUDECOMP(
        cudecompGetHaloWorkspaceSize(handle, grid_desc, options->halo_axis, options->halo_extents, &num_elements_work));
    int64_t dtype_size;
    CHECK_CUDECOMP(cudecompGetDataTypeSize(options->dtype, &dtype_size));
    int64_t data_sz_new = pinfo.size * dtype_size;
    int64_t work_sz_new = num_elements_work * dtype_size;
    if (data_sz_new > data_sz) {
      data_sz = data_sz_new;
      if (data) CHECK_CUDA(cudaFree(data));
      CHECK_CUDA(cudaMalloc(&data, data_sz));
    }

    // For nvshmem, buffers must be the same size. Find global maximums.
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &work_sz_new, 1, MPI_LONG_LONG_INT, MPI_MAX, handle->mpi_comm));

    if (work_sz_new > work_sz) {
      work_sz = work_sz_new;
      if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
        if (work && work != work_nvshmem) CHECK_CUDA(cudaFree(work));
        // Temporarily set backend to force nvshmem_malloc patch in cudecompMalloc/Free
        auto tmp = grid_desc->config.halo_comm_backend;
        grid_desc->config.halo_comm_backend = CUDECOMP_HALO_COMM_NVSHMEM;
        if (work_nvshmem) CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_nvshmem), work_sz));
        grid_desc->config.halo_comm_backend = tmp;

        auto ret = cudaMalloc(&work, work_sz);
        if (ret == cudaErrorMemoryAllocation) {
          if (handle->rank == 0) {
            printf("CUDECOMP:WARN: Cannot allocate separate workspace for non-NVSHMEM backends during "
                   "autotuning. Using NVSHMEM allocated workspace for all backends, which may cause issues "
                   "for some MPI implementations. See documentation for more details and suggested workarounds.\n");
          }
          work = work_nvshmem;
          cudaGetLastError(); // Reset CUDA error state
        } else {
          CHECK_CUDA(ret);
        }
#endif
      } else {
        if (work) CHECK_CUDA(cudaFree(work));
        CHECK_CUDA(cudaMalloc(&work, work_sz));
      }
    }

    // Create test row and column communicators
    int color_row = grid_desc->pidx[0];
    MPI_Comm row_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_row, handle->rank, &row_comm));
    setCommInfo(handle, grid_desc, row_comm, CUDECOMP_COMM_ROW);

    int color_col = grid_desc->pidx[1];
    MPI_Comm col_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_col, handle->rank, &col_comm));
    setCommInfo(handle, grid_desc, col_comm, CUDECOMP_COMM_COL);
    if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
      nvshmem_team_config_t tmp;
      nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, grid_desc->config.pdims[1], &tmp, 0,
                            &grid_desc->row_comm_info.nvshmem_team, &tmp, 0, &grid_desc->col_comm_info.nvshmem_team);
#endif
    }

    for (auto& comm : comm_backend_list) {
      grid_desc->config.halo_comm_backend = comm;
      void* d = data;
      void* w = work;
#ifdef ENABLE_NVSHMEM
      if (haloBackendRequiresNvshmem(comm)) { w = work_nvshmem; }
#endif

      // Warmup
      for (int i = 0; i < NWARMUP; ++i) {
        for (int dim = 0; dim < 3; ++dim) {
          switch (options->halo_axis) {
          case 0:
            CHECK_CUDECOMP(cudecompUpdateHalosX(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          case 1:
            CHECK_CUDECOMP(cudecompUpdateHalosY(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          case 2:
            CHECK_CUDECOMP(cudecompUpdateHalosZ(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          }
        }
      }

      // Trials
      std::vector<double> trial_times(NTRIALS);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_MPI(MPI_Barrier(handle->mpi_comm));
      double ts = MPI_Wtime();
      for (int i = 0; i < NTRIALS; ++i) {
        for (int dim = 0; dim < 3; ++dim) {
          switch (options->halo_axis) {
          case 0:
            CHECK_CUDECOMP(cudecompUpdateHalosX(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          case 1:
            CHECK_CUDECOMP(cudecompUpdateHalosY(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          case 2:
            CHECK_CUDECOMP(cudecompUpdateHalosZ(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, 0));
            break;
          }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(handle->mpi_comm));
        double te = MPI_Wtime();
        trial_times[i] = te - ts;
        ts = te;
      }
      auto times = processTimings(handle, trial_times, 1000.);

      if (handle->rank == 0) {
        printf("CUDECOMP:\tgrid: %d x %d, halo backend: %s \n"
               "CUDECOMP:\tTotal time min/max/avg/std [ms]: %f/%f/%f/%f\n",
               grid_desc->config.pdims[0], grid_desc->config.pdims[1],
               cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend), times[0], times[1], times[2],
               times[3]);
      }

      if (times[2] < t_best) {
        pdims_best[0] = grid_desc->config.pdims[0];
        pdims_best[1] = grid_desc->config.pdims[1];
        comm_backend_best = grid_desc->config.halo_comm_backend;
        t_best = times[2];
      }
    }

    // Destroy test communicators
    CHECK_MPI(MPI_Comm_free(&grid_desc->row_comm_info.mpi_comm));
    CHECK_MPI(MPI_Comm_free(&grid_desc->col_comm_info.mpi_comm));
    if (need_nvshmem) {
#ifdef ENABLE_NVSHMEM
      nvshmem_team_destroy(grid_desc->row_comm_info.nvshmem_team);
      nvshmem_team_destroy(grid_desc->col_comm_info.nvshmem_team);
      grid_desc->row_comm_info.nvshmem_team = NVSHMEM_TEAM_INVALID;
      grid_desc->col_comm_info.nvshmem_team = NVSHMEM_TEAM_INVALID;
#endif
    }
  }

  // Free test data and workspace
  if (need_nvshmem) {
    if (work != work_nvshmem) { CHECK_CUDA(cudaFree(work)); }
#ifdef ENABLE_NVSHMEM
    // Temporarily set backend to force nvshmem_malloc patch in cudecompMalloc/Free
    auto tmp = grid_desc->config.halo_comm_backend;
    grid_desc->config.halo_comm_backend = CUDECOMP_HALO_COMM_NVSHMEM;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
    grid_desc->config.halo_comm_backend = tmp;
#endif
  } else {
    CHECK_CUDA(cudaFree(work));
  }

  CHECK_CUDA(cudaFree(data));

  // Set handle to best option (broadcast from rank 0 for consistency)
  CHECK_MPI(MPI_Bcast(&comm_backend_best, sizeof(cudecompHaloCommBackend_t), MPI_CHAR, 0, handle->mpi_comm));
  CHECK_MPI(MPI_Bcast(pdims_best, 2 * sizeof(int), MPI_INT, 0, handle->mpi_comm));

  grid_desc->config.halo_comm_backend = comm_backend_best;
  grid_desc->config.pdims[0] = pdims_best[0];
  grid_desc->config.pdims[1] = pdims_best[1];

  if (handle->rank == 0) {
    printf("CUDECOMP: SELECTED: grid: %d x %d, halo backend: %s, Avg. time [s] %f\n", grid_desc->config.pdims[0],
           grid_desc->config.pdims[1], cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend), t_best);
  }

  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_end = MPI_Wtime();
  if (handle->rank == 0) printf("CUDECOMP: halo autotuning time [s]: %f\n", t_end - t_start);
}

} // namespace cudecomp
