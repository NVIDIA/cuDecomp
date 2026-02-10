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
#include "internal/performance.h"
#include "internal/transpose.h"

namespace cudecomp {
namespace {

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

  for (auto& t : times) {
    t = (t - t_avg) * (t - t_avg);
  }
  double t_var = std::accumulate(times.begin(), times.end(), T(0)) / times.size();
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_var, 1, MPI_DOUBLE, MPI_SUM, handle->mpi_comm));
  t_var /= handle->nranks;
  double t_std = std::sqrt(t_var);

  return {static_cast<T>(t_min) * scale, static_cast<T>(t_max) * scale, static_cast<T>(t_avg) * scale,
          static_cast<T>(t_std) * scale};
}

} // namespace

void autotuneTransposeBackend(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                              const cudecompGridDescAutotuneOptions_t* options) {
  if (handle->rank == 0) printf("CUDECOMP: Running transpose autotuning...\n");
  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_start = MPI_Wtime();

  // Create cuda_events for intermediate timings
  std::vector<cudaEvent_t> events(5);
  for (auto& event : events) {
    CHECK_CUDA(cudaEventCreate(&event));
  }

  bool autotune_comm = options->autotune_transpose_backend;
  bool autotune_pdims = (grid_desc->config.pdims[0] == 0 && grid_desc->config.pdims[1] == 0);

  std::vector<cudecompTransposeCommBackend_t> comm_backend_list;
  bool need_nccl = false;
  bool need_nvshmem = false;
  if (autotune_comm) {
    comm_backend_list = {CUDECOMP_TRANSPOSE_COMM_MPI_P2P, CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL,
                         CUDECOMP_TRANSPOSE_COMM_MPI_A2A};
    if (!options->disable_nccl_backends) {
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NCCL);
      comm_backend_list.push_back(CUDECOMP_TRANSPOSE_COMM_NCCL_PL);
      need_nccl = true;
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
    if (transposeBackendRequiresNccl(comm_backend_list[0])) { need_nccl = true; }
#ifdef ENABLE_NVSHMEM
    if (transposeBackendRequiresNvshmem(comm_backend_list[0])) { need_nvshmem = true; }
#endif
  }

  bool need_data2 = false;
  for (int i = 0; i < 4; ++i) {
    if (!options->transpose_use_inplace_buffers[i]) need_data2 = true;
  }

  std::vector<int> pdim1_list;
  if (autotune_pdims) {
    pdim1_list = getFactors(handle->nranks);
  } else {
    pdim1_list = {grid_desc->config.pdims[1]};
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
  for (auto& pdim1 : pdim1_list) {
    grid_desc->config.pdims[0] = handle->nranks / pdim1;
    grid_desc->config.pdims[1] = pdim1;
    if (handle->use_col_major_rank_order) {
      grid_desc->pidx[0] = handle->rank % grid_desc->config.pdims[0];
      grid_desc->pidx[1] = handle->rank / grid_desc->config.pdims[0];
    } else {
      grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
      grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];
    }

    cudecompPencilInfo_t pinfo_x0, pinfo_x3;
    cudecompPencilInfo_t pinfo_y0, pinfo_y1, pinfo_y2, pinfo_y3;
    cudecompPencilInfo_t pinfo_z1, pinfo_z2;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x0, 0, options->transpose_input_halo_extents[0],
                                         options->transpose_input_padding[0]));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x3, 0, options->transpose_output_halo_extents[3],
                                         options->transpose_output_padding[3]));

    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y0, 1, options->transpose_output_halo_extents[0],
                                         options->transpose_output_padding[0]));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y1, 1, options->transpose_input_halo_extents[1],
                                         options->transpose_input_padding[1]));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y2, 1, options->transpose_output_halo_extents[2],
                                         options->transpose_output_padding[2]));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y3, 1, options->transpose_input_halo_extents[3],
                                         options->transpose_input_padding[3]));

    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z1, 2, options->transpose_output_halo_extents[1],
                                         options->transpose_output_padding[1]));
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z2, 2, options->transpose_input_halo_extents[2],
                                         options->transpose_input_padding[2]));

    // Skip any decompositions with empty pencils, if disabled
    if (!options->allow_empty_pencils &&
        (grid_desc->config.pdims[0] > std::min(grid_desc->config.gdims_dist[0], grid_desc->config.gdims_dist[1]) ||
         grid_desc->config.pdims[1] > std::min(grid_desc->config.gdims_dist[1], grid_desc->config.gdims_dist[2]))) {
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
    int64_t size_x = std::max(pinfo_x0.size, pinfo_x3.size);
    int64_t size_y = std::max(std::max(std::max(pinfo_y0.size, pinfo_y1.size), pinfo_y2.size), pinfo_y3.size);
    int64_t size_z = std::max(pinfo_z1.size, pinfo_z2.size);
    int64_t data_sz_new = std::max(std::max(size_x, size_y), size_z) * dtype_size;
    int64_t work_sz_new = num_elements_work * dtype_size;
    if (data_sz_new > data_sz) {
      data_sz = data_sz_new;
      if (data) CHECK_CUDA(cudaFree(data));
      CHECK_CUDA(cudaMalloc(&data, data_sz));
      if (need_data2) {
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
        if (work && work != work_nvshmem) {
          auto tmp = grid_desc->config.transpose_comm_backend;
          grid_desc->config.transpose_comm_backend =
              (need_nccl) ? CUDECOMP_TRANSPOSE_COMM_NCCL : CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
          CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
          grid_desc->config.transpose_comm_backend = tmp;
        }
        // Temporarily set backend to force nvshmem_malloc path in cudecompMalloc/Free
        auto tmp = grid_desc->config.transpose_comm_backend;
        grid_desc->config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NVSHMEM;
        if (work_nvshmem) CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_nvshmem), work_sz));
        grid_desc->config.transpose_comm_backend = tmp;

        // Check if there is enough memory for separate non-NVSHMEM allocated work buffer
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
          CHECK_CUDA(cudaFree(work));
          auto tmp = grid_desc->config.transpose_comm_backend;
          grid_desc->config.transpose_comm_backend =
              (need_nccl) ? CUDECOMP_TRANSPOSE_COMM_NCCL : CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
          cudecompResult_t ret = cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work), work_sz);
          grid_desc->config.transpose_comm_backend = tmp;
        }
#endif
      } else {
        auto tmp = grid_desc->config.transpose_comm_backend;
        grid_desc->config.transpose_comm_backend =
            (need_nccl) ? CUDECOMP_TRANSPOSE_COMM_NCCL : CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
        if (work) { CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work)); }
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work), work_sz));
        grid_desc->config.transpose_comm_backend = tmp;
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
      void* w = work;
#ifdef ENABLE_NVSHMEM
      if (transposeBackendRequiresNvshmem(comm)) { w = work_nvshmem; }
#endif

      // Reset performance samples
      resetPerformanceSamples(handle, grid_desc);

      // Warmup
      for (int i = 0; i < options->n_warmup_trials; ++i) {
        if (options->transpose_op_weights[0] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeXToY(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[0] ? data : data2, w,
                                               options->dtype, pinfo_x0.halo_extents, pinfo_y0.halo_extents,
                                               pinfo_x0.padding, pinfo_y0.padding, 0));
        }
        if (options->transpose_op_weights[1] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeYToZ(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[1] ? data : data2, w,
                                               options->dtype, pinfo_y1.halo_extents, pinfo_z1.halo_extents,
                                               pinfo_y1.padding, pinfo_z1.padding, 0));
        }
        if (options->transpose_op_weights[2] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeZToY(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[2] ? data : data2, w,
                                               options->dtype, pinfo_z2.halo_extents, pinfo_y2.halo_extents,
                                               pinfo_z2.padding, pinfo_y2.padding, 0));
        }
        if (options->transpose_op_weights[3] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeYToX(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[3] ? data : data2, w,
                                               options->dtype, pinfo_y3.halo_extents, pinfo_x3.halo_extents,
                                               pinfo_y3.padding, pinfo_x3.padding, 0));
        }
      }

      // Trials
      std::vector<float> trial_times(options->n_trials);
      std::vector<float> trial_times_w(options->n_trials);
      std::vector<float> trial_xy_times(options->n_trials);
      std::vector<float> trial_yz_times(options->n_trials);
      std::vector<float> trial_zy_times(options->n_trials);
      std::vector<float> trial_yx_times(options->n_trials);
      bool skip_case = false;
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_MPI(MPI_Barrier(handle->mpi_comm));
      double ts = MPI_Wtime();
      for (int i = 0; i < options->n_trials; ++i) {
        CHECK_CUDA(cudaEventRecord(events[0], 0));
        if (options->transpose_op_weights[0] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeXToY(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[0] ? data : data2, w,
                                               options->dtype, pinfo_x0.halo_extents, pinfo_y0.halo_extents,
                                               pinfo_x0.padding, pinfo_y0.padding, 0));
        }
        CHECK_CUDA(cudaEventRecord(events[1], 0));
        if (options->transpose_op_weights[1] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeYToZ(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[1] ? data : data2, w,
                                               options->dtype, pinfo_y1.halo_extents, pinfo_z1.halo_extents,
                                               pinfo_y1.padding, pinfo_z1.padding, 0));
        }
        CHECK_CUDA(cudaEventRecord(events[2], 0));
        if (options->transpose_op_weights[2] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeZToY(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[2] ? data : data2, w,
                                               options->dtype, pinfo_z2.halo_extents, pinfo_y2.halo_extents,
                                               pinfo_z2.padding, pinfo_y2.padding, 0));
        }
        CHECK_CUDA(cudaEventRecord(events[3], 0));
        if (options->transpose_op_weights[3] != 0.0) {
          CHECK_CUDECOMP(cudecompTransposeYToX(handle, grid_desc, data,
                                               options->transpose_use_inplace_buffers[3] ? data : data2, w,
                                               options->dtype, pinfo_y3.halo_extents, pinfo_x3.halo_extents,
                                               pinfo_y3.padding, pinfo_x3.padding, 0));
        }
        CHECK_CUDA(cudaEventRecord(events[4], 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(handle->mpi_comm));

        if (options->transpose_op_weights[0] != 0.0)
          CHECK_CUDA(cudaEventElapsedTime(&trial_xy_times[i], events[0], events[1]));
        if (options->transpose_op_weights[1] != 0.0)
          CHECK_CUDA(cudaEventElapsedTime(&trial_yz_times[i], events[1], events[2]));
        if (options->transpose_op_weights[2] != 0.0)
          CHECK_CUDA(cudaEventElapsedTime(&trial_zy_times[i], events[2], events[3]));
        if (options->transpose_op_weights[3] != 0.0)
          CHECK_CUDA(cudaEventElapsedTime(&trial_yx_times[i], events[3], events[4]));

        trial_times[i] = trial_xy_times[i] + trial_yz_times[i] + trial_zy_times[i] + trial_yx_times[i];

        trial_times_w[i] = options->transpose_op_weights[0] * trial_xy_times[i] +
                           options->transpose_op_weights[1] * trial_yz_times[i] +
                           options->transpose_op_weights[2] * trial_zy_times[i] +
                           options->transpose_op_weights[3] * trial_yx_times[i];

        if (i == 0) {
          std::vector<float> t0 = {trial_times_w[0]};
          auto times = processTimings(handle, t0);
          auto t_avg = times[2];

          if (options->skip_threshold * t_avg > t_best) {
            // Performance of first iteration of this configuration meets skip threshold. Skipping.
            skip_case = true;
            break;
          }
        }
      }

      // Clear CUDA graph cache between backend/process decomposition pairs
      grid_desc->graph_cache.clear();

      auto times = processTimings(handle, trial_times);
      auto times_w = processTimings(handle, trial_times_w);
      auto xy_times = processTimings(handle, trial_xy_times);
      auto yz_times = processTimings(handle, trial_yz_times);
      auto zy_times = processTimings(handle, trial_zy_times);
      auto yx_times = processTimings(handle, trial_yx_times);

      const char* t_skipped[4];
      for (int i = 0; i < 4; ++i) {
        if (options->transpose_op_weights[i] == 0.0) {
          t_skipped[i] = " (skipped)";
        } else {
          t_skipped[i] = "";
        }
      }

      if (handle->rank == 0) {
        if (skip_case) {
          printf("CUDECOMP:\tgrid: %d x %d, backend: %s \n"
                 "CUDECOMP:\t(skipped) \n",
                 grid_desc->config.pdims[0], grid_desc->config.pdims[1],
                 cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend));
        } else {
          printf("CUDECOMP:\tgrid: %d x %d, backend: %s \n"
                 "CUDECOMP:\tTotal time min/max/avg/std [ms]: %f/%f/%f/%f\n"
                 "CUDECOMP:\t           min/max/avg/std [ms]: %f/%f/%f/%f (weighted)\n"
                 "CUDECOMP:\tTransposeXY time min/max/avg/std [ms]: %f/%f/%f/%f%s\n"
                 "CUDECOMP:\tTransposeYZ time min/max/avg/std [ms]: %f/%f/%f/%f%s\n"
                 "CUDECOMP:\tTransposeZY time min/max/avg/std [ms]: %f/%f/%f/%f%s\n"
                 "CUDECOMP:\tTransposeYX time min/max/avg/std [ms]: %f/%f/%f/%f%s\n",
                 grid_desc->config.pdims[0], grid_desc->config.pdims[1],
                 cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend), times[0], times[1],
                 times[2], times[3], times_w[0], times_w[1], times_w[2], times_w[3], xy_times[0], xy_times[1],
                 xy_times[2], xy_times[3], t_skipped[0], yz_times[0], yz_times[1], yz_times[2], yz_times[3],
                 t_skipped[1], zy_times[0], zy_times[1], zy_times[2], zy_times[3], t_skipped[2], yx_times[0],
                 yx_times[1], yx_times[2], yx_times[3], t_skipped[3]);
        }
      }

      // Print performance report for this configuration if enabled
      if (handle->performance_report_enable && !skip_case) { printPerformanceReport(handle, grid_desc); }

      if (skip_case) continue;

      if (times_w[2] < t_best) {
        pdims_best[0] = grid_desc->config.pdims[0];
        pdims_best[1] = grid_desc->config.pdims[1];
        comm_backend_best = grid_desc->config.transpose_comm_backend;
        t_best = times_w[2];
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
    if (work != work_nvshmem) {
      auto tmp = grid_desc->config.transpose_comm_backend;
      grid_desc->config.transpose_comm_backend =
          (need_nccl) ? CUDECOMP_TRANSPOSE_COMM_NCCL : CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
      CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
      grid_desc->config.transpose_comm_backend = tmp;
    }
#ifdef ENABLE_NVSHMEM
    // Temporarily set backend to force nvshmem_malloc path in cudecompMalloc/Free
    auto tmp = grid_desc->config.transpose_comm_backend;
    grid_desc->config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NVSHMEM;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
    grid_desc->config.transpose_comm_backend = tmp;
#endif
  } else {
    auto tmp = grid_desc->config.transpose_comm_backend;
    grid_desc->config.transpose_comm_backend =
        (need_nccl) ? CUDECOMP_TRANSPOSE_COMM_NCCL : CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
    grid_desc->config.transpose_comm_backend = tmp;
  }

  CHECK_CUDA(cudaFree(data));
  if (need_data2) { CHECK_CUDA(cudaFree(data2)); }

  // Delete cuda events
  for (auto& event : events) {
    CHECK_CUDA(cudaEventDestroy(event));
  }

  // Set handle to best option (broadcast from rank 0 for consistency)
  CHECK_MPI(MPI_Bcast(&comm_backend_best, sizeof(cudecompTransposeCommBackend_t), MPI_CHAR, 0, handle->mpi_comm));
  CHECK_MPI(MPI_Bcast(pdims_best, 2 * sizeof(int), MPI_INT, 0, handle->mpi_comm));

  grid_desc->config.transpose_comm_backend = comm_backend_best;
  grid_desc->config.pdims[0] = pdims_best[0];
  grid_desc->config.pdims[1] = pdims_best[1];

  if (handle->rank == 0) {
    printf("CUDECOMP: SELECTED: grid: %d x %d, backend: %s, Avg. time (weighted) [ms]: %f\n",
           grid_desc->config.pdims[0], grid_desc->config.pdims[1],
           cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend), t_best);
  }

  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_end = MPI_Wtime();
  if (handle->rank == 0) printf("CUDECOMP: transpose autotuning time [s]: %f\n", t_end - t_start);

  // Perform an MPI_Alltoall on small (~1 MiB) buffers to force some MPI backends (e.g. UCX) to release stale
  // registration handles on larger test data buffers. These stale registration handles can cause to delays in freeing
  // the test buffer GPU memory.
  char *tmp1, *tmp2;
  size_t per_rank_size = (1024 * 1024 + handle->nranks - 1) / handle->nranks;
  CHECK_CUDA(cudaMalloc(&tmp1, per_rank_size * handle->nranks * sizeof(*tmp1)));
  CHECK_CUDA(cudaMalloc(&tmp2, per_rank_size * handle->nranks * sizeof(*tmp2)));
  CHECK_MPI(MPI_Alltoall(tmp1, per_rank_size, MPI_CHAR, tmp2, per_rank_size, MPI_CHAR, handle->mpi_comm));
  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  CHECK_CUDA(cudaFree(tmp1));
  CHECK_CUDA(cudaFree(tmp2));

  // Reset performance samples after autotuning
  resetPerformanceSamples(handle, grid_desc);
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
  bool need_nccl = false;
  bool need_nvshmem = false;
  if (autotune_comm) {
    comm_backend_list = {CUDECOMP_HALO_COMM_MPI, CUDECOMP_HALO_COMM_MPI_BLOCKING};
    if (!options->disable_nccl_backends) {
      comm_backend_list.push_back(CUDECOMP_HALO_COMM_NCCL);
      need_nccl = true;
    }
#ifdef ENABLE_NVSHMEM
    if (!options->disable_nvshmem_backends) {
      comm_backend_list.push_back(CUDECOMP_HALO_COMM_NVSHMEM);
      comm_backend_list.push_back(CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING);
      need_nvshmem = true;
    }
#endif
  } else {
    comm_backend_list = {grid_desc->config.halo_comm_backend};
    if (haloBackendRequiresNccl(comm_backend_list[0])) { need_nccl = true; }
#ifdef ENABLE_NVSHMEM
    if (haloBackendRequiresNvshmem(comm_backend_list[0])) { need_nvshmem = true; }
#endif
  }

  std::vector<int> pdim1_list;
  if (autotune_pdims) {
    pdim1_list = getFactors(handle->nranks);
  } else {
    pdim1_list = {grid_desc->config.pdims[1]};
  }

  int32_t pdims_best[2]{grid_desc->config.pdims[0], grid_desc->config.pdims[1]};
  auto comm_backend_best = grid_desc->config.halo_comm_backend;
  double t_best = 1e12;

  void* data = nullptr;
  void* work = nullptr;
  void* work_nvshmem = nullptr;

  int64_t data_sz = 0;
  int64_t work_sz = 0;
  for (auto& pdim1 : pdim1_list) {
    grid_desc->config.pdims[0] = handle->nranks / pdim1;
    grid_desc->config.pdims[1] = pdim1;
    if (handle->use_col_major_rank_order) {
      grid_desc->pidx[0] = handle->rank % grid_desc->config.pdims[0];
      grid_desc->pidx[1] = handle->rank / grid_desc->config.pdims[0];
    } else {
      grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
      grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];
    }

    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, options->halo_axis, options->halo_extents,
                                         options->halo_padding));

    // Skip any decompositions with empty pencils
    if ((options->halo_axis == 0 && (grid_desc->config.pdims[0] > grid_desc->config.gdims_dist[1] ||
                                     grid_desc->config.pdims[1] > grid_desc->config.gdims_dist[2])) ||
        (options->halo_axis == 1 && (grid_desc->config.pdims[0] > grid_desc->config.gdims_dist[0] ||
                                     grid_desc->config.pdims[1] > grid_desc->config.gdims_dist[2])) ||
        (options->halo_axis == 2 && (grid_desc->config.pdims[0] > grid_desc->config.gdims_dist[0] ||
                                     grid_desc->config.pdims[1] > grid_desc->config.gdims_dist[1]))) {
      if (options->allow_empty_pencils) {
        THROW_NOT_SUPPORTED("cannot perform halo autotuning on distributions with empty pencils");
      }
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
        if (work && work != work_nvshmem) {
          auto tmp = grid_desc->config.halo_comm_backend;
          grid_desc->config.halo_comm_backend = (need_nccl) ? CUDECOMP_HALO_COMM_NCCL : CUDECOMP_HALO_COMM_MPI;
          CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
          grid_desc->config.halo_comm_backend = tmp;
        }
        // Temporarily set backend to force nvshmem_malloc path in cudecompMalloc/Free
        auto tmp = grid_desc->config.halo_comm_backend;
        grid_desc->config.halo_comm_backend = CUDECOMP_HALO_COMM_NVSHMEM;
        if (work_nvshmem) CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work_nvshmem), work_sz));
        grid_desc->config.halo_comm_backend = tmp;

        // Check if there is enough memory for separate non-NVSHMEM allocated work buffer
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
          CHECK_CUDA(cudaFree(work));
          auto tmp = grid_desc->config.halo_comm_backend;
          grid_desc->config.halo_comm_backend = (need_nccl) ? CUDECOMP_HALO_COMM_NCCL : CUDECOMP_HALO_COMM_MPI;
          cudecompResult_t ret = cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work), work_sz);
          grid_desc->config.halo_comm_backend = tmp;
        }
#endif
      } else {
        auto tmp = grid_desc->config.halo_comm_backend;
        grid_desc->config.halo_comm_backend = (need_nccl) ? CUDECOMP_HALO_COMM_NCCL : CUDECOMP_HALO_COMM_MPI;
        if (work) { CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work)); }
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&work), work_sz));
        grid_desc->config.halo_comm_backend = tmp;
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

    bool skip_case = false;
    for (auto& comm : comm_backend_list) {
      grid_desc->config.halo_comm_backend = comm;
      void* d = data;
      void* w = work;
#ifdef ENABLE_NVSHMEM
      if (haloBackendRequiresNvshmem(comm)) { w = work_nvshmem; }
#endif

      // Reset performance samples
      resetPerformanceSamples(handle, grid_desc);

      // Warmup
      for (int i = 0; i < options->n_warmup_trials; ++i) {
        for (int dim = 0; dim < 3; ++dim) {
          switch (options->halo_axis) {
          case 0:
            CHECK_CUDECOMP(cudecompUpdateHalosX(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          case 1:
            CHECK_CUDECOMP(cudecompUpdateHalosY(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          case 2:
            CHECK_CUDECOMP(cudecompUpdateHalosZ(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          }
        }
      }

      // Trials
      std::vector<double> trial_times(options->n_trials);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_MPI(MPI_Barrier(handle->mpi_comm));
      double ts = MPI_Wtime();
      for (int i = 0; i < options->n_trials; ++i) {
        for (int dim = 0; dim < 3; ++dim) {
          switch (options->halo_axis) {
          case 0:
            CHECK_CUDECOMP(cudecompUpdateHalosX(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          case 1:
            CHECK_CUDECOMP(cudecompUpdateHalosY(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          case 2:
            CHECK_CUDECOMP(cudecompUpdateHalosZ(handle, grid_desc, d, w, options->dtype, pinfo.halo_extents,
                                                options->halo_periods, dim, pinfo.padding, 0));
            break;
          }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(handle->mpi_comm));
        double te = MPI_Wtime();
        trial_times[i] = te - ts;
        ts = te;

        if (i == 0) {
          double t_avg = trial_times[0];
          CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &t_avg, 1, MPI_DOUBLE, MPI_SUM, handle->mpi_comm));
          t_avg /= handle->nranks;

          if (options->skip_threshold * (t_avg * 1000.) > t_best) {
            // Performance of first iteration of this configuration meets skip threshold. Skipping.
            skip_case = true;
            break;
          }
        }
      }

      auto times = processTimings(handle, trial_times, 1000.);

      if (handle->rank == 0) {
        if (skip_case) {
          printf("CUDECOMP:\tgrid: %d x %d, halo backend: %s \n"
                 "CUDECOMP:\t(skipped) \n",
                 grid_desc->config.pdims[0], grid_desc->config.pdims[1],
                 cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend));
        } else {
          printf("CUDECOMP:\tgrid: %d x %d, halo backend: %s \n"
                 "CUDECOMP:\tTotal time min/max/avg/std [ms]: %f/%f/%f/%f\n",
                 grid_desc->config.pdims[0], grid_desc->config.pdims[1],
                 cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend), times[0], times[1], times[2],
                 times[3]);
        }
      }

      // Print performance report for this configuration if enabled
      if (handle->performance_report_enable && !skip_case) { printPerformanceReport(handle, grid_desc); }

      if (skip_case) continue;

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
    if (work != work_nvshmem) {
      auto tmp = grid_desc->config.halo_comm_backend;
      grid_desc->config.halo_comm_backend = (need_nccl) ? CUDECOMP_HALO_COMM_NCCL : CUDECOMP_HALO_COMM_MPI;
      CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
      grid_desc->config.halo_comm_backend = tmp;
    }
#ifdef ENABLE_NVSHMEM
    // Temporarily set backend to force nvshmem_malloc path in cudecompMalloc/Free
    auto tmp = grid_desc->config.halo_comm_backend;
    grid_desc->config.halo_comm_backend = CUDECOMP_HALO_COMM_NVSHMEM;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_nvshmem));
    grid_desc->config.halo_comm_backend = tmp;
#endif
  } else {
    auto tmp = grid_desc->config.halo_comm_backend;
    grid_desc->config.halo_comm_backend = (need_nccl) ? CUDECOMP_HALO_COMM_NCCL : CUDECOMP_HALO_COMM_MPI;
    CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work));
    grid_desc->config.halo_comm_backend = tmp;
  }

  CHECK_CUDA(cudaFree(data));

  // Set handle to best option (broadcast from rank 0 for consistency)
  CHECK_MPI(MPI_Bcast(&comm_backend_best, sizeof(cudecompHaloCommBackend_t), MPI_CHAR, 0, handle->mpi_comm));
  CHECK_MPI(MPI_Bcast(pdims_best, 2 * sizeof(int), MPI_INT, 0, handle->mpi_comm));

  grid_desc->config.halo_comm_backend = comm_backend_best;
  grid_desc->config.pdims[0] = pdims_best[0];
  grid_desc->config.pdims[1] = pdims_best[1];

  if (handle->rank == 0) {
    printf("CUDECOMP: SELECTED: grid: %d x %d, halo backend: %s, Avg. time [ms]: %f\n", grid_desc->config.pdims[0],
           grid_desc->config.pdims[1], cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend), t_best);
  }

  CHECK_MPI(MPI_Barrier(handle->mpi_comm));
  double t_end = MPI_Wtime();
  if (handle->rank == 0) printf("CUDECOMP: halo autotuning time [s]: %f\n", t_end - t_start);

  // Reset performance samples after autotuning
  resetPerformanceSamples(handle, grid_desc);
}

} // namespace cudecomp
