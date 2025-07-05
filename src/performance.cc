/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <mpi.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/performance.h"

namespace cudecomp {

// Helper function to create transpose configuration key (no longer template)
cudecompTransposeConfigKey createTransposeConfig(int ax, int dir, void* input, void* output,
                                                const int32_t input_halo_extents_ptr[],
                                                const int32_t output_halo_extents_ptr[],
                                                const int32_t input_padding_ptr[],
                                                const int32_t output_padding_ptr[],
                                                cudecompDataType_t datatype) {
  std::array<int32_t, 3> input_halo_extents{0, 0, 0};
  std::array<int32_t, 3> output_halo_extents{0, 0, 0};
  std::array<int32_t, 3> input_padding{0, 0, 0};
  std::array<int32_t, 3> output_padding{0, 0, 0};

  if (input_halo_extents_ptr) {
    std::copy(input_halo_extents_ptr, input_halo_extents_ptr + 3, input_halo_extents.begin());
  }
  if (output_halo_extents_ptr) {
    std::copy(output_halo_extents_ptr, output_halo_extents_ptr + 3, output_halo_extents.begin());
  }
  if (input_padding_ptr) {
    std::copy(input_padding_ptr, input_padding_ptr + 3, input_padding.begin());
  }
  if (output_padding_ptr) {
    std::copy(output_padding_ptr, output_padding_ptr + 3, output_padding.begin());
  }

  bool inplace = (input == output);
  bool managed_memory = isManagedPointer(input) || isManagedPointer(output);

  return std::make_tuple(ax, dir, input_halo_extents, output_halo_extents,
                         input_padding, output_padding, inplace, managed_memory, datatype);
}

// Helper function to get or create performance sample collection for a configuration
cudecompPerformanceSampleCollection& getOrCreatePerformanceSamples(const cudecompHandle_t handle,
                                                                   cudecompGridDesc_t grid_desc,
                                                                   const cudecompTransposeConfigKey& config) {
  auto& samples_map = grid_desc->perf_samples_map;

  if (samples_map.find(config) == samples_map.end()) {
    // Create new sample collection for this configuration
    cudecompPerformanceSampleCollection collection;
    collection.samples.resize(handle->performance_report_samples);
    collection.sample_idx = 0;

    // Create events for each sample
    for (auto& sample : collection.samples) {
      CHECK_CUDA(cudaEventCreate(&sample.transpose_start_event));
      CHECK_CUDA(cudaEventCreate(&sample.transpose_end_event));
      sample.alltoall_start_events.resize(handle->nranks);
      sample.alltoall_end_events.resize(handle->nranks);
      for (auto& event : sample.alltoall_start_events) {
        CHECK_CUDA(cudaEventCreate(&event));
      }
      for (auto& event : sample.alltoall_end_events) {
        CHECK_CUDA(cudaEventCreate(&event));
      }
      sample.valid = false;
    }

    samples_map[config] = std::move(collection);
  }

  return samples_map[config];
}

// Helper function to format array as compact string
std::string formatArray(const std::array<int32_t, 3>& arr) {
  std::ostringstream oss;
  oss << "[" << arr[0] << "," << arr[1] << "," << arr[2] << "]";
  return oss.str();
}

// Helper function to get operation name from config
std::string getOperationName(const cudecompTransposeConfigKey& config) {
  int ax = std::get<0>(config);
  int dir = std::get<1>(config);

  if (ax == 0) {
    return "TransposeXY";
  } else if (ax == 1 && dir == 1) {
    return "TransposeYZ";
  } else if (ax == 2) {
    return "TransposeZY";
  } else if (ax == 1 && dir == -1) {
    return "TransposeYX";
  }
  return "Unknown";
}

// Helper function to convert datatype to string
std::string getDatatypeString(cudecompDataType_t datatype) {
  switch (datatype) {
    case CUDECOMP_FLOAT: return "S";
    case CUDECOMP_DOUBLE: return "D";
    case CUDECOMP_FLOAT_COMPLEX: return "C";
    case CUDECOMP_DOUBLE_COMPLEX: return "Z";
    default: return "unknown";
  }
}

// Helper structure for statistics
struct PerformanceStats {
  std::string operation;
  std::string datatype;
  std::string halos;        // Combined input/output halos
  std::string padding;      // Combined input/output padding
  std::string inplace;
  std::string managed;
  int samples;
  float total_time_avg;
  float alltoall_time_avg;
  float local_time_avg;
  float alltoall_bw_avg;
};

// Helper structure to hold pre-computed timing data
struct ConfigTimingData {
  PerformanceStats stats;
  std::vector<float> total_times;
  std::vector<float> alltoall_times;
  std::vector<float> local_times;
  std::vector<float> alltoall_bws;
  std::vector<int> sample_indices;
};

void printFinalPerformanceReport(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc) {
  // Synchronize to ensure all events are recorded
  CHECK_CUDA(cudaDeviceSynchronize());

  // Collect all statistics and timing data
  std::vector<ConfigTimingData> all_config_data;

  for (const auto& entry : grid_desc->perf_samples_map) {
    const auto& config = entry.first;
    const auto& collection = entry.second;

    ConfigTimingData config_data;

    // Collect valid samples and compute elapsed times once
    for (int i = 0; i < collection.samples.size(); ++i) {
      const auto& sample = collection.samples[i];
      if (!sample.valid) continue;

      float alltoall_timing_ms = 0.0f;
      for (int j = 0; j < sample.alltoall_timing_count; ++j) {
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, sample.alltoall_start_events[j], sample.alltoall_end_events[j]));
        alltoall_timing_ms += elapsed_time;
      }

      float transpose_timing_ms;
      CHECK_CUDA(cudaEventElapsedTime(&transpose_timing_ms, sample.transpose_start_event, sample.transpose_end_event));

      config_data.total_times.push_back(transpose_timing_ms);
      config_data.alltoall_times.push_back(alltoall_timing_ms);
      config_data.local_times.push_back(transpose_timing_ms - alltoall_timing_ms);

      float alltoall_bw = (alltoall_timing_ms > 0) ? sample.alltoall_bytes * 1e-6 / alltoall_timing_ms : 0;
      config_data.alltoall_bws.push_back(alltoall_bw);
      config_data.sample_indices.push_back(i);
    }

    if (config_data.total_times.empty()) continue;

    // Prepare aggregated statistics
    PerformanceStats& stats = config_data.stats;
    stats.operation = getOperationName(config);
    stats.datatype = getDatatypeString(std::get<8>(config));

    // Format combined halos and padding
    auto input_halos = std::get<2>(config);
    auto output_halos = std::get<3>(config);
    auto input_padding = std::get<4>(config);
    auto output_padding = std::get<5>(config);

    stats.halos = formatArray(input_halos) + "/" + formatArray(output_halos);
    stats.padding = formatArray(input_padding) + "/" + formatArray(output_padding);
    stats.inplace = std::get<6>(config) ? "Y" : "N";
    stats.managed = std::get<7>(config) ? "Y" : "N";
    stats.samples = config_data.total_times.size();

    // Compute average statistics across all ranks
    stats.total_time_avg = std::accumulate(config_data.total_times.begin(), config_data.total_times.end(), 0.0f) / config_data.total_times.size();
    stats.alltoall_time_avg = std::accumulate(config_data.alltoall_times.begin(), config_data.alltoall_times.end(), 0.0f) / config_data.alltoall_times.size();
    stats.local_time_avg = std::accumulate(config_data.local_times.begin(), config_data.local_times.end(), 0.0f) / config_data.local_times.size();
    stats.alltoall_bw_avg = std::accumulate(config_data.alltoall_bws.begin(), config_data.alltoall_bws.end(), 0.0f) / config_data.alltoall_bws.size();

    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &stats.total_time_avg, 1, MPI_FLOAT, MPI_SUM, handle->mpi_comm));
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &stats.alltoall_time_avg, 1, MPI_FLOAT, MPI_SUM, handle->mpi_comm));
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &stats.local_time_avg, 1, MPI_FLOAT, MPI_SUM, handle->mpi_comm));
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &stats.alltoall_bw_avg, 1, MPI_FLOAT, MPI_SUM, handle->mpi_comm));

    stats.total_time_avg /= handle->nranks;
    stats.alltoall_time_avg /= handle->nranks;
    stats.local_time_avg /= handle->nranks;
    stats.alltoall_bw_avg /= handle->nranks;

    all_config_data.push_back(std::move(config_data));
  }

  // Print summary information on rank 0 only
  if (handle->rank == 0) {
    printf("CUDECOMP: ===== Performance Summary =====\n");

    // Print grid descriptor configuration information
    printf("CUDECOMP: Grid Configuration:\n");
    printf("CUDECOMP:\tTranspose backend: %s\n",
           cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend));
    printf("CUDECOMP:\tProcess grid: [%d, %d]\n",
           grid_desc->config.pdims[0], grid_desc->config.pdims[1]);
    printf("CUDECOMP:\tGlobal dimensions: [%d, %d, %d]\n",
           grid_desc->config.gdims[0], grid_desc->config.gdims[1], grid_desc->config.gdims[2]);

    // Print memory ordering information
    printf("CUDECOMP:\tMemory order: ");
    for (int axis = 0; axis < 3; ++axis) {
      printf("[%d,%d,%d]", grid_desc->config.transpose_mem_order[axis][0],
                          grid_desc->config.transpose_mem_order[axis][1],
                          grid_desc->config.transpose_mem_order[axis][2]);
      if (axis < 2) printf("; ");
    }
    printf("\n");

    printf("CUDECOMP:\n");
    printf("CUDECOMP: Transpose Performance Data:\n");
    printf("CUDECOMP:\n");

    if (all_config_data.empty()) {
      printf("CUDECOMP: No performance data collected\n");
      printf("CUDECOMP: ================================\n");
      return;
    }

    // Print compact table header
    printf("CUDECOMP: %-12s %-6s %-16s %-16s %-8s %-8s %-8s %-9s %-9s %-9s %-9s\n",
           "operation", "dtype", "halo extents", "padding", "inplace", "managed", "samples",
           "total", "A2A", "local", "A2A BW");
    printf("CUDECOMP: %-12s %-6s %-16s %-16s %-8s %-8s %-8s %-9s %-9s %-9s %-9s\n",
           "", "", "", "", "", "", "",
           "[ms]", "[ms]", "[ms]", "[GB/s]");
    printf("CUDECOMP: ");
    for (int i = 0; i < 120; ++i) printf("-");
    printf("\n");

    // Print table rows
    for (const auto& config_data : all_config_data) {
      const auto& stats = config_data.stats;
      printf("CUDECOMP: %-12s %-6s %-16s %-16s %-8s %-8s %-8d %-9.3f %-9.3f %-9.3f %-9.3f\n",
             stats.operation.c_str(),
             stats.datatype.c_str(),
             stats.halos.c_str(),
             stats.padding.c_str(),
             stats.inplace.c_str(),
             stats.managed.c_str(),
             stats.samples,
             stats.total_time_avg,
             stats.alltoall_time_avg,
             stats.local_time_avg,
             stats.alltoall_bw_avg
             );
    }
  }

  // Print per-sample data if detail level > 0
  if (handle->performance_report_detail > 0) {
    if (handle->rank == 0) {
      printf("CUDECOMP:\n");
      printf("CUDECOMP: Per-Sample Details:\n");
      printf("CUDECOMP:\n");
    }

    for (const auto& config_data : all_config_data) {
      const auto& stats = config_data.stats;

      // Print configuration header on rank 0
      if (handle->rank == 0) {
        printf("CUDECOMP: %s (dtype=%s, halos=%s, padding=%s, inplace=%s, managed=%s) samples:\n",
               stats.operation.c_str(),
               stats.datatype.c_str(),
               stats.halos.c_str(),
               stats.padding.c_str(),
               stats.inplace.c_str(),
               stats.managed.c_str());
      }

      const auto& total_times = config_data.total_times;
      const auto& alltoall_times = config_data.alltoall_times;
      const auto& local_times = config_data.local_times;
      const auto& alltoall_bws = config_data.alltoall_bws;
      const auto& sample_indices = config_data.sample_indices;

      if (total_times.empty()) continue;

      if (handle->performance_report_detail == 1) {
        // Print per-sample data for rank 0 only
        if (handle->rank == 0) {
          printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
                 "rank", "sample", "total", "A2A", "local", "A2A BW");
          printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
                 "", "", "[ms]", "[ms]", "[ms]", "[GB/s]");

          for (int i = 0; i < total_times.size(); ++i) {
            printf("CUDECOMP: %-6d %-12d %-9.3f %-9.3f %-9.3f %-9.3f\n",
                   handle->rank, sample_indices[i], total_times[i], alltoall_times[i],
                   local_times[i], alltoall_bws[i]);
          }
        }
      } else if (handle->performance_report_detail == 2) {
        // Gather data from all ranks to rank 0
        // Note: We assume all entries have the same number of samples per rank
        int num_samples = total_times.size();

        if (handle->rank == 0) {
          // Use MPI_Gather instead of MPI_Gatherv since all ranks have the same number of samples
          std::vector<float> all_total_times(num_samples * handle->nranks);
          std::vector<float> all_alltoall_times(num_samples * handle->nranks);
          std::vector<float> all_local_times(num_samples * handle->nranks);
          std::vector<float> all_alltoall_bws(num_samples * handle->nranks);
          std::vector<int> all_sample_indices(num_samples * handle->nranks);

          CHECK_MPI(MPI_Gather(total_times.data(), num_samples, MPI_FLOAT,
                               all_total_times.data(), num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(alltoall_times.data(), num_samples, MPI_FLOAT,
                               all_alltoall_times.data(), num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(local_times.data(), num_samples, MPI_FLOAT,
                               all_local_times.data(), num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(alltoall_bws.data(), num_samples, MPI_FLOAT,
                               all_alltoall_bws.data(), num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(sample_indices.data(), num_samples, MPI_INT,
                               all_sample_indices.data(), num_samples, MPI_INT, 0, handle->mpi_comm));

          // Print header
          printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
                 "rank", "sample", "total", "A2A", "local", "A2A BW");
          printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
                 "", "", "[ms]", "[ms]", "[ms]", "[GB/s]");

          // Print data sorted by rank
          for (int r = 0; r < handle->nranks; ++r) {
            for (int s = 0; s < num_samples; ++s) {
              int idx = r * num_samples + s;
              printf("CUDECOMP: %-6d %-12d %-9.3f %-9.3f %-9.3f %-9.3f\n",
                     r, all_sample_indices[idx], all_total_times[idx],
                     all_alltoall_times[idx], all_local_times[idx],
                     all_alltoall_bws[idx]);
            }
          }
        } else {
          // Non-rank-0 processes just send their data
          CHECK_MPI(MPI_Gather(total_times.data(), num_samples, MPI_FLOAT,
                               nullptr, num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(alltoall_times.data(), num_samples, MPI_FLOAT,
                               nullptr, num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(local_times.data(), num_samples, MPI_FLOAT,
                               nullptr, num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(alltoall_bws.data(), num_samples, MPI_FLOAT,
                               nullptr, num_samples, MPI_FLOAT, 0, handle->mpi_comm));
          CHECK_MPI(MPI_Gather(sample_indices.data(), num_samples, MPI_INT,
                               nullptr, num_samples, MPI_INT, 0, handle->mpi_comm));
        }
      }

      if (handle->rank == 0) {
        printf("CUDECOMP:\n");
      }
    }
  }

  if (handle->rank == 0) {
    printf("CUDECOMP: ================================\n");
  }
}

void resetPerformanceSamples(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc) {
  if (!handle->performance_report_enable) return;

  // Reset all sample collections in the map
  for (auto& entry : grid_desc->perf_samples_map) {
    auto& collection = entry.second;
    collection.sample_idx = 0;
    collection.warmup_count = 0;

    // Mark all samples as invalid and reset counters
    for (auto& sample : collection.samples) {
      sample.valid = false;
      sample.alltoall_timing_count = 0;
      sample.alltoall_bytes = 0;
    }
  }
}

// Helper function to advance sample index with warmup handling
void advancePerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                             const cudecompTransposeConfigKey& config) {
  if (!handle->performance_report_enable) return;

  auto& collection = getOrCreatePerformanceSamples(handle, grid_desc, config);

  // Check if we're still in warmup phase
  if (collection.warmup_count < handle->performance_report_warmup_samples) {
    collection.warmup_count++;
    // During warmup, don't advance the circular buffer, just mark current sample as invalid
    collection.samples[collection.sample_idx].valid = false;
  } else {
    // Past warmup, advance the circular buffer normally
    collection.sample_idx = (collection.sample_idx + 1) % handle->performance_report_samples;
  }
}

} // namespace cudecomp
