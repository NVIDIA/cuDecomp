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
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <filesystem>

#include <mpi.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/performance.h"

namespace cudecomp {

// Helper function to create file name with grid descriptor information
std::filesystem::path createPerformanceReportFileName(const std::string& write_dir,
                                                      const std::string& table_type,
                                                      const cudecompGridDesc_t grid_desc) {
  std::ostringstream filename;
  filename << "cudecomp-perf-report-" << table_type << "-";
  filename << "tcomm_" << grid_desc->config.transpose_comm_backend << "-";
  filename << "hcomm_" << grid_desc->config.halo_comm_backend << "-";
  filename << "pdims_" << grid_desc->config.pdims[0] << "x" << grid_desc->config.pdims[1] << "-";
  filename << "gdims_" << grid_desc->config.gdims[0] << "x" << grid_desc->config.gdims[1] << "x" << grid_desc->config.gdims[2] << "-";
  filename << "memorder_";
  for (int axis = 0; axis < 3; ++axis) {
    filename << grid_desc->config.transpose_mem_order[axis][0] << grid_desc->config.transpose_mem_order[axis][1] << grid_desc->config.transpose_mem_order[axis][2];
  }
  filename << ".csv";
  
  return std::filesystem::path(write_dir) / filename.str();
}

// Helper function to write CSV header comment with grid configuration information
void writeCSVHeader(std::ofstream& file, const cudecompGridDesc_t grid_desc) {
  file << "# Transpose backend: " << cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend) << "\n";
  file << "# Halo backend: " << cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend) << "\n";
  file << "# Process grid: [" << grid_desc->config.pdims[0] << ", " << grid_desc->config.pdims[1] << "]\n";
  file << "# Global dimensions: [" << grid_desc->config.gdims[0] << ", " << grid_desc->config.gdims[1] << ", " << grid_desc->config.gdims[2] << "]\n";
  file << "# Memory order: ";
  for (int axis = 0; axis < 3; ++axis) {
    file << "[" << grid_desc->config.transpose_mem_order[axis][0] << ","
         << grid_desc->config.transpose_mem_order[axis][1] << ","
         << grid_desc->config.transpose_mem_order[axis][2] << "]";
    if (axis < 2) file << "; ";
  }
  file << "\n#\n";
}

// Write transpose performance table to CSV
void writeTransposePerformanceTableCSV(const std::vector<TransposeConfigTimingData>& all_transpose_config_data,
                                       const cudecompGridDesc_t grid_desc,
                                       const std::string& write_dir) {
  if (all_transpose_config_data.empty()) return;

  std::filesystem::path filename = createPerformanceReportFileName(write_dir, "transpose-aggregated", grid_desc);
  std::ofstream file(filename);
  if (!file.is_open()) {
    printf("CUDECOMP:WARN: Could not open file %s for writing\n", filename.string().c_str());
    return;
  }

  writeCSVHeader(file, grid_desc);

  // Write table header
  file << "operation,dtype,input_halo_extents,output_halo_extents,input_padding,output_padding,inplace,managed,samples,total_ms,A2A_ms,local_ms,A2A_BW_GBps\n";

  // Write table data rows
  for (const auto& config_data : all_transpose_config_data) {
    const auto& stats = config_data.stats;
    if (stats.samples > 0) {
      file << stats.operation << ","
           << stats.datatype << ","
           << "\"" << stats.input_halos << "\","
           << "\"" << stats.output_halos << "\","
           << "\"" << stats.input_padding << "\","
           << "\"" << stats.output_padding << "\","
           << stats.inplace << ","
           << stats.managed << ","
           << stats.samples << ","
           << std::fixed << std::setprecision(3) << stats.total_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.alltoall_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.local_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.alltoall_bw_avg << "\n";
    }
  }

  file.close();
  printf("CUDECOMP:\n");
  printf("CUDECOMP: Wrote transpose performance data to %s\n", filename.string().c_str());
}

// Write halo performance table to CSV
void writeHaloPerformanceTableCSV(const std::vector<HaloConfigTimingData>& all_halo_config_data,
                                  const cudecompGridDesc_t grid_desc,
                                  const std::string& write_dir) {
  if (all_halo_config_data.empty()) return;

  std::filesystem::path filename = createPerformanceReportFileName(write_dir, "halo-aggregated", grid_desc);
  std::ofstream file(filename);
  if (!file.is_open()) {
    printf("CUDECOMP:WARN: Could not open file %s for writing\n", filename.string().c_str());
    return;
  }

  writeCSVHeader(file, grid_desc);

  // Write table header
  file << "operation,dtype,dim,halo_extent,periods,padding,managed,samples,total_ms,SR_ms,local_ms,SR_BW_GBps\n";

  // Write table data rows
  for (const auto& config_data : all_halo_config_data) {
    const auto& stats = config_data.stats;
    if (stats.samples > 0) {
      file << stats.operation << ","
           << stats.datatype << ","
           << stats.dim << ","
           << "\"" << stats.halos << "\","
           << "\"" << stats.periods << "\","
           << "\"" << stats.padding << "\","
           << stats.managed << ","
           << stats.samples << ","
           << std::fixed << std::setprecision(3) << stats.total_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.sendrecv_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.local_time_avg << ","
           << std::fixed << std::setprecision(3) << stats.sendrecv_bw_avg << "\n";
    }
  }

  file.close();
  printf("CUDECOMP:\n");
  printf("CUDECOMP: Wrote halo performance data to %s\n", filename.string().c_str());
}

// Helper function to create transpose configuration key
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

// Helper function to create halo configuration key
cudecompHaloConfigKey createHaloConfig(int ax, int dim, void* input,
                                       const int32_t halo_extents_ptr[],
                                       const bool halo_periods_ptr[],
                                       const int32_t padding_ptr[],
                                       cudecompDataType_t datatype) {
  std::array<int32_t, 3> halo_extents{0, 0, 0};
  std::array<bool, 3> halo_periods{false, false, false};
  std::array<int32_t, 3> padding{0, 0, 0};

  if (halo_extents_ptr) {
    std::copy(halo_extents_ptr, halo_extents_ptr + 3, halo_extents.begin());
  }
  if (halo_periods_ptr) {
    std::copy(halo_periods_ptr, halo_periods_ptr + 3, halo_periods.begin());
  }
  if (padding_ptr) {
    std::copy(padding_ptr, padding_ptr + 3, padding.begin());
  }

  bool managed_memory = isManagedPointer(input);

  return std::make_tuple(ax, dim, halo_extents, halo_periods, padding, managed_memory, datatype);
}

// Helper function to get or create transpose performance sample collection for a configuration
cudecompTransposePerformanceSampleCollection& getOrCreateTransposePerformanceSamples(const cudecompHandle_t handle,
                                                                                     cudecompGridDesc_t grid_desc,
                                                                                     const cudecompTransposeConfigKey& config) {
  auto& samples_map = grid_desc->transpose_perf_samples_map;

  if (samples_map.find(config) == samples_map.end()) {
    // Create new sample collection for this configuration
    cudecompTransposePerformanceSampleCollection collection;
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

// Helper function to get or create halo performance sample collection for a configuration
cudecompHaloPerformanceSampleCollection& getOrCreateHaloPerformanceSamples(const cudecompHandle_t handle,
                                                                           cudecompGridDesc_t grid_desc,
                                                                           const cudecompHaloConfigKey& config) {
  auto& samples_map = grid_desc->halo_perf_samples_map;

  if (samples_map.find(config) == samples_map.end()) {
    // Create new sample collection for this configuration
    cudecompHaloPerformanceSampleCollection collection;
    collection.samples.resize(handle->performance_report_samples);
    collection.sample_idx = 0;

    // Create events for each sample
    for (auto& sample : collection.samples) {
      CHECK_CUDA(cudaEventCreate(&sample.halo_start_event));
      CHECK_CUDA(cudaEventCreate(&sample.halo_end_event));
      CHECK_CUDA(cudaEventCreate(&sample.sendrecv_start_event));
      CHECK_CUDA(cudaEventCreate(&sample.sendrecv_end_event));
      sample.valid = false;
    }

    samples_map[config] = std::move(collection);
  }

  return samples_map[config];
}

// Helper function to format integer array as compact string
std::string formatArray(const std::array<int32_t, 3>& arr) {
  std::ostringstream oss;
  oss << "[" << arr[0] << "," << arr[1] << "," << arr[2] << "]";
  return oss.str();
}

// Helper function to format boolean array as compact string
std::string formatArray(const std::array<bool, 3>& arr) {
  std::ostringstream oss;
  oss << "[" << (arr[0] ? "1" : "0") << "," << (arr[1] ? "1" : "0") << "," << (arr[2] ? "1" : "0") << "]";
  return oss.str();
}

// Helper function to get operation name from transpose config
std::string getTransposeOperationName(const cudecompTransposeConfigKey& config) {
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

// Helper function to get operation name from halo config
std::string getHaloOperationName(const cudecompHaloConfigKey& config) {
  int ax = std::get<0>(config);

  if (ax == 0) {
    return "HaloX";
  } else if (ax == 1) {
    return "HaloY";
  } else if (ax == 2) {
    return "HaloZ";
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

// Comparison function to order transpose configurations.
// Ordering is (operation, datatype, halos, padding, inplace, managed).
bool compareTransposeConfigData(const TransposeConfigTimingData& a,
                                const TransposeConfigTimingData& b) {
  static const std::map<std::string, int> op_priority = {
    {"TransposeXY", 0},
    {"TransposeYZ", 1},
    {"TransposeZY", 2},
    {"TransposeYX", 3}
  };

  static const std::map<std::string, int> dtype_priority = {
    {"S", 0}, {"D", 1}, {"C", 2}, {"Z", 3}
  };

  if (a.stats.operation != b.stats.operation) {
    return op_priority.at(a.stats.operation) < op_priority.at(b.stats.operation);
  }
  if (a.stats.datatype != b.stats.datatype) {
    return dtype_priority.at(a.stats.datatype) < dtype_priority.at(b.stats.datatype);
  }
  if (a.stats.input_halos != b.stats.input_halos) {
    return a.stats.input_halos < b.stats.input_halos;
  }
  if (a.stats.output_halos != b.stats.output_halos) {
    return a.stats.output_halos < b.stats.output_halos;
  }
  if (a.stats.input_padding != b.stats.input_padding) {
    return a.stats.input_padding < b.stats.input_padding;
  }
  if (a.stats.output_padding != b.stats.output_padding) {
    return a.stats.output_padding < b.stats.output_padding;
  }
  if (a.stats.inplace != b.stats.inplace) {
    return a.stats.inplace < b.stats.inplace;
  }
  return a.stats.managed < b.stats.managed;
}

// Comparison function to order halo configurations.
// Ordering is (operation, datatype, dim, halos, periods, padding, managed).
bool compareHaloConfigData(const HaloConfigTimingData& a,
                           const HaloConfigTimingData& b) {
  static const std::map<std::string, int> op_priority = {
    {"HaloX", 0},
    {"HaloY", 1},
    {"HaloZ", 2}
  };

  static const std::map<std::string, int> dtype_priority = {
    {"S", 0}, {"D", 1}, {"C", 2}, {"Z", 3}
  };

  if (a.stats.operation != b.stats.operation) {
    return op_priority.at(a.stats.operation) < op_priority.at(b.stats.operation);
  }
  if (a.stats.dim != b.stats.dim) {
    return a.stats.dim < b.stats.dim;
  }
  if (a.stats.datatype != b.stats.datatype) {
    return dtype_priority.at(a.stats.datatype) < dtype_priority.at(b.stats.datatype);
  }
  if (a.stats.halos != b.stats.halos) {
    return a.stats.halos < b.stats.halos;
  }
  if (a.stats.periods != b.stats.periods) {
    return a.stats.periods < b.stats.periods;
  }
  if (a.stats.padding != b.stats.padding) {
    return a.stats.padding < b.stats.padding;
  }
  return a.stats.managed < b.stats.managed;
}

// Function to compute average across ranks
float computeGlobalAverage(const std::vector<float>& values, const cudecompHandle_t handle) {
  if (values.size() == 0) { return 0.0f; }

  float value = std::accumulate(values.begin(), values.end(), 0.0f);
  value /= values.size();
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_FLOAT, MPI_SUM, handle->mpi_comm));
  value /= handle->nranks;
  return value;
}

// Function to gather data from all ranks
void gatherSampleData(const std::vector<float>& local_data, std::vector<float>& all_data,
                      const cudecompHandle_t handle) {
  int num_samples = local_data.size();
  if (handle->rank == 0) {
    all_data.resize(num_samples * handle->nranks);
  }

  CHECK_MPI(MPI_Gather(local_data.data(), num_samples, MPI_FLOAT,
                       all_data.data(), num_samples, MPI_FLOAT, 0, handle->mpi_comm));
}

// Process transpose timing data from sample collections
TransposeConfigTimingData processTransposeConfig(const cudecompTransposeConfigKey& config,
                                                 const cudecompTransposePerformanceSampleCollection& collection,
                                                 const cudecompHandle_t handle) {
  TransposeConfigTimingData config_data;

  config_data.total_times.reserve(collection.samples.size());
  config_data.alltoall_times.reserve(collection.samples.size());
  config_data.local_times.reserve(collection.samples.size());
  config_data.alltoall_bws.reserve(collection.samples.size());

  // Collect valid samples and compute elapsed times
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
  }

  // Prepare aggregated statistics
  TransposePerformanceStats& stats = config_data.stats;
  stats.operation = getTransposeOperationName(config);
  stats.datatype = getDatatypeString(std::get<8>(config));

  auto input_halos = std::get<2>(config);
  auto output_halos = std::get<3>(config);
  auto input_padding = std::get<4>(config);
  auto output_padding = std::get<5>(config);

  stats.input_halos = formatArray(input_halos);
  stats.output_halos = formatArray(output_halos);
  stats.input_padding = formatArray(input_padding);
  stats.output_padding = formatArray(output_padding);
  stats.inplace = std::get<6>(config) ? "T" : "F";
  stats.managed = std::get<7>(config) ? "T" : "F";
  stats.samples = config_data.total_times.size();

  // Compute average statistics and reduce across all ranks
  stats.total_time_avg = computeGlobalAverage(config_data.total_times, handle);
  stats.alltoall_time_avg = computeGlobalAverage(config_data.alltoall_times, handle);
  stats.local_time_avg = computeGlobalAverage(config_data.local_times, handle);
  stats.alltoall_bw_avg = computeGlobalAverage(config_data.alltoall_bws, handle);

  return config_data;
}

// Process halo timing data from sample collections
HaloConfigTimingData processHaloConfig(const cudecompHaloConfigKey& config,
                                       const cudecompHaloPerformanceSampleCollection& collection,
                                       const cudecompHandle_t handle) {
  HaloConfigTimingData config_data;

  config_data.total_times.reserve(collection.samples.size());
  config_data.sendrecv_times.reserve(collection.samples.size());
  config_data.local_times.reserve(collection.samples.size());
  config_data.sendrecv_bws.reserve(collection.samples.size());

  // Collect valid samples and compute elapsed times
  for (int i = 0; i < collection.samples.size(); ++i) {
    const auto& sample = collection.samples[i];
    if (!sample.valid) continue;

    float sendrecv_timing_ms = 0.0f;
    if (sample.sendrecv_bytes > 0) {
      CHECK_CUDA(cudaEventElapsedTime(&sendrecv_timing_ms, sample.sendrecv_start_event, sample.sendrecv_end_event));
    }

    float halo_timing_ms;
    CHECK_CUDA(cudaEventElapsedTime(&halo_timing_ms, sample.halo_start_event, sample.halo_end_event));

    config_data.total_times.push_back(halo_timing_ms);
    config_data.sendrecv_times.push_back(sendrecv_timing_ms);
    config_data.local_times.push_back(halo_timing_ms - sendrecv_timing_ms);

    float sendrecv_bw = (sendrecv_timing_ms > 0) ? sample.sendrecv_bytes * 1e-6 / sendrecv_timing_ms : 0;
    config_data.sendrecv_bws.push_back(sendrecv_bw);
  }

  // Prepare aggregated statistics
  HaloPerformanceStats& stats = config_data.stats;
  stats.operation = getHaloOperationName(config);
  stats.datatype = getDatatypeString(std::get<6>(config));
  stats.dim = std::get<1>(config);

  // Format halo extents, periods, and padding
  auto halo_extents = std::get<2>(config);
  auto halo_periods = std::get<3>(config);
  auto padding = std::get<4>(config);

  stats.halos = formatArray(halo_extents);
  stats.periods = formatArray(halo_periods);
  stats.padding = formatArray(padding);
  stats.managed = std::get<5>(config) ? "T" : "F";
  stats.samples = config_data.total_times.size();

  // Compute average statistics across all ranks
  stats.total_time_avg = computeGlobalAverage(config_data.total_times, handle);
  stats.sendrecv_time_avg = computeGlobalAverage(config_data.sendrecv_times, handle);
  stats.local_time_avg = computeGlobalAverage(config_data.local_times, handle);
  stats.sendrecv_bw_avg = computeGlobalAverage(config_data.sendrecv_bws, handle);

  return config_data;
}

// Print grid configuration information
void printGridConfiguration(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc) {
  if (handle->rank != 0) return;

  printf("CUDECOMP:\n");
  printf("CUDECOMP: ===== Performance Summary =====\n");
  printf("CUDECOMP: Grid Configuration:\n");
  printf("CUDECOMP:\tTranspose backend: %s\n",
         cudecompTransposeCommBackendToString(grid_desc->config.transpose_comm_backend));
  printf("CUDECOMP:\tHalo backend: %s\n",
         cudecompHaloCommBackendToString(grid_desc->config.halo_comm_backend));
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
}

// Print transpose performance table
void printTransposePerformanceTable(const std::vector<TransposeConfigTimingData>& all_transpose_config_data) {
  printf("CUDECOMP: Transpose Performance Data:\n");
  printf("CUDECOMP:\n");

  // Print compact table header
  printf("CUDECOMP: %-12s %-6s %-15s %-15s %-8s %-8s %-8s %-9s %-9s %-9s %-9s\n",
    "operation", "dtype", "halo extents", "padding", "inplace", "managed", "samples",
         "total", "A2A", "local", "A2A BW");
  printf("CUDECOMP: %-12s %-6s %-15s %-15s %-8s %-8s %-8s %-9s %-9s %-9s %-9s\n",
         "", "", "", "", "", "", "",
         "[ms]", "[ms]", "[ms]", "[GB/s]");
  printf("CUDECOMP: ");
  for (int i = 0; i < 120; ++i) printf("-");
  printf("\n");

  // Print table rows
  for (const auto& config_data : all_transpose_config_data) {
    const auto& stats = config_data.stats;
    if (stats.samples > 0) {
      printf("CUDECOMP: %-12s %-6s %-7s/%-7s %-7s/%-7s %-8s %-8s %-8d %-9.3f %-9.3f %-9.3f %-9.3f\n",
             stats.operation.c_str(),
             stats.datatype.c_str(),
             stats.input_halos.c_str(),
             stats.output_halos.c_str(),
             stats.input_padding.c_str(),
             stats.output_padding.c_str(),
             stats.inplace.c_str(),
             stats.managed.c_str(),
             stats.samples,
             stats.total_time_avg,
             stats.alltoall_time_avg,
             stats.local_time_avg,
             stats.alltoall_bw_avg);
    }
  }
}

// Print halo performance table
void printHaloPerformanceTable(const std::vector<HaloConfigTimingData>& all_halo_config_data) {
  printf("CUDECOMP:\n");
  printf("CUDECOMP: Halo Performance Data:\n");
  printf("CUDECOMP:\n");

  // Print compact table header
  printf("CUDECOMP: %-12s %-6s %-5s %-12s %-12s %-12s %-8s %-8s %-9s %-9s %-9s %-9s\n",
         "operation", "dtype", "dim", "halo extent", "periods", "padding", "managed", "samples",
         "total", "SR", "local", "SR BW");
  printf("CUDECOMP: %-12s %-6s %-5s %-12s %-12s %-12s %-8s %-8s %-9s %-9s %-9s %-9s\n",
         "", "", "", "", "", "", "", "",
         "[ms]", "[ms]", "[ms]", "[GB/s]");
  printf("CUDECOMP: ");
  for (int i = 0; i < 125; ++i) printf("-");
  printf("\n");

  // Print table rows
  for (const auto& config_data : all_halo_config_data) {
    const auto& stats = config_data.stats;
    if (stats.samples > 0) {
      printf("CUDECOMP: %-12s %-6s %-5d %-12s %-12s %-12s %-8s %-8d %-9.3f %-9.3f %-9.3f %-9.3f\n",
             stats.operation.c_str(),
             stats.datatype.c_str(),
             stats.dim,
             stats.halos.c_str(),
             stats.periods.c_str(),
             stats.padding.c_str(),
             stats.managed.c_str(),
             stats.samples,
             stats.total_time_avg,
             stats.sendrecv_time_avg,
             stats.local_time_avg,
             stats.sendrecv_bw_avg);
    }
  }
}

// Print per-sample transpose data for a single configuration
void printTransposePerSampleDetailsForConfig(const TransposeConfigTimingData& config_data,
                                            const cudecompHandle_t handle, int detail_level,
                                            std::ofstream* csv_file = nullptr) {
  const auto& stats = config_data.stats;
  const auto& total_times = config_data.total_times;
  const auto& alltoall_times = config_data.alltoall_times;
  const auto& local_times = config_data.local_times;
  const auto& alltoall_bws = config_data.alltoall_bws;

  if (total_times.empty()) return;

  int num_samples = total_times.size();
  std::vector<float> all_total_times, all_alltoall_times, all_local_times, all_alltoall_bws;

  if (detail_level == 1) {
    // Detail level 1: Only rank 0 prints its own data
    if (handle->rank != 0) return;

    all_total_times = total_times;
    all_alltoall_times = alltoall_times;
    all_local_times = local_times;
    all_alltoall_bws = alltoall_bws;

  } else if (detail_level == 2) {
    // Detail level 2: Gather data from all ranks
    gatherSampleData(total_times, all_total_times, handle);
    gatherSampleData(alltoall_times, all_alltoall_times, handle);
    gatherSampleData(local_times, all_local_times, handle);
    gatherSampleData(alltoall_bws, all_alltoall_bws, handle);

    if (handle->rank != 0) return;
  }

  printf("CUDECOMP: %s (dtype=%s, halo extents=%s/%s, padding=%s/%s, inplace=%s, managed=%s) samples:\n",
         stats.operation.c_str(),
         stats.datatype.c_str(),
         stats.input_halos.c_str(),
         stats.output_halos.c_str(),
         stats.input_padding.c_str(),
         stats.output_padding.c_str(),
         stats.inplace.c_str(),
         stats.managed.c_str());

  printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
         "rank", "sample", "total", "A2A", "local", "A2A BW");
  printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
         "", "", "[ms]", "[ms]", "[ms]", "[GB/s]");

  for (int r = 0; r < (detail_level == 1 ? 1 : handle->nranks); ++r) {
    for (int s = 0; s < num_samples; ++s) {
      int idx = r * num_samples + s;
      printf("CUDECOMP: %-6d %-12d %-9.3f %-9.3f %-9.3f %-9.3f\n",
             r, s, all_total_times[idx],
             all_alltoall_times[idx], all_local_times[idx],
             all_alltoall_bws[idx]);
      
      // Write to CSV if file is provided
      if (csv_file && csv_file->is_open()) {
        *csv_file << stats.operation << ","
                  << stats.datatype << ","
                  << "\"" << stats.input_halos << "\","
                  << "\"" << stats.output_halos << "\","
                  << "\"" << stats.input_padding << "\","
                  << "\"" << stats.output_padding << "\","
                  << stats.inplace << ","
                  << stats.managed << ","
                  << r << ","
                  << s << ","
                  << std::fixed << std::setprecision(3) << all_total_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_alltoall_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_local_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_alltoall_bws[idx] << "\n";
      }
    }
  }
  printf("CUDECOMP:\n");
}

// Print per-sample halo data for a single configuration
void printHaloPerSampleDetailsForConfig(const HaloConfigTimingData& config_data,
                                        const cudecompHandle_t handle, int detail_level,
                                        std::ofstream* csv_file = nullptr) {
  const auto& stats = config_data.stats;
  const auto& total_times = config_data.total_times;
  const auto& sendrecv_times = config_data.sendrecv_times;
  const auto& local_times = config_data.local_times;
  const auto& sendrecv_bws = config_data.sendrecv_bws;

  if (total_times.empty()) return;

  int num_samples = total_times.size();
  std::vector<float> all_total_times, all_sendrecv_times, all_local_times, all_sendrecv_bws;

  if (detail_level == 1) {
    // Detail level 1: Only rank 0 prints its own data
    if (handle->rank != 0) return;

    all_total_times = total_times;
    all_sendrecv_times = sendrecv_times;
    all_local_times = local_times;
    all_sendrecv_bws = sendrecv_bws;

  } else if (detail_level == 2) {
    // Detail level 2: Gather data from all ranks
    gatherSampleData(total_times, all_total_times, handle);
    gatherSampleData(sendrecv_times, all_sendrecv_times, handle);
    gatherSampleData(local_times, all_local_times, handle);
    gatherSampleData(sendrecv_bws, all_sendrecv_bws, handle);

    if (handle->rank != 0) return;
  }

  printf("CUDECOMP: %s (dtype=%s, dim=%d, halos=%s, periods=%s, padding=%s, managed=%s) samples:\n",
         stats.operation.c_str(),
         stats.datatype.c_str(),
         stats.dim,
         stats.halos.c_str(),
         stats.periods.c_str(),
         stats.padding.c_str(),
         stats.managed.c_str());

  printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
         "rank", "sample", "total", "SR", "local", "SR BW");
  printf("CUDECOMP: %-6s %-12s %-9s %-9s %-9s %-9s\n",
         "", "", "[ms]", "[ms]", "[ms]", "[GB/s]");

  for (int r = 0; r < (detail_level == 1 ? 1 : handle->nranks); ++r) {
    for (int s = 0; s < num_samples; ++s) {
      int idx = r * num_samples + s;
      printf("CUDECOMP: %-6d %-12d %-9.3f %-9.3f %-9.3f %-9.3f\n",
             r, s, all_total_times[idx],
             all_sendrecv_times[idx], all_local_times[idx],
             all_sendrecv_bws[idx]);
      
      // Write to CSV if file is provided
      if (csv_file && csv_file->is_open()) {
        *csv_file << stats.operation << ","
                  << stats.datatype << ","
                  << stats.dim << ","
                  << "\"" << stats.halos << "\","
                  << "\"" << stats.periods << "\","
                  << "\"" << stats.padding << "\","
                  << stats.managed << ","
                  << r << ","
                  << s << ","
                  << std::fixed << std::setprecision(3) << all_total_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_sendrecv_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_local_times[idx] << ","
                  << std::fixed << std::setprecision(3) << all_sendrecv_bws[idx] << "\n";
      }
    }
  }
  printf("CUDECOMP:\n");
}

// Print per-sample details for transpose configurations
void printTransposePerSampleDetails(const std::vector<TransposeConfigTimingData>& all_transpose_config_data,
                                   const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                                   const std::string& write_dir = "") {
  std::ofstream csv_file;
  bool csv_enabled = !write_dir.empty();
  
  // Check if there are any samples before creating CSV file
  bool has_samples = false;
  for (const auto& config_data : all_transpose_config_data) {
    if (config_data.stats.samples > 0) {
      has_samples = true;
      break;
    }
  }
  
  if (csv_enabled && handle->rank == 0 && has_samples) {
    std::filesystem::path filename = createPerformanceReportFileName(write_dir, "transpose-samples", grid_desc);
    csv_file.open(filename);
    if (csv_file.is_open()) {
      writeCSVHeader(csv_file, grid_desc);
      csv_file << "operation,dtype,input_halo_extents,output_halo_extents,input_padding,output_padding,inplace,managed,rank,sample,total_ms,A2A_ms,local_ms,A2A_BW_GBps\n";
    } else {
      printf("CUDECOMP: Warning: Could not open file %s for writing\n", filename.string().c_str());
    }
  }

  for (const auto& config_data : all_transpose_config_data) {
    if (config_data.stats.samples == 0) continue;

    printTransposePerSampleDetailsForConfig(config_data, handle, handle->performance_report_detail,
                                           csv_enabled && has_samples ? &csv_file : nullptr);
  }
  
  if (csv_file.is_open()) {
    csv_file.close();
    printf("CUDECOMP:\n");
    printf("CUDECOMP: Wrote per-sample transpose data to %s\n", createPerformanceReportFileName(write_dir, "transpose-samples", grid_desc).string().c_str());
  }
}

// Print per-sample details for halo configurations
void printHaloPerSampleDetails(const std::vector<HaloConfigTimingData>& all_halo_config_data,
                               const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                               const std::string& write_dir = "") {
  std::ofstream csv_file;
  bool csv_enabled = !write_dir.empty();
  
  // Check if there are any samples before creating CSV file
  bool has_samples = false;
  for (const auto& config_data : all_halo_config_data) {
    if (config_data.stats.samples > 0) {
      has_samples = true;
      break;
    }
  }
  
  if (csv_enabled && handle->rank == 0 && has_samples) {
    std::filesystem::path filename = createPerformanceReportFileName(write_dir, "halo-samples", grid_desc);
    csv_file.open(filename);
    if (csv_file.is_open()) {
      writeCSVHeader(csv_file, grid_desc);
      csv_file << "operation,dtype,dim,halo_extent,periods,padding,managed,rank,sample,total_ms,SR_ms,local_ms,SR_BW_GBps\n";
    } else {
      printf("CUDECOMP: Warning: Could not open file %s for writing\n", filename.string().c_str());
    }
  }

  for (const auto& config_data : all_halo_config_data) {
    if (config_data.stats.samples == 0) continue;

    printHaloPerSampleDetailsForConfig(config_data, handle, handle->performance_report_detail,
                                      csv_enabled && has_samples ? &csv_file : nullptr);
  }
  
  if (csv_file.is_open()) {
    csv_file.close();
    printf("CUDECOMP:\n");
    printf("CUDECOMP: Wrote per-sample halo data to %s\n", createPerformanceReportFileName(write_dir, "halo-samples", grid_desc).string().c_str());
  }
}

void printPerformanceReport(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc) {
  // Synchronize to ensure all events are recorded
  CHECK_CUDA(cudaDeviceSynchronize());

  // Collect all transpose statistics and timing data
  std::vector<TransposeConfigTimingData> all_transpose_config_data;
  all_transpose_config_data.reserve(grid_desc->transpose_perf_samples_map.size());

  for (const auto& entry : grid_desc->transpose_perf_samples_map) {
    const auto& config = entry.first;
    const auto& collection = entry.second;

    TransposeConfigTimingData config_data = processTransposeConfig(config, collection, handle);
    if (config_data.stats.samples > 0) {
      all_transpose_config_data.emplace_back(std::move(config_data));
    }
  }

  // Sort transpose configuration data for consistent ordering
  std::sort(all_transpose_config_data.begin(), all_transpose_config_data.end(), compareTransposeConfigData);

  // Collect all halo statistics and timing data with performance optimizations
  std::vector<HaloConfigTimingData> all_halo_config_data;
  all_halo_config_data.reserve(grid_desc->halo_perf_samples_map.size());

  for (const auto& entry : grid_desc->halo_perf_samples_map) {
    const auto& config = entry.first;
    const auto& collection = entry.second;

    HaloConfigTimingData config_data = processHaloConfig(config, collection, handle);
    if (config_data.stats.samples > 0) {
      all_halo_config_data.emplace_back(std::move(config_data));
    }
  }

  // Sort halo configuration data for consistent ordering
  std::sort(all_halo_config_data.begin(), all_halo_config_data.end(), compareHaloConfigData);

  // Check if CSV writing is enabled
  const std::string& write_dir = handle->performance_report_write_dir;
  bool csv_enabled = !write_dir.empty();

  // Print grid configuration information
  printGridConfiguration(handle, grid_desc);

  if (handle->rank == 0) {
    if (all_transpose_config_data.empty() && all_halo_config_data.empty()) {
      printf("CUDECOMP: No performance data collected\n");
      printf("CUDECOMP: ================================\n");
      printf("CUDECOMP:\n");
      return;
    }

    // Print transpose performance data
    if (!all_transpose_config_data.empty()) {
      printTransposePerformanceTable(all_transpose_config_data);
      
      // Write transpose performance data to CSV if enabled
      if (csv_enabled) {
        writeTransposePerformanceTableCSV(all_transpose_config_data, grid_desc, write_dir);
      }
    }

    // Print halo performance data
    if (!all_halo_config_data.empty()) {
      printHaloPerformanceTable(all_halo_config_data);
      
      // Write halo performance data to CSV if enabled
      if (csv_enabled) {
        writeHaloPerformanceTableCSV(all_halo_config_data, grid_desc, write_dir);
      }
    }
  }

  // Print per-sample data if detail level > 0
  if (handle->performance_report_detail > 0) {
    if (handle->rank == 0) {
      printf("CUDECOMP:\n");
      printf("CUDECOMP: Per-Sample Details:\n");
      printf("CUDECOMP:\n");
    }

    // Print transpose per-sample details (and write CSV if enabled)
    printTransposePerSampleDetails(all_transpose_config_data, handle, grid_desc, csv_enabled ? write_dir : "");

    // Print halo per-sample details (and write CSV if enabled)
    printHaloPerSampleDetails(all_halo_config_data, handle, grid_desc, csv_enabled ? write_dir : "");
  }

  if (handle->rank == 0) {
    printf("CUDECOMP: ================================\n");
    printf("CUDECOMP:\n");
  }
}

void resetPerformanceSamples(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc) {
  if (!handle->performance_report_enable) return;

  // Reset all transpose sample collections in the map
  for (auto& entry : grid_desc->transpose_perf_samples_map) {
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

  // Reset all halo sample collections in the map
  for (auto& entry : grid_desc->halo_perf_samples_map) {
    auto& collection = entry.second;
    collection.sample_idx = 0;
    collection.warmup_count = 0;

    // Mark all samples as invalid and reset counters
    for (auto& sample : collection.samples) {
      sample.valid = false;
      sample.sendrecv_bytes = 0;
    }
  }
}

// Helper function to advance transpose sample index with warmup handling
void advanceTransposePerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       const cudecompTransposeConfigKey& config) {
  if (!handle->performance_report_enable) return;

  auto& collection = getOrCreateTransposePerformanceSamples(handle, grid_desc, config);

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

// Helper function to advance halo sample index with warmup handling
void advanceHaloPerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                  const cudecompHaloConfigKey& config) {
  if (!handle->performance_report_enable) return;

  auto& collection = getOrCreateHaloPerformanceSamples(handle, grid_desc, config);

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
