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

#ifndef CUDECOMP_PERFORMANCE_H
#define CUDECOMP_PERFORMANCE_H

#include <array>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "internal/common.h"

namespace cudecomp {

using cudecompTransposeConfigKey = std::tuple<int32_t,                // ax (axis)
                                              int32_t,                // dir (direction)
                                              std::array<int32_t, 3>, // input_halo_extents
                                              std::array<int32_t, 3>, // output_halo_extents
                                              std::array<int32_t, 3>, // input_padding
                                              std::array<int32_t, 3>, // output_padding
                                              bool,                   // inplace
                                              bool,                   // managed_memory
                                              cudecompDataType_t      // datatype
                                              >;

using cudecompHaloConfigKey = std::tuple<int32_t,                // ax (axis)
                                         int32_t,                // dim (dimension)
                                         std::array<int32_t, 3>, // halo_extents
                                         std::array<bool, 3>,    // halo_periods
                                         std::array<int32_t, 3>, // padding
                                         bool,                   // managed_memory
                                         cudecompDataType_t      // datatype
                                         >;

// Helper structure for transpose statistics
struct TransposePerformanceStats {
  std::string operation;
  std::string datatype;
  std::string input_halos;
  std::string output_halos;
  std::string input_padding;
  std::string output_padding;
  std::string inplace;
  std::string managed;
  int samples;
  float total_time_avg;
  float alltoall_time_avg;
  float local_time_avg;
  float alltoall_bw_avg;
};

// Helper structure for halo statistics
struct HaloPerformanceStats {
  std::string operation;
  std::string datatype;
  int dim;
  std::string halos;
  std::string periods;
  std::string padding;
  std::string managed;
  int samples;
  float total_time_avg;
  float sendrecv_time_avg;
  float local_time_avg;
  float sendrecv_bw_avg;
};

// Helper structure to hold pre-computed transpose timing data
struct TransposeConfigTimingData {
  TransposePerformanceStats stats;
  std::vector<float> total_times;
  std::vector<float> alltoall_times;
  std::vector<float> local_times;
  std::vector<float> alltoall_bws;
  std::vector<int> sample_indices;
};

// Helper structure to hold pre-computed halo timing data
struct HaloConfigTimingData {
  HaloPerformanceStats stats;
  std::vector<float> total_times;
  std::vector<float> sendrecv_times;
  std::vector<float> local_times;
  std::vector<float> sendrecv_bws;
  std::vector<int> sample_indices;
};

void printPerformanceReport(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc);

void resetPerformanceSamples(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc);

void advanceTransposePerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       const cudecompTransposeConfigKey& config);

cudecompTransposePerformanceSampleCollection&
getOrCreateTransposePerformanceSamples(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       const cudecompTransposeConfigKey& config);

void advanceHaloPerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                  const cudecompHaloConfigKey& config);

cudecompHaloPerformanceSampleCollection& getOrCreateHaloPerformanceSamples(const cudecompHandle_t handle,
                                                                           cudecompGridDesc_t grid_desc,
                                                                           const cudecompHaloConfigKey& config);

// Helper function to create transpose configuration key
cudecompTransposeConfigKey createTransposeConfig(int ax, int dir, void* input, void* output,
                                                 const int32_t input_halo_extents_ptr[],
                                                 const int32_t output_halo_extents_ptr[],
                                                 const int32_t input_padding_ptr[], const int32_t output_padding_ptr[],
                                                 cudecompDataType_t datatype);

// Helper function to create halo configuration key
cudecompHaloConfigKey createHaloConfig(int ax, int dim, void* input, const int32_t halo_extents_ptr[],
                                       const bool halo_periods_ptr[], const int32_t padding_ptr[],
                                       cudecompDataType_t datatype);

} // namespace cudecomp

#endif // CUDECOMP_PERFORMANCE_H
