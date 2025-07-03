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

#ifndef CUDECOMP_PERFORMANCE_H
#define CUDECOMP_PERFORMANCE_H

#include <array>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "internal/common.h"

namespace cudecomp {

using cudecompTransposeConfigKey = std::tuple<
  int32_t,                         // ax (axis)
  int32_t,                         // dir (direction)
  std::array<int32_t, 3>,          // input_halo_extents
  std::array<int32_t, 3>,          // output_halo_extents
  std::array<int32_t, 3>,          // input_padding
  std::array<int32_t, 3>,          // output_padding
  bool,                            // inplace
  bool,                            // managed_memory
  cudecompDataType_t               // datatype
>;

void printPerformanceReport(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                           int ax, int dir, size_t alltoall_bytes, cudecompPerformanceSample* current_sample);

void printFinalPerformanceReport(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc);

void resetPerformanceSamples(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc);

void advancePerformanceSample(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                             const cudecompTransposeConfigKey& config);

cudecompPerformanceSampleCollection& getOrCreatePerformanceSamples(const cudecompHandle_t handle,
                                                                   cudecompGridDesc_t grid_desc,
                                                                   const cudecompTransposeConfigKey& config);

// Helper function to create transpose configuration key
cudecompTransposeConfigKey createTransposeConfig(int ax, int dir, void* input, void* output,
                                                const int32_t input_halo_extents_ptr[],
                                                const int32_t output_halo_extents_ptr[],
                                                const int32_t input_padding_ptr[],
                                                const int32_t output_padding_ptr[],
                                                cudecompDataType_t datatype);

} // namespace cudecomp

#endif // CUDECOMP_PERFORMANCE_H
