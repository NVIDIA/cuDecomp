/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDECOMP_AUTOTUNE_H
#define CUDECOMP_AUTOTUNE_H

#include <array>
#include <cstdint>
#include <vector>

#include "cudecomp.h"

namespace cudecomp {
std::vector<cudecompTransposeCommBackend_t>
getAutotuneTransposeBackendCandidates(const cudecompGridDescAutotuneOptions_t* options);
std::vector<cudecompHaloCommBackend_t>
getAutotuneHaloBackendCandidates(const cudecompGridDescAutotuneOptions_t* options);
std::vector<std::array<int32_t, 2>> getAutotunePdimCandidates(int nranks, cudecompRankOrder_t rank_order);

void autotuneTransposeBackend(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                              const cudecompGridDescAutotuneOptions_t* options);
void autotuneHaloBackend(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                         const cudecompGridDescAutotuneOptions_t* options);
} // namespace cudecomp

#endif // CUDECOMP_AUTOTUNE_H
