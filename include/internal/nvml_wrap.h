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

#ifndef CUDECOMP_NVML_WRAP_H
#define CUDECOMP_NVML_WRAP_H

#include <cuda_runtime.h>
#include <nvml.h>

namespace cudecomp {

struct nvmlFunctionTable {
  nvmlReturn_t (*pfn_nvmlInit)(void) = nullptr;
  nvmlReturn_t (*pfn_nvmlShutdown)(void) = nullptr;
  const char* (*pfn_nvmlErrorString)(nvmlReturn_t result) = nullptr;
  nvmlReturn_t (*pfn_nvmlDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device) = nullptr;
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  nvmlReturn_t (*pfn_nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) = nullptr;
#endif
};

extern nvmlFunctionTable nvmlFnTable;

void initNvmlFunctionTable();

bool nvmlHasFabricSupport();

} // namespace cudecomp

#endif // CUDECOMP_NVML_WRAP_H
