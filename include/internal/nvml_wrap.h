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

#ifndef CUDECOMP_NVML_WRAP_H
#define CUDECOMP_NVML_WRAP_H

#include <hip/hip_runtime.h>
#include <nvml.h>

namespace cudecomp {

struct nvmlFunctionTable {
  nvmlReturn_t (*pfn_nvmlInit)(void) = nullptr;
  nvmlReturn_t (*pfn_nvmlShutdown)(void) = nullptr;
  const char* (*pfn_nvmlErrorString)(nvmlReturn_t result) = nullptr;
  nvmlReturn_t (*pfn_nvmlDeviceGetFieldValues)(nvmlDevice_t device, unsigned int fieldCount,
                                               nvmlFieldValue_t* fieldValues) = nullptr;
  nvmlReturn_t (*pfn_nvmlDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device) = nullptr;
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  nvmlReturn_t (*pfn_nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) = nullptr;
#endif
  nvmlReturn_t (*pfn_nvmlDeviceGetNvLinkCapability)(nvmlDevice_t device, unsigned int linkIndex,
                                                    nvmlNvLinkCapability_t capability, unsigned int* value) = nullptr;
  nvmlReturn_t (*pfn_nvmlDeviceGetNvLinkState)(nvmlDevice_t device, unsigned int linkIndex,
                                               nvmlEnableState_t* state) = nullptr;
  nvmlReturn_t (*pfn_nvmlDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t device, unsigned int linkIndex,
                                                       nvmlPciInfo_t* pciInfo) = nullptr;
};

extern nvmlFunctionTable nvmlFnTable;

void initNvmlFunctionTable();

bool nvmlHasFabricSupport();

} // namespace cudecomp

#endif // CUDECOMP_NVML_WRAP_H
