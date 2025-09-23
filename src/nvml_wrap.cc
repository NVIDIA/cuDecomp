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

#include <cuda_runtime.h>
#include <dlfcn.h>

#include "internal/checks.h"
#include "internal/exceptions.h"
#include "internal/nvml_wrap.h"

#define LOAD_SYM(symbol)                                                                                               \
  do {                                                                                                                 \
    void** fptr = (void**)(&nvmlFnTable.pfn_##symbol);                                                                 \
    void* sym = dlsym(nvml_handle, #symbol);                                                                           \
    *fptr = sym;                                                                                                       \
  } while (false)

namespace cudecomp {

nvmlFunctionTable nvmlFnTable; // global table of required NVML functions

void initNvmlFunctionTable() {
  void* nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvml_handle) { THROW_INVALID_USAGE("Could not dlopen libnvidia-ml.so.1"); }
  LOAD_SYM(nvmlInit);
  LOAD_SYM(nvmlShutdown);
  LOAD_SYM(nvmlErrorString);
  LOAD_SYM(nvmlDeviceGetFieldValues);
  LOAD_SYM(nvmlDeviceGetHandleByPciBusId);
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  LOAD_SYM(nvmlDeviceGetGpuFabricInfoV);
#endif
  LOAD_SYM(nvmlDeviceGetNvLinkCapability);
  LOAD_SYM(nvmlDeviceGetNvLinkState);
  LOAD_SYM(nvmlDeviceGetNvLinkRemotePciInfo);
}

bool nvmlHasFabricSupport() {
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  return (nvmlFnTable.pfn_nvmlDeviceGetGpuFabricInfoV != nullptr);
#else
  return false;
#endif
}

} // namespace cudecomp

#undef LOAD_SYM
