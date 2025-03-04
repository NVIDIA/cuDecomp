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

#include <cuda_runtime.h>
#include <dlfcn.h>

#include "internal/checks.h"
#include "internal/nvml_wrap.h"
#include "internal/exceptions.h"

#define LOAD_SYM(symbol)                                                        \
  do {                                                                          \
    void **fptr = (void**)(&nvmlFnTable.pfn_##symbol);                          \
    void *sym = dlsym(nvml_handle, #symbol);                                    \
    *fptr = sym;                                                                \
  } while(false)

namespace cudecomp {

nvmlFunctionTable nvmlFnTable; // global table of required NVML functions

void initNvmlFunctionTable() {
  void *nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvml_handle) {
    THROW_INVALID_USAGE("Could not dlopen libnvidia-ml.so.1");
  }
  LOAD_SYM(nvmlInit);
  LOAD_SYM(nvmlShutdown);
  LOAD_SYM(nvmlErrorString);
  LOAD_SYM(nvmlDeviceGetHandleByPciBusId);
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  LOAD_SYM(nvmlDeviceGetGpuFabricInfoV);
#endif
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
