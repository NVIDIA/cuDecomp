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

#include "internal/checks.h"
#include "internal/cuda_wrap.h"
#include "internal/exceptions.h"

#if CUDART_VERSION >= 12000
#define LOAD_SYM(symbol)                                                                                               \
  do {                                                                                                                 \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound;                                 \
    CHECK_CUDA(cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault, &driverStatus)); \
    if (driverStatus != cudaDriverEntryPointSuccess) { THROW_CUDA_ERROR("cudaGetDriverEntryPoint failed."); }          \
  } while (false)
#else
#define LOAD_SYM(symbol)                                                                                               \
  do {                                                                                                                 \
    CHECK_CUDA(cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault));                \
  } while (false)
#endif

namespace cudecomp {

cuFunctionTable cuFnTable; // global table of required CUDA driver functions

void initCuFunctionTable() {
#if CUDART_VERSION >= 11030
  LOAD_SYM(cuDeviceGet);
  LOAD_SYM(cuDeviceGetAttribute);
  LOAD_SYM(cuGetErrorString);
  LOAD_SYM(cuMemAddressFree);
  LOAD_SYM(cuMemAddressReserve);
  LOAD_SYM(cuMemCreate);
  LOAD_SYM(cuMemGetAddressRange);
  LOAD_SYM(cuMemGetAllocationGranularity);
  LOAD_SYM(cuMemMap);
  LOAD_SYM(cuMemRetainAllocationHandle);
  LOAD_SYM(cuMemRelease);
  LOAD_SYM(cuMemSetAccess);
  LOAD_SYM(cuMemUnmap);
#endif
}

} // namespace cudecomp

#undef LOAD_SYM
