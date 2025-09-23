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
