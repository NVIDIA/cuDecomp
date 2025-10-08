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

#if CUDART_VERSION >= 13000
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound;                                 \
    CHECK_CUDA(cudaGetDriverEntryPointByVersion(#symbol, (void**)(&cuFnTable.pfn_##symbol), version,                   \
                                                cudaEnableDefault, &driverStatus));                                    \
    if (driverStatus != cudaDriverEntryPointSuccess) { THROW_CUDA_ERROR("cudaGetDriverEntryPointByVersion failed."); } \
  } while (false)
#elif CUDART_VERSION >= 12000
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound;                                 \
    CHECK_CUDA(cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault, &driverStatus)); \
    if (driverStatus != cudaDriverEntryPointSuccess) { THROW_CUDA_ERROR("cudaGetDriverEntryPoint failed."); }          \
  } while (false)
#else
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    CHECK_CUDA(cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault));                \
  } while (false)
#endif

namespace cudecomp {

cuFunctionTable cuFnTable; // global table of required CUDA driver functions

void initCuFunctionTable() {
#if CUDART_VERSION >= 11030
  LOAD_SYM(cuDeviceGet, 2000);
  LOAD_SYM(cuDeviceGetAttribute, 2000);
  LOAD_SYM(cuGetErrorString, 6000);
  LOAD_SYM(cuMemAddressFree, 10020);
  LOAD_SYM(cuMemAddressReserve, 10020);
  LOAD_SYM(cuMemCreate, 10020);
  LOAD_SYM(cuMemGetAddressRange, 3020);
  LOAD_SYM(cuMemGetAllocationGranularity, 10020);
  LOAD_SYM(cuMemMap, 10020);
  LOAD_SYM(cuMemRetainAllocationHandle, 11000);
  LOAD_SYM(cuMemRelease, 10020);
  LOAD_SYM(cuMemSetAccess, 10020);
  LOAD_SYM(cuMemUnmap, 10020);
#endif
}

} // namespace cudecomp

#undef LOAD_SYM
