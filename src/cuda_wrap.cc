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

#include <hip/hip_runtime.h>

#include "internal/checks.h"
#include "internal/cuda_wrap.h"
#include "internal/exceptions.h"

#if CUDART_VERSION >= 13000
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    hipDriverEntryPointQueryResult driverStatus = hipDriverEntryPointSymbolNotFound;                                   \
    CHECK_CUDA(cudaGetDriverEntryPointByVersion(#symbol, (void**)(&cuFnTable.pfn_##symbol), version, hipEnableDefault, \
                                                &driverStatus));                                                       \
    if (driverStatus != hipDriverEntryPointSuccess) { THROW_CUDA_ERROR("cudaGetDriverEntryPointByVersion failed."); }  \
  } while (false)
#elif CUDART_VERSION >= 12000
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    hipDriverEntryPointQueryResult driverStatus = hipDriverEntryPointSymbolNotFound;                                   \
    CHECK_CUDA(hipGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), hipEnableDefault, &driverStatus));   \
    if (driverStatus != hipDriverEntryPointSuccess) { THROW_CUDA_ERROR("hipGetDriverEntryPoint failed."); }            \
  } while (false)
#else
#define LOAD_SYM(symbol, version)                                                                                      \
  do {                                                                                                                 \
    CHECK_CUDA(hipGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), hipEnableDefault));                  \
  } while (false)
#endif

namespace cudecomp {

cuFunctionTable cuFnTable; // global table of required CUDA driver functions

void initCuFunctionTable() {
#if CUDART_VERSION >= 11030
  LOAD_SYM(hipDeviceGet, 2000);
  LOAD_SYM(hipDeviceGetAttribute, 2000);
  LOAD_SYM(hipDrvGetErrorString, 6000);
  LOAD_SYM(hipMemAddressFree, 10020);
  LOAD_SYM(hipMemAddressReserve, 10020);
  LOAD_SYM(hipMemCreate, 10020);
  LOAD_SYM(hipMemGetAddressRange, 3020);
  LOAD_SYM(hipMemGetAllocationGranularity, 10020);
  LOAD_SYM(hipMemMap, 10020);
  LOAD_SYM(hipMemRetainAllocationHandle, 11000);
  LOAD_SYM(hipMemRelease, 10020);
  LOAD_SYM(hipMemSetAccess, 10020);
  LOAD_SYM(hipMemUnmap, 10020);
#endif
}

} // namespace cudecomp

#undef LOAD_SYM
