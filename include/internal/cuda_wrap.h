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

#ifndef CUDECOMP_CUDA_WRAP_H
#define CUDECOMP_CUDA_WRAP_H

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#endif

#define DECLARE_CUDA_PFN(symbol, version) PFN_##symbol##_v##version pfn_##symbol = nullptr

namespace cudecomp {

struct cuFunctionTable {
#if CUDART_VERSION >= 11030
  DECLARE_CUDA_PFN(hipDeviceGet, 2000);
  DECLARE_CUDA_PFN(hipDeviceGetAttribute, 2000);
  DECLARE_CUDA_PFN(hipDrvGetErrorString, 6000);
  DECLARE_CUDA_PFN(hipMemAddressFree, 10020);
  DECLARE_CUDA_PFN(hipMemAddressReserve, 10020);
  DECLARE_CUDA_PFN(hipMemCreate, 10020);
  DECLARE_CUDA_PFN(hipMemGetAddressRange, 3020);
  DECLARE_CUDA_PFN(hipMemGetAllocationGranularity, 10020);
  DECLARE_CUDA_PFN(hipMemMap, 10020);
  DECLARE_CUDA_PFN(hipMemRetainAllocationHandle, 11000);
  DECLARE_CUDA_PFN(hipMemRelease, 10020);
  DECLARE_CUDA_PFN(hipMemSetAccess, 10020);
  DECLARE_CUDA_PFN(hipMemUnmap, 10020);
#endif
};

extern cuFunctionTable cuFnTable;

void initCuFunctionTable();

} // namespace cudecomp

#undef DECLARE_CUDA_PFN

#endif // CUDECOMP_CUDA_WRAP_H
