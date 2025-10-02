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
  DECLARE_CUDA_PFN(cuDeviceGet, 2000);
  DECLARE_CUDA_PFN(cuDeviceGetAttribute, 2000);
  DECLARE_CUDA_PFN(cuGetErrorString, 6000);
  DECLARE_CUDA_PFN(cuMemAddressFree, 10020);
  DECLARE_CUDA_PFN(cuMemAddressReserve, 10020);
  DECLARE_CUDA_PFN(cuMemCreate, 10020);
  DECLARE_CUDA_PFN(cuMemGetAddressRange, 3020);
  DECLARE_CUDA_PFN(cuMemGetAllocationGranularity, 10020);
  DECLARE_CUDA_PFN(cuMemMap, 10020);
  DECLARE_CUDA_PFN(cuMemRetainAllocationHandle, 11000);
  DECLARE_CUDA_PFN(cuMemRelease, 10020);
  DECLARE_CUDA_PFN(cuMemSetAccess, 10020);
  DECLARE_CUDA_PFN(cuMemUnmap, 10020);
#endif
};

extern cuFunctionTable cuFnTable;

void initCuFunctionTable();

} // namespace cudecomp

#undef DECLARE_CUDA_PFN

#endif // CUDECOMP_CUDA_WRAP_H
