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

namespace cudecomp {

struct cuFunctionTable {
#if CUDART_VERSION >= 11030
  PFN_cuDeviceGet pfn_cuDeviceGet = nullptr;
  PFN_cuDeviceGetAttribute pfn_cuDeviceGetAttribute = nullptr;
  PFN_cuGetErrorString pfn_cuGetErrorString = nullptr;
  PFN_cuMemAddressFree pfn_cuMemAddressFree = nullptr;
  PFN_cuMemAddressReserve pfn_cuMemAddressReserve = nullptr;
  PFN_cuMemCreate pfn_cuMemCreate = nullptr;
  PFN_cuMemGetAddressRange pfn_cuMemGetAddressRange = nullptr;
  PFN_cuMemGetAllocationGranularity pfn_cuMemGetAllocationGranularity = nullptr;
  PFN_cuMemMap pfn_cuMemMap = nullptr;
  PFN_cuMemRetainAllocationHandle pfn_cuMemRetainAllocationHandle = nullptr;
  PFN_cuMemRelease pfn_cuMemRelease = nullptr;
  PFN_cuMemSetAccess pfn_cuMemSetAccess = nullptr;
  PFN_cuMemUnmap pfn_cuMemUnmap = nullptr;
#endif
};

extern cuFunctionTable cuFnTable;

void initCuFunctionTable();

} // namespace cudecomp

#endif // CUDECOMP_CUDA_WRAP_H
