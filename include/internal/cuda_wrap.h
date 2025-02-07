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
