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

#include <cuda/std/complex>

#include "internal/checks.h"
#include "internal/cudecomp_kernels.cuh"

namespace cudecomp {

#ifdef ENABLE_NVSHMEM
template <typename T>
static void launch_nvshmem_alltoallv(const cudecompNvshmemA2AParams<T>& params, uint64_t* sig_addr,
                                     cudaStream_t stream) {
  if (sig_addr) {
    cudecomp_nvshmem_alltoallv_signal_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, sig_addr);
  } else {
    cudecomp_nvshmem_alltoallv_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  }
  CHECK_CUDA_LAUNCH();
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<float>& params, uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv(params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<double>& params, uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv(params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<float>>& params, uint64_t* sig_addr,
                                cudaStream_t stream) {
  launch_nvshmem_alltoallv(params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<double>>& params, uint64_t* sig_addr,
                                cudaStream_t stream) {
  launch_nvshmem_alltoallv(params, sig_addr, stream);
}

static int nvshmem_p2p_nblocks(int ntransfers, cudecompHandle_t handle) {
  // Theoretical max blocks per SM based on thread count. Round down to a multiple of ntransfers so
  // that all launched blocks fit in a single resident wave and each copy receives exactly the same
  // number of blocks, ensuring even NVLink subscription across peers.
  int max_blocks_per_sm = handle->device_max_threads_per_sm / CUDECOMP_NVSHMEM_NTHREADS;
  int nblocks_max = max_blocks_per_sm * std::min(handle->device_num_sms, (int32_t)CUDECOMP_NVSHMEM_MAX_SMS);
  if (ntransfers == 0) return nblocks_max;
  int blocks_per_copy = std::max(1, nblocks_max / ntransfers);
  return blocks_per_copy * ntransfers;
}

template <typename T>
static void launch_nvshmem_alltoallv_p2p(cudecompHandle_t handle, const cudecompNvshmemP2PParams<T>& params,
                                         uint64_t* sig_addr, cudaStream_t stream) {
  int nblocks = nvshmem_p2p_nblocks(params.ntransfers, handle);
  cudecomp_nvshmem_alltoallv_p2p_k<<<nblocks, CUDECOMP_NVSHMEM_NTHREADS, 0, stream>>>(params, sig_addr);
  CHECK_CUDA_LAUNCH();
}

void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle, const cudecompNvshmemP2PParams<float>& params,
                                    uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv_p2p(handle, params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle, const cudecompNvshmemP2PParams<double>& params,
                                    uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv_p2p(handle, params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle,
                                    const cudecompNvshmemP2PParams<cuda::std::complex<float>>& params,
                                    uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv_p2p(handle, params, sig_addr, stream);
}

void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle,
                                    const cudecompNvshmemP2PParams<cuda::std::complex<double>>& params,
                                    uint64_t* sig_addr, cudaStream_t stream) {
  launch_nvshmem_alltoallv_p2p(handle, params, sig_addr, stream);
}
#endif

} // namespace cudecomp
