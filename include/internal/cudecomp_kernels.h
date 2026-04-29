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

#ifndef CUDECOMP_KERNELS_H
#define CUDECOMP_KERNELS_H

#include <cstddef>
#include <cstdint>

#include <cuda/std/complex>

#include "internal/common.h"

namespace cudecomp {

#ifdef ENABLE_NVSHMEM

// Capacity for the inter-group alltoallv kernel.
#define CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY (96)
template <typename T> struct cudecompNvshmemA2AParams {
  T* send_buff = nullptr;
  T* recv_buff = nullptr;
  size_t send_offsets[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t recv_offsets[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t send_counts[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  int peer_ranks[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  int ntransfers;
};

// Capacity for the intra-group SM P2P kernel.
#define CUDECOMP_NVSHMEM_MAX_SMS (32)
#define CUDECOMP_NVSHMEM_P2P_PARAM_CAPACITY (CUDECOMP_NVSHMEM_MAX_SMS * 2)
template <typename T> struct cudecompNvshmemP2PParams {
  T* send_buff = nullptr;
  T* recv_buff = nullptr;
  int* block_counters = nullptr;
  size_t send_offsets[CUDECOMP_NVSHMEM_P2P_PARAM_CAPACITY];
  size_t recv_offsets[CUDECOMP_NVSHMEM_P2P_PARAM_CAPACITY];
  size_t send_counts[CUDECOMP_NVSHMEM_P2P_PARAM_CAPACITY];
  int peer_ranks[CUDECOMP_NVSHMEM_P2P_PARAM_CAPACITY];
  int ntransfers;
};

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<float>& params, uint64_t* sig_addr, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<double>& params, uint64_t* sig_addr,
                                cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<float>>& params, uint64_t* sig_addr,
                                cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<double>>& params, uint64_t* sig_addr,
                                cudaStream_t stream);

void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle, const cudecompNvshmemP2PParams<float>& params,
                                    uint64_t* sig_addr, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle, const cudecompNvshmemP2PParams<double>& params,
                                    uint64_t* sig_addr, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle,
                                    const cudecompNvshmemP2PParams<cuda::std::complex<float>>& params,
                                    uint64_t* sig_addr, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv_p2p(cudecompHandle_t handle,
                                    const cudecompNvshmemP2PParams<cuda::std::complex<double>>& params,
                                    uint64_t* sig_addr, cudaStream_t stream);
#endif

#define CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY 56
template <typename T> struct cudecompBatchedD2DMemcpy3DParams {
  T* src[CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  T* dest[CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  size_t src_strides[2][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];  // [depth stride, row stride] col_stride=1 assumed
  size_t dest_strides[2][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY]; // [depth stride, row stride] col_stride=1 assumed
  size_t extents[3][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];      // [depth, height, width]
  int ncopies;
};

void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle, cudecompBatchedD2DMemcpy3DParams<float>& params,
                                    cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle, cudecompBatchedD2DMemcpy3DParams<double>& params,
                                    cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle,
                                    cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<float>>& params,
                                    cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle,
                                    cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<double>>& params,
                                    cudaStream_t stream);

} // namespace cudecomp

#endif // CUDECOMP_KERNELS_H
