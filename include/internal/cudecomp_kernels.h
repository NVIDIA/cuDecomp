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

#include <cuda/std/complex>

namespace cudecomp {

#ifdef ENABLE_NVSHMEM
#define CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY 96
template <typename T> struct cudecompNvshmemA2AParams {
  int ntransfers;
  T* send_buff = nullptr;
  T* recv_buff = nullptr;
  size_t send_offsets[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t recv_offsets[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t send_counts[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  int peer_ranks[CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
};

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<float>& params, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<double>& params, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<float>>& params, cudaStream_t stream);
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<double>>& params,
                                cudaStream_t stream);
#endif

#define CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY 56
template <typename T> struct cudecompBatchedD2DMemcpy3DParams {
  int ncopies;
  T* src[CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  T* dest[CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  size_t src_strides[2][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];  // [depth stride, row stride] col_stride=1 assumed
  size_t dest_strides[2][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY]; // [depth stride, row stride] col_stride=1 assumed
  size_t extents[3][CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];      // [depth, height, width]
};

void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<float>& params, cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<double>& params, cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<float>>& params,
                                    cudaStream_t stream);
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<double>>& params,
                                    cudaStream_t stream);

} // namespace cudecomp

#endif // CUDECOMP_KERNELS_H
