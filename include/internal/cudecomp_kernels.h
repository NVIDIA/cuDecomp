/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
