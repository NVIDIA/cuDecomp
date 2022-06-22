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

#ifndef CUDECOMP_KERNELS_CUH
#define CUDECOMP_KERNELS_CUH

#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#endif

#include "internal/cudecomp_kernels.h"

#define CUDECOMP_CUDA_NTHREADS 256

namespace cudecomp {

#ifdef ENABLE_NVSHMEM
template <typename T>
__launch_bounds__(CUDECOMP_CUDA_NTHREADS) __global__
    void cudecomp_nvshmem_alltoallv_k(cudecompNvshmemA2AParams<T> params) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= params.ntransfers) return;

  int peer_rank = params.peer_ranks[tid];
  T* send_buff = params.send_buff;
  T* recv_buff = params.recv_buff;
  size_t send_offset = params.send_offsets[tid];
  size_t recv_offset = params.recv_offsets[tid];
  size_t send_count = params.send_counts[tid];

  nvshmem_putmem_nbi(recv_buff + recv_offset, send_buff + send_offset, send_count * sizeof(T), peer_rank);
}
#endif

template <typename T>
__launch_bounds__(CUDECOMP_CUDA_NTHREADS) __global__
    void cudecomp_batched_d2d_memcpy_3d_k(cudecompBatchedD2DMemcpy3DParams<T> params, int blocks_per_copy) {

  int copyid = blockIdx.x / blocks_per_copy;

  if (copyid >= params.ncopies) return;

  T* src = params.src[copyid];
  T* dest = params.dest[copyid];
  const size_t src_stride_r = params.src_strides[1][copyid];
  const size_t src_stride_d = params.src_strides[0][copyid];
  const size_t dest_stride_r = params.dest_strides[1][copyid];
  const size_t dest_stride_d = params.dest_strides[0][copyid];
  const size_t width = params.extents[2][copyid];
  const size_t height = params.extents[1][copyid];
  const size_t depth = params.extents[0][copyid];

  const size_t tid = (blockIdx.x % blocks_per_copy) * blockDim.x + threadIdx.x;

  for (size_t n = tid; n < width * height * depth; n += blocks_per_copy * blockDim.x) {
    int c = (n % width);
    int r = (n / width) % height;
    int d = n / (width * height);

    dest[dest_stride_d * d + dest_stride_r * r + c] = src[src_stride_d * d + src_stride_r * r + c];
  }
}

} // namespace cudecomp

#endif // CUDECOMP_KERNELS_CUH
