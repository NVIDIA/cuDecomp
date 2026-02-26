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

#ifndef CUDECOMP_KERNELS_CUH
#define CUDECOMP_KERNELS_CUH

#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#endif

#include "internal/cudecomp_kernels.h"

#define CUDECOMP_CUDA_NTHREADS (128)
#define CUDECOMP_UNROLL_FACTOR (4)
#define CUDECOMP_MIN_BLOCKS_PER_SM (16)

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

template <int src_nd, int dest_nd, typename T>
__launch_bounds__(CUDECOMP_CUDA_NTHREADS) __global__
    void cudecomp_batched_d2d_memcpy_3d_k(cudecompBatchedD2DMemcpy3DParams<T> params, int blocks_per_copy) {

  int copyid = blockIdx.x / blocks_per_copy;

  if (copyid >= params.ncopies) return;

  T* src = params.src[copyid];
  T* dest = params.dest[copyid];
  const size_t width = params.extents[2][copyid];
  const size_t height = params.extents[1][copyid];
  const size_t depth = params.extents[0][copyid];

  size_t src_stride_r, src_stride_d;
  if (src_nd > 1) { src_stride_r = params.src_strides[1][copyid]; }
  if (src_nd > 2) { src_stride_d = params.src_strides[0][copyid]; }

  size_t dest_stride_r, dest_stride_d;
  if (dest_nd > 1) { dest_stride_r = params.dest_strides[1][copyid]; }
  if (dest_nd > 2) { dest_stride_d = params.dest_strides[0][copyid]; }

  const size_t tid = (blockIdx.x % blocks_per_copy) * blockDim.x + threadIdx.x;

  for (size_t n = tid; n < width * height * depth; n += blocks_per_copy * blockDim.x) {
    T *dptr, *sptr;

    if (src_nd == 1) {
      sptr = src + n;
    } else if (src_nd == 2) {
      size_t c = n % width;
      size_t r = n / width;
      sptr = src + src_stride_r * r + c;
    } else if (src_nd == 3) {
      size_t c = (n % width);
      size_t r = (n / width) % height;
      size_t d = n / (width * height);
      sptr = src + src_stride_d * d + src_stride_r * r + c;
    }

    if (dest_nd == 1) {
      dptr = dest + n;
    } else if (dest_nd == 2) {
      size_t c = n % width;
      size_t r = n / width;
      dptr = dest + dest_stride_r * r + c;
    } else if (dest_nd == 3) {
      size_t c = (n % width);
      size_t r = (n / width) % height;
      size_t d = n / (width * height);
      dptr = dest + dest_stride_d * d + dest_stride_r * r + c;
    }

    *dptr = *sptr;
  }
}

template <typename T>
void cudecomp_batched_d2d_memcpy_3d_nd_dispatch(const cudecompBatchedD2DMemcpy3DParams<T>& params, hipStream_t stream) {
  size_t N = params.extents[0][0] * params.extents[1][0] * params.extents[2][0];

  // Determine reduced copy dimension to simplify indexing
  int src_nd = 1;
  int dest_nd = 1;
  for (int i = 0; i < params.ncopies; ++i) {
    N = std::max(N, params.extents[0][i] * params.extents[1][i] * params.extents[2][i]);
    if (params.src_strides[1][i] == params.extents[2][i] &&
        params.src_strides[0][i] / params.src_strides[1][i] == params.extents[1][i]) {
      src_nd = std::max(1, src_nd);
    } else if (params.src_strides[0][i] / params.src_strides[1][i] == params.extents[1][i]) {
      src_nd = std::max(2, src_nd);
    } else {
      src_nd = 3;
    }
    if (params.dest_strides[1][i] == params.extents[2][i] &&
        params.dest_strides[0][i] / params.dest_strides[1][i] == params.extents[1][i]) {
      dest_nd = std::max(1, dest_nd);
    } else if (params.dest_strides[0][i] / params.dest_strides[1][i] == params.extents[1][i]) {
      dest_nd = std::max(2, dest_nd);
    } else {
      dest_nd = 3;
    }
  }

  int blocks_per_copy = (N + CUDECOMP_CUDA_NTHREADS - 1) / CUDECOMP_CUDA_NTHREADS;
  int blocks_per_copy_unroll = (blocks_per_copy + CUDECOMP_UNROLL_FACTOR - 1) / CUDECOMP_UNROLL_FACTOR;
  size_t total_blocks_unroll = params.ncopies * blocks_per_copy_unroll;

  // Clamp minimum number of blocks from unrolling
  int dev, num_sms;
  CHECK_CUDA(hipGetDevice(&dev));
  CHECK_CUDA(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, dev));

  if (total_blocks_unroll > CUDECOMP_MIN_BLOCKS_PER_SM * num_sms) { blocks_per_copy = blocks_per_copy_unroll; }

  switch (src_nd) {
  case 1:
    switch (dest_nd) {
    case 1:
      cudecomp_batched_d2d_memcpy_3d_k<1, 1>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 2:
      cudecomp_batched_d2d_memcpy_3d_k<1, 2>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 3:
      cudecomp_batched_d2d_memcpy_3d_k<1, 3>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    }
    break;
  case 2:
    switch (dest_nd) {
    case 1:
      cudecomp_batched_d2d_memcpy_3d_k<2, 1>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 2:
      cudecomp_batched_d2d_memcpy_3d_k<2, 2>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 3:
      cudecomp_batched_d2d_memcpy_3d_k<2, 3>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    }
    break;
  case 3:
    switch (dest_nd) {
    case 1:
      cudecomp_batched_d2d_memcpy_3d_k<3, 1>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 2:
      cudecomp_batched_d2d_memcpy_3d_k<3, 2>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    case 3:
      cudecomp_batched_d2d_memcpy_3d_k<3, 3>
          <<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params, blocks_per_copy);
      break;
    }
    break;
  }
  CHECK_CUDA_LAUNCH();
}

} // namespace cudecomp

#endif // CUDECOMP_KERNELS_CUH
