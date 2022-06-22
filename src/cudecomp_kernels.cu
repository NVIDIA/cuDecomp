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

#include <cuda/std/complex>

#include "internal/checks.h"
#include "internal/cudecomp_kernels.cuh"

namespace cudecomp {

#ifdef ENABLE_NVSHMEM
void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<float>& params, cudaStream_t stream) {
  cudecomp_nvshmem_alltoallv_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_CUDA_LAUNCH();
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<double>& params, cudaStream_t stream) {
  cudecomp_nvshmem_alltoallv_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_CUDA_LAUNCH();
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<float>>& params,
                                cudaStream_t stream) {
  cudecomp_nvshmem_alltoallv_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_CUDA_LAUNCH();
}

void cudecomp_nvshmem_alltoallv(const cudecompNvshmemA2AParams<cuda::std::complex<double>>& params,
                                cudaStream_t stream) {
  cudecomp_nvshmem_alltoallv_k<<<1, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_CUDA_LAUNCH();
}
#endif

void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<float>& params, cudaStream_t stream) {
  size_t N = params.extents[0][0] * params.extents[1][0] * params.extents[2][0];
  for (int i = 1; i < params.ncopies; ++i) {
    N = std::max(N, params.extents[0][i] * params.extents[1][i] * params.extents[2][i]);
  }
  int blocks_per_copy = (N + CUDECOMP_CUDA_NTHREADS - 1) / CUDECOMP_CUDA_NTHREADS;
  cudecomp_batched_d2d_memcpy_3d_k<<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(
      params, blocks_per_copy);
  CHECK_CUDA_LAUNCH();
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<double>& params, cudaStream_t stream) {
  size_t N = params.extents[0][0] * params.extents[1][0] * params.extents[2][0];
  for (int i = 1; i < params.ncopies; ++i) {
    N = std::max(N, params.extents[0][i] * params.extents[1][i] * params.extents[2][i]);
  }
  int blocks_per_copy = (N + CUDECOMP_CUDA_NTHREADS - 1) / CUDECOMP_CUDA_NTHREADS;
  cudecomp_batched_d2d_memcpy_3d_k<<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(
      params, blocks_per_copy);
  CHECK_CUDA_LAUNCH();
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<float>>& params,
                                    cudaStream_t stream) {
  size_t N = params.extents[0][0] * params.extents[1][0] * params.extents[2][0];
  for (int i = 1; i < params.ncopies; ++i) {
    N = std::max(N, params.extents[0][i] * params.extents[1][i] * params.extents[2][i]);
  }
  int blocks_per_copy = (N + CUDECOMP_CUDA_NTHREADS - 1) / CUDECOMP_CUDA_NTHREADS;
  cudecomp_batched_d2d_memcpy_3d_k<<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(
      params, blocks_per_copy);
  CHECK_CUDA_LAUNCH();
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<double>>& params,
                                    cudaStream_t stream) {
  size_t N = params.extents[0][0] * params.extents[1][0] * params.extents[2][0];
  for (int i = 1; i < params.ncopies; ++i) {
    N = std::max(N, params.extents[0][i] * params.extents[1][i] * params.extents[2][i]);
  }
  int blocks_per_copy = (N + CUDECOMP_CUDA_NTHREADS - 1) / CUDECOMP_CUDA_NTHREADS;

  cudecomp_batched_d2d_memcpy_3d_k<<<params.ncopies * blocks_per_copy, CUDECOMP_CUDA_NTHREADS, 0, stream>>>(
      params, blocks_per_copy);
  CHECK_CUDA_LAUNCH();
}

} // namespace cudecomp
