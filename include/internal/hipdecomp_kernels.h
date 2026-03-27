/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 The Authors.
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

#ifndef HIPDECOMP_KERNELS_H
#define HIPDECOMP_KERNELS_H

#include <complex>

namespace hipdecomp {

#ifdef ENABLE_NVSHMEM
#define HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY 96
template <typename T> struct hipdecompNvshmemA2AParams {
  int ntransfers;
  T* send_buff = nullptr;
  T* recv_buff = nullptr;
  size_t send_offsets[HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t recv_offsets[HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  size_t send_counts[HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
  int peer_ranks[HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY];
};

void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<float>& params, hipStream_t stream);
void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<double>& params, hipStream_t stream);
void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<std::complex<float>>& params, hipStream_t stream);
void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<std::complex<double>>& params, hipStream_t stream);
#endif

#define HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY 56
template <typename T> struct hipdecompBatchedD2DMemcpy3DParams {
  int ncopies;
  T* src[HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  T* dest[HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];
  size_t src_strides[2][HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];  // [depth stride, row stride] col_stride=1 assumed
  size_t dest_strides[2][HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY]; // [depth stride, row stride] col_stride=1 assumed
  size_t extents[3][HIPDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY];      // [depth, height, width]
};

void hipdecomp_batched_d2d_memcpy_3d(hipdecompBatchedD2DMemcpy3DParams<float>& params, hipStream_t stream);
void hipdecomp_batched_d2d_memcpy_3d(hipdecompBatchedD2DMemcpy3DParams<double>& params, hipStream_t stream);
void hipdecomp_batched_d2d_memcpy_3d(hipdecompBatchedD2DMemcpy3DParams<std::complex<float>>& params,
                                     hipStream_t stream);
void hipdecomp_batched_d2d_memcpy_3d(hipdecompBatchedD2DMemcpy3DParams<std::complex<double>>& params,
                                     hipStream_t stream);

} // namespace hipdecomp

#endif // HIPDECOMP_KERNELS_H
