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

#include <cstring>

#include "internal/cudecomp_kernels.h"

namespace cudecomp {

template <typename T>
static void batchedCopy(cudecompHandle_t handle, cudecompBatchedD2DMemcpy3DParams<T>& params, cudaStream_t stream) {
  for (int n = 0; n < params.ncopies; ++n) {
    if (params.extents[2][n] == 0) continue;
    for (size_t k = 0; k < params.extents[0][n]; ++k) {
      for (size_t j = 0; j < params.extents[1][n]; ++j) {
        std::memcpy(params.dest[n] + k * params.dest_strides[0][n] + j * params.dest_strides[1][n],
                    params.src[n] + k * params.src_strides[0][n] + j * params.src_strides[1][n],
                    params.extents[2][n] * sizeof(T));
      }
    }
  }
}

void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle, cudecompBatchedD2DMemcpy3DParams<float>& params,
                                    cudaStream_t stream) {
  batchedCopy(handle, params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle, cudecompBatchedD2DMemcpy3DParams<double>& params,
                                    cudaStream_t stream) {
  batchedCopy(handle, params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle,
                                    cudecompBatchedD2DMemcpy3DParams<cudecomp::complex<float>>& params,
                                    cudaStream_t stream) {
  batchedCopy(handle, params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompHandle_t handle,
                                    cudecompBatchedD2DMemcpy3DParams<cudecomp::complex<double>>& params,
                                    cudaStream_t stream) {
  batchedCopy(handle, params, stream);
}

} // namespace cudecomp
