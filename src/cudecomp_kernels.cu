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

void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<float>& params, hipStream_t stream) {
  cudecomp_batched_d2d_memcpy_3d_nd_dispatch(params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<double>& params, hipStream_t stream) {
  cudecomp_batched_d2d_memcpy_3d_nd_dispatch(params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<float>>& params,
                                    hipStream_t stream) {
  cudecomp_batched_d2d_memcpy_3d_nd_dispatch(params, stream);
}
void cudecomp_batched_d2d_memcpy_3d(cudecompBatchedD2DMemcpy3DParams<cuda::std::complex<double>>& params,
                                    hipStream_t stream) {
  cudecomp_batched_d2d_memcpy_3d_nd_dispatch(params, stream);
}

} // namespace cudecomp
