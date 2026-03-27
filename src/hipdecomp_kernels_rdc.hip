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

#include <complex>

#include "internal/checks.h"
#include "internal/hipdecomp_kernels.hiph"

namespace hipdecomp {

#ifdef ENABLE_NVSHMEM
void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<float>& params, hipStream_t stream) {
  hipdecomp_nvshmem_alltoallv_k<<<1, HIPDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_HIP_LAUNCH();
}

void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<double>& params, hipStream_t stream) {
  hipdecomp_nvshmem_alltoallv_k<<<1, HIPDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_HIP_LAUNCH();
}

void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<std::complex<float>>& params, hipStream_t stream) {
  hipdecomp_nvshmem_alltoallv_k<<<1, HIPDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_HIP_LAUNCH();
}

void hipdecomp_nvshmem_alltoallv(const hipdecompNvshmemA2AParams<std::complex<double>>& params, hipStream_t stream) {
  hipdecomp_nvshmem_alltoallv_k<<<1, HIPDECOMP_CUDA_NTHREADS, 0, stream>>>(params);
  CHECK_HIP_LAUNCH();
}
#endif

} // namespace hipdecomp
