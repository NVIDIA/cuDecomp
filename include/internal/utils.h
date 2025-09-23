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

#ifndef CUDECOMP_UTILS_H
#define CUDECOMP_UTILS_H

#include <cuda/std/complex>

#include "cudecomp.h"

inline bool operator==(const cudecompPencilInfo_t& a, const cudecompPencilInfo_t& b) {
  if (a.size != b.size) return false;
  for (int i = 0; i < 3; ++i) {
    if ((a.shape[i] != b.shape[i]) || (a.lo[i] != b.lo[i]) || (a.hi[i] != b.hi[i]) || (a.order[i] != b.order[i]) ||
        (a.halo_extents[i] != b.halo_extents[i]) || (a.padding[i] != b.padding[i])) {
      return false;
    }
  }
  return true;
}

inline cudecompDataType_t getCudecompDataType(float) { return CUDECOMP_FLOAT; }
inline cudecompDataType_t getCudecompDataType(double) { return CUDECOMP_DOUBLE; }
inline cudecompDataType_t getCudecompDataType(cuda::std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
inline cudecompDataType_t getCudecompDataType(cuda::std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }
template <typename T> inline cudecompDataType_t getCudecompDataType() { return getCudecompDataType(T(0)); }

#endif // CUDECOMP_UTILS_H
