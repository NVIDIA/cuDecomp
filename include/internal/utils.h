/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
