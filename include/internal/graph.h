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

#ifndef CUDECOMP_GRAPH_H
#define CUDECOMP_GRAPH_H

#include <tuple>
#include <unordered_map>

#include <cuda_runtime.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/hashes.h"
#include "internal/utils.h"

namespace cudecomp {

class graphCache {
  using key_type = std::tuple<void*, void*, int, int, cudecompPencilInfo_t, cudecompPencilInfo_t, cudecompDataType_t>;

public:
  graphCache();
  ~graphCache();
  void replay(const key_type& key, cudaStream_t stream) const;
  cudaStream_t startCapture(const key_type& key, cudaStream_t stream) const;
  void endCapture(const key_type& key);
  bool cached(const key_type& key) const;
  void clear();

private:
  std::unordered_map<key_type, cudaGraphExec_t> graph_cache_;
  cudaStream_t graph_stream_;
};

} // namespace cudecomp

#endif // CUDECOMP_GRAPH_H
