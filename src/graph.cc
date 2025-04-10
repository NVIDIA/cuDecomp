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

#include <tuple>
#include <unordered_map>

#include <cuda_runtime.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/graph.h"
#include "internal/hashes.h"

namespace cudecomp {

graphCache::graphCache() {
  CHECK_CUDA(cudaStreamCreateWithFlags(&graph_stream_, cudaStreamNonBlocking));
}

graphCache::~graphCache() {
  CHECK_CUDA(cudaStreamDestroy(graph_stream_));
}

void graphCache::replay(const graphCache::key_type& key, cudaStream_t stream) const {
  CHECK_CUDA(cudaGraphLaunch(graph_cache_.at(key), stream));
}

cudaStream_t graphCache::startCapture(const graphCache::key_type& key, cudaStream_t stream) const {
  CHECK_CUDA(cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeGlobal));
  return graph_stream_;
}

void graphCache::endCapture(const graphCache::key_type& key){
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CHECK_CUDA(cudaStreamEndCapture(graph_stream_, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaGraphDestroy(graph));

  graph_cache_[key] = graph_exec;
}

bool graphCache::cached(const graphCache::key_type& key) const { return graph_cache_.count(key) > 0; }

} // namespace cudecomp

