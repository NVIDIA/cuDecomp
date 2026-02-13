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

#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/graph.h"
#include "internal/hashes.h"

namespace cudecomp {

graphCache::graphCache() { CHECK_CUDA(hipStreamCreateWithFlags(&graph_stream_, hipStreamNonBlocking)); }

graphCache::~graphCache() {
  CHECK_CUDA(hipStreamDestroy(graph_stream_));
  this->clear();
}

void graphCache::replay(const graphCache::key_type& key, hipStream_t stream) const {
  CHECK_CUDA(hipGraphLaunch(graph_cache_.at(key), stream));
}

hipStream_t graphCache::startCapture(const graphCache::key_type& key, hipStream_t stream) const {
  CHECK_CUDA(hipStreamBeginCapture(graph_stream_, hipStreamCaptureModeGlobal));
  return graph_stream_;
}

void graphCache::endCapture(const graphCache::key_type& key) {
  hipGraph_t graph;
  hipGraphExec_t graph_exec;
  CHECK_CUDA(hipStreamEndCapture(graph_stream_, &graph));
  CHECK_CUDA(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(hipGraphDestroy(graph));

  graph_cache_[key] = graph_exec;
}

bool graphCache::cached(const graphCache::key_type& key) const { return graph_cache_.count(key) > 0; }

void graphCache::clear() {
  for (auto& entry : graph_cache_) {
    CHECK_CUDA(hipGraphExecDestroy(entry.second));
  }

  graph_cache_.clear();
}

} // namespace cudecomp
