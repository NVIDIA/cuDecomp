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
#include <utility>

#include <cuda_runtime.h>

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/graph.h"
#include "internal/hashes.h"

namespace cudecomp {

graphCache::~graphCache() noexcept { clear(); }

void graphCache::replay(const graphCache::key_type& key, cudaStream_t stream) const {
  CHECK_CUDA(cudaGraphLaunch(graph_cache_.at(key).get(), stream));
}

cudaStream_t graphCache::startCapture(const graphCache::key_type& key, cudaStream_t stream) const {
  CHECK_CUDA(cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeGlobal));
  return graph_stream_;
}

void graphCache::endCapture(const graphCache::key_type& key) {
  cudaGraph graph;
  cudaGraphExec graph_exec;
  CHECK_CUDA(cudaStreamEndCapture(graph_stream_, graph.put()));
  CHECK_CUDA(cudaGraphInstantiate(graph_exec.put(), graph.get(), nullptr, nullptr, 0));

  graph_cache_[key] = std::move(graph_exec);
}

bool graphCache::cached(const graphCache::key_type& key) const { return graph_cache_.count(key) > 0; }

void graphCache::clear() noexcept { graph_cache_.clear(); }

} // namespace cudecomp
