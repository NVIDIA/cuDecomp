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

#ifndef CUDECOMP_GRAPH_H
#define CUDECOMP_GRAPH_H

#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>

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
  void replay(const key_type& key, hipStream_t stream) const;
  hipStream_t startCapture(const key_type& key, hipStream_t stream) const;
  void endCapture(const key_type& key);
  bool cached(const key_type& key) const;
  void clear();

private:
  std::unordered_map<key_type, hipGraphExec_t> graph_cache_;
  hipStream_t graph_stream_;
};

} // namespace cudecomp

#endif // CUDECOMP_GRAPH_H
