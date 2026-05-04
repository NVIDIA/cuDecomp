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

#ifndef CUDECOMP_CUDA_EVENT_H
#define CUDECOMP_CUDA_EVENT_H

#include <utility>

#include <cuda_runtime.h>

#include "internal/checks.h"

namespace cudecomp {

class cudaEvent {
public:
  cudaEvent() = default;
  ~cudaEvent() noexcept { resetNoThrow(); }

  cudaEvent(const cudaEvent&) = delete;
  cudaEvent& operator=(const cudaEvent&) = delete;

  cudaEvent(cudaEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

  cudaEvent& operator=(cudaEvent&& other) noexcept {
    if (this != &other) {
      resetNoThrow();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  void create() {
    reset();
    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreate(&event));
    event_ = event;
  }

  void createWithFlags(unsigned int flags) {
    reset();
    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&event, flags));
    event_ = event;
  }

  void reset() {
    if (event_) {
      CHECK_CUDA(cudaEventDestroy(event_));
      event_ = nullptr;
    }
  }

  void resetNoThrow() noexcept {
    if (event_) {
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
  }

  cudaEvent_t get() const noexcept { return event_; }
  operator cudaEvent_t() const noexcept { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

} // namespace cudecomp

#endif // CUDECOMP_CUDA_EVENT_H
