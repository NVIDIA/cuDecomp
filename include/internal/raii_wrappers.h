/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDECOMP_RAII_WRAPPERS_H
#define CUDECOMP_RAII_WRAPPERS_H

#include <utility>

#include <cuda_runtime.h>

#include "internal/checks.h"

namespace cudecomp {

template <typename T, auto destroy_fn> class uniqueHandle {
public:
  uniqueHandle() = default;
  explicit uniqueHandle(T handle) : handle_(handle) {}
  ~uniqueHandle() noexcept { resetNoThrow(); }

  uniqueHandle(const uniqueHandle&) = delete;
  uniqueHandle& operator=(const uniqueHandle&) = delete;

  uniqueHandle(uniqueHandle&& other) noexcept : handle_(std::exchange(other.handle_, T{})) {}

  uniqueHandle& operator=(uniqueHandle&& other) noexcept {
    if (this != &other) {
      resetNoThrow();
      handle_ = std::exchange(other.handle_, T{});
    }
    return *this;
  }

  T get() const noexcept { return handle_; }
  T* put() noexcept {
    resetNoThrow();
    return &handle_;
  }
  T release() noexcept { return std::exchange(handle_, T{}); }
  operator T() const noexcept { return handle_; }

private:
  void resetNoThrow() noexcept {
    if (handle_) {
      destroy_fn(handle_);
      handle_ = T{};
    }
  }

  T handle_{};
};

template <unsigned int flags> class cudaEventBase {
public:
  cudaEventBase() { CHECK_CUDA(cudaEventCreateWithFlags(&event_, flags)); }
  ~cudaEventBase() noexcept { resetNoThrow(); }

  cudaEventBase(const cudaEventBase&) = delete;
  cudaEventBase& operator=(const cudaEventBase&) = delete;

  cudaEventBase(cudaEventBase&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

  cudaEventBase& operator=(cudaEventBase&& other) noexcept {
    if (this != &other) {
      resetNoThrow();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  cudaEvent_t get() const noexcept { return event_; }
  operator cudaEvent_t() const noexcept { return event_; }

private:
  void resetNoThrow() noexcept {
    if (event_) {
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
  }

  cudaEvent_t event_ = nullptr;
};

using cudaEvent = cudaEventBase<cudaEventDisableTiming>;
using cudaEventTimed = cudaEventBase<cudaEventDefault>;

template <unsigned int flags> class cudaStreamBase {
public:
  cudaStreamBase() {
    int greatest_priority;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
    CHECK_CUDA(cudaStreamCreateWithPriority(&stream_, flags, greatest_priority));
  }
  ~cudaStreamBase() noexcept { resetNoThrow(); }

  cudaStreamBase(const cudaStreamBase&) = delete;
  cudaStreamBase& operator=(const cudaStreamBase&) = delete;

  cudaStreamBase(cudaStreamBase&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

  cudaStreamBase& operator=(cudaStreamBase&& other) noexcept {
    if (this != &other) {
      resetNoThrow();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t get() const noexcept { return stream_; }
  operator cudaStream_t() const noexcept { return stream_; }

private:
  void resetNoThrow() noexcept {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  cudaStream_t stream_ = nullptr;
};

using cudaStream = cudaStreamBase<cudaStreamNonBlocking>;

} // namespace cudecomp

#endif // CUDECOMP_RAII_WRAPPERS_H
