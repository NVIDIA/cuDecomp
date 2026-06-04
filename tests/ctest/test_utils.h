/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDECOMP_TEST_UTILS_H
#define CUDECOMP_TEST_UTILS_H

#include <mpi.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudecomp.h"
#include "mpi_test_utils.h"

namespace cudecomp_test {

class cudecompHandleGuard {
public:
  explicit cudecompHandleGuard(cudecompHandle_t handle = nullptr);
  cudecompHandleGuard(const cudecompHandleGuard&) = delete;
  cudecompHandleGuard& operator=(const cudecompHandleGuard&) = delete;
  ~cudecompHandleGuard();

private:
  cudecompHandle_t handle_ = nullptr;
};

class gridDescGuard {
public:
  gridDescGuard(cudecompHandle_t handle, cudecompGridDesc_t grid_desc);
  gridDescGuard(const gridDescGuard&) = delete;
  gridDescGuard& operator=(const gridDescGuard&) = delete;
  ~gridDescGuard();

private:
  cudecompHandle_t handle_ = nullptr;
  cudecompGridDesc_t grid_desc_ = nullptr;
};

class cudaBufferGuard {
public:
  explicit cudaBufferGuard(void* ptr = nullptr);
  cudaBufferGuard(const cudaBufferGuard&) = delete;
  cudaBufferGuard& operator=(const cudaBufferGuard&) = delete;
  ~cudaBufferGuard();

  void reset(void* ptr = nullptr);

private:
  void* ptr_ = nullptr;
};

class cudecompBufferGuard {
public:
  cudecompBufferGuard(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* ptr);
  cudecompBufferGuard(const cudecompBufferGuard&) = delete;
  cudecompBufferGuard& operator=(const cudecompBufferGuard&) = delete;
  ~cudecompBufferGuard();

  void release() noexcept;

private:
  cudecompHandle_t handle_ = nullptr;
  cudecompGridDesc_t grid_desc_ = nullptr;
  void* ptr_ = nullptr;
};

testing::AssertionResult checkCudaGlobal(const MpiTestComm& comm, cudaError_t result, const char* file, int line);
testing::AssertionResult checkCudecompGlobal(const MpiTestComm& comm, cudecompResult_t result, const char* file,
                                             int line);
testing::AssertionResult checkMpiGlobal(const MpiTestComm& comm, int result, const char* file, int line);

} // namespace cudecomp_test

#define CHECK_CUDA_GLOBAL(comm, call) ASSERT_TRUE(::cudecomp_test::checkCudaGlobal((comm), (call), __FILE__, __LINE__))
#define CHECK_CUDECOMP_GLOBAL(comm, call)                                                                              \
  ASSERT_TRUE(::cudecomp_test::checkCudecompGlobal((comm), (call), __FILE__, __LINE__))
#define CHECK_MPI_GLOBAL(comm, call) ASSERT_TRUE(::cudecomp_test::checkMpiGlobal((comm), (call), __FILE__, __LINE__))

#endif
