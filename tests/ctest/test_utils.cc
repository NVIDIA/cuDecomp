/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <string>

namespace cudecomp_test {
namespace {

std::string sourceLocation(const char* file, int line) { return std::string(file) + ":" + std::to_string(line); }

testing::AssertionResult checkLocalFailureGlobal(const MpiTestComm& comm, bool local_success, const char* error_kind,
                                                 const std::string& local_error, const char* file, int line) {
  const int local_failure = local_success ? 0 : 1;
  int global_failure = 0;
  const int reduce_result = MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT, MPI_MAX, comm.mpiComm());
  if (reduce_result != MPI_SUCCESS) {
    return testing::AssertionFailure() << "MPI_Allreduce failed in global check at " << sourceLocation(file, line);
  }

  if (global_failure == 0) return testing::AssertionSuccess();
  if (!local_success) {
    return testing::AssertionFailure() << error_kind << " failed at " << sourceLocation(file, line) << " with "
                                       << local_error;
  }
  return testing::AssertionFailure() << error_kind << " failed at " << sourceLocation(file, line) << " on another rank";
}

} // namespace

cudecompHandleGuard::cudecompHandleGuard(cudecompHandle_t handle) : handle_(handle) {}

cudecompHandleGuard::~cudecompHandleGuard() {
  if (handle_) { (void)cudecompFinalize(handle_); }
}

gridDescGuard::gridDescGuard(cudecompHandle_t handle, cudecompGridDesc_t grid_desc)
    : handle_(handle), grid_desc_(grid_desc) {}

gridDescGuard::~gridDescGuard() {
  if (grid_desc_) { (void)cudecompGridDescDestroy(handle_, grid_desc_); }
}

cudaBufferGuard::cudaBufferGuard(void* ptr) : ptr_(ptr) {}

cudaBufferGuard::~cudaBufferGuard() { reset(); }

void cudaBufferGuard::reset(void* ptr) {
  if (ptr_) { (void)cudaFree(ptr_); }
  ptr_ = ptr;
}

cudecompBufferGuard::cudecompBufferGuard(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* ptr)
    : handle_(handle), grid_desc_(grid_desc), ptr_(ptr) {}

cudecompBufferGuard::~cudecompBufferGuard() {
  if (ptr_) { (void)cudecompFree(handle_, grid_desc_, ptr_); }
}

void cudecompBufferGuard::release() noexcept { ptr_ = nullptr; }

testing::AssertionResult checkCudaGlobal(const MpiTestComm& comm, cudaError_t result, const char* file, int line) {
  return checkLocalFailureGlobal(comm, result == cudaSuccess, "CUDA", cudaGetErrorString(result), file, line);
}

testing::AssertionResult checkCudecompGlobal(const MpiTestComm& comm, cudecompResult_t result, const char* file,
                                             int line) {
  return checkLocalFailureGlobal(comm, result == CUDECOMP_RESULT_SUCCESS, "cuDecomp", std::to_string(result), file,
                                 line);
}

testing::AssertionResult checkMpiGlobal(const MpiTestComm& comm, int result, const char* file, int line) {
  char error_string[MPI_MAX_ERROR_STRING] = {};
  int error_string_len = 0;
  if (result != MPI_SUCCESS) { MPI_Error_string(result, error_string, &error_string_len); }
  return checkLocalFailureGlobal(comm, result == MPI_SUCCESS, "MPI", error_string, file, line);
}

} // namespace cudecomp_test
