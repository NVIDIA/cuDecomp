/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDECOMP_TEST_BACKEND_TEST_CONTEXT_H
#define CUDECOMP_TEST_BACKEND_TEST_CONTEXT_H

#include <memory>

#include <gtest/gtest.h>

#include "cudecomp.h"
#include "gpu_test_utils.h"
#include "mpi_test_utils.h"
#include "test_utils.h"

namespace cudecomp_test {

class BackendTestContext {
public:
  testing::AssertionResult initialize(const MpiTestComm& world_comm, int active_ranks, const char* backend_label,
                                      bool check_nccl, const cudecompGridDescConfig_t& config,
                                      TestSetupDecision* setup_decision);

  const MpiTestComm& comm() const { return *active_comm_; }
  cudecompHandle_t handle() const { return handle_; }

private:
  MpiTestComm local_active_comm_;
  cudecompHandle_t local_handle_ = nullptr;
  std::unique_ptr<cudecompHandleGuard> local_handle_guard_;
  const MpiTestComm* active_comm_ = nullptr;
  cudecompHandle_t handle_ = nullptr;
};

void resetSharedBackendTestContext();

} // namespace cudecomp_test

#endif
