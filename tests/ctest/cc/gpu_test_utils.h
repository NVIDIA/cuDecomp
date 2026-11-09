/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDECOMP_TEST_GPU_TEST_UTILS_H
#define CUDECOMP_TEST_GPU_TEST_UTILS_H

#include <string>

#include "mpi_test_utils.h"

namespace cudecomp_test {

struct TestSetupDecision {
  bool skip = false;
  bool fail = false;
  std::string reason;
};

struct GpuTestRuntime {
  bool mps_active = false;
  int nccl_version = 0;
  bool nccl_multi_rank_gpu_enabled = false;
};

void initializeGpuTestRuntime();
const GpuTestRuntime& gpuTestRuntime();
bool mpsActive();
int ncclVersion();
bool ncclSupportsMultiRankPerGpu();
bool ncclMultiRankGpuEnabled();
TestSetupDecision checkGpuTestRequirements(const MpiTestComm& comm);
TestSetupDecision initializeGpuForTest(const MpiTestComm& comm, bool check_nccl = false);

} // namespace cudecomp_test

#endif
