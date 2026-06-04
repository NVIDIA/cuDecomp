/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include <mpi.h>

#include <gtest/gtest.h>

#include "backend_test_context.h"
#include "gpu_test_utils.h"
#include "mpi_test_utils.h"

namespace {

class RankFailurePrinter : public ::testing::EmptyTestEventListener {
public:
  explicit RankFailurePrinter(int rank) : rank_(rank) {}

  void OnTestPartResult(const ::testing::TestPartResult& result) override {
    if (!result.failed()) return;

    std::cerr << "[rank " << rank_ << "] " << result.file_name() << ":" << result.line_number() << ": "
              << result.summary() << "\n";
  }

private:
  int rank_;
};

bool gpuRequirementsSatisfied(const cudecomp_test::MpiTestComm& world_comm) {
  const auto setup_decision = cudecomp_test::checkGpuTestRequirements(world_comm);
  if (!setup_decision.fail) return true;

  std::cerr << "[rank " << world_comm.rank() << "] GPU test requirements are not satisfied: " << setup_decision.reason
            << "\n";
  return false;
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int global_failure = 0;

  {
    const auto world_comm = cudecomp_test::MpiTestComm::world();
    cudecomp_test::initializeGpuTestRuntime();

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new RankFailurePrinter(world_comm.rank()));

    if (world_comm.rank() == 0) { std::cout << "[cuDecomp test] MPI ranks: " << world_comm.size() << "\n"; }

    const int local_result = gpuRequirementsSatisfied(world_comm) ? RUN_ALL_TESTS() : 1;
    cudecomp_test::resetSharedBackendTestContext();
    const int local_failure = local_result == 0 ? 0 : 1;
    MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT, MPI_MAX, world_comm.mpiComm());
  }

  MPI_Finalize();
  return global_failure == 0 ? 0 : 1;
}
