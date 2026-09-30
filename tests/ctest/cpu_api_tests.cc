/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>

#include "internal/common.h"
#include "test_utils.h"

TEST(CpuInitTest, HostAllocationAndUnsupportedFeatures) {
  const auto comm = cudecomp_test::MpiTestComm::world();
  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(comm, cudecompInit(&handle, comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);
  EXPECT_FALSE(handle->nvml_initialized);
  EXPECT_FALSE(handle->performance_report_enable);

  cudecompGridDescConfig_t config;
  ASSERT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));
  config.gdims[0] = config.gdims[1] = config.gdims[2] = 8;
  config.pdims[0] = comm.size();
  config.pdims[1] = 1;
  cudecompGridDesc_t grid = nullptr;
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NCCL;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, nullptr));
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
  config.halo_comm_backend = CUDECOMP_HALO_COMM_NCCL;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, nullptr));
  config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI;
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NVSHMEM;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, nullptr));
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
  config.halo_comm_backend = CUDECOMP_HALO_COMM_NVSHMEM;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, nullptr));
  config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI;

  cudecompGridDescAutotuneOptions_t tune;
  ASSERT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescAutotuneOptionsSetDefaults(&tune));
  tune.autotune_transpose_backend = true;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, &tune));
  tune.autotune_transpose_backend = false;
  tune.autotune_halo_backend = true;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, &tune));
  tune.autotune_halo_backend = false;
  config.pdims[0] = config.pdims[1] = 0;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED, cudecompGridDescCreate(handle, &grid, &config, &tune));
  config.pdims[0] = comm.size();
  config.pdims[1] = 1;

  CHECK_CUDECOMP_GLOBAL(comm, cudecompGridDescCreate(handle, &grid, &config, &tune));
  cudecomp_test::gridDescGuard grid_guard(handle, grid);
  void* buffer = nullptr;
  CHECK_CUDECOMP_GLOBAL(comm, cudecompMalloc(handle, grid, &buffer, 512 * sizeof(float)));
  cudecomp_test::cudecompBufferGuard buffer_guard(handle, grid, buffer);
  std::fill_n(static_cast<float*>(buffer), 512, 7.0f);
  EXPECT_EQ(static_cast<float*>(buffer)[511], 7.0f);
  const int32_t extents[3] = {0, 0, 0};
  const bool periods[3] = {false, false, false};
  for (auto halo : {cudecompUpdateHalosX, cudecompUpdateHalosY, cudecompUpdateHalosZ}) {
    EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, halo(handle, grid, buffer, nullptr, CUDECOMP_FLOAT, extents, periods, 0, nullptr,
                                            reinterpret_cast<cudaStream_t>(1)));
  }
  for (auto transpose : {cudecompTransposeXToY, cudecompTransposeYToZ, cudecompTransposeZToY, cudecompTransposeYToX}) {
    EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, transpose(handle, grid, buffer, buffer, nullptr, CUDECOMP_FLOAT, nullptr,
                                                 nullptr, nullptr, nullptr, reinterpret_cast<cudaStream_t>(1)));
  }
  EXPECT_TRUE(std::all_of(static_cast<float*>(buffer), static_cast<float*>(buffer) + 512,
                          [](float value) { return value == 7.0f; }));
}
