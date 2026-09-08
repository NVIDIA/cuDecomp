/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>

#include <mpi.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudecomp.h"
#include "internal/common.h"

#include "gpu_test_utils.h"
#include "mpi_test_utils.h"
#include "test_utils.h"

namespace {

constexpr std::array<int32_t, 3> kGdims{9, 10, 11};
constexpr std::array<int32_t, 2> kPdims{2, 2};

cudecompGridDescConfig_t makeConfig() {
  cudecompGridDescConfig_t config;
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));
  std::copy(kGdims.begin(), kGdims.end(), config.gdims);
  std::copy(kPdims.begin(), kPdims.end(), config.pdims);
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NCCL;
  config.halo_comm_backend = CUDECOMP_HALO_COMM_NCCL;
  return config;
}

} // namespace

TEST(CudaGraphContractTest, AutomaticWorkspaceCaptureRequiresWarmup) {
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "automatic workspace capture test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm, true);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);

  auto config = makeConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t data_elements = std::max(x_pinfo.size, y_pinfo.size);
  float* input = nullptr;
  float* output = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&input, data_elements * sizeof(*input)));
  cudecomp_test::cudaBufferGuard input_guard(input);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&output, data_elements * sizeof(*output)));
  cudecomp_test::cudaBufferGuard output_guard(output);

  cudaStream_t stream = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED,
            cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO, CUDECOMP_FLOAT, nullptr,
                                  nullptr, nullptr, nullptr, stream));
  cudaGraph_t graph = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamEndCapture(stream, &graph));
  if (graph) { CHECK_CUDA_GLOBAL(world_comm, cudaGraphDestroy(graph)); }
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamDestroy(stream));
  EXPECT_FALSE(handle->ordinary_workspace.capture_frozen);
}

TEST(CudaGraphContractTest, InternalGraphsRejectExternalCapture) {
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "CUDA Graph mode conflict test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm, true);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);

  auto config = makeConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t data_elements = std::max(x_pinfo.size, y_pinfo.size);
  float* input = nullptr;
  float* output = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&input, data_elements * sizeof(*input)));
  cudecomp_test::cudaBufferGuard input_guard(input);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&output, data_elements * sizeof(*output)));
  cudecomp_test::cudaBufferGuard output_guard(output);

  int64_t workspace_elements = 0;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetTransposeWorkspaceSize(handle, grid_desc, &workspace_elements));
  void* explicit_workspace = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompMalloc(handle, grid_desc, &explicit_workspace, workspace_elements * sizeof(float)));
  cudecomp_test::cudecompBufferGuard workspace_guard(handle, grid_desc, explicit_workspace);

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  handle->cuda_graphs_enable = true;

  for (void* workspace : {explicit_workspace, CUDECOMP_WORKSPACE_AUTO}) {
    cudaStream_t stream = nullptr;
    CHECK_CUDA_GLOBAL(world_comm, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA_GLOBAL(world_comm, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED,
              cudecompTransposeXToY(handle, grid_desc, input, output, workspace, CUDECOMP_FLOAT, nullptr, nullptr,
                                    nullptr, nullptr, stream));
    cudaGraph_t graph = nullptr;
    CHECK_CUDA_GLOBAL(world_comm, cudaStreamEndCapture(stream, &graph));
    if (graph) { CHECK_CUDA_GLOBAL(world_comm, cudaGraphDestroy(graph)); }
    CHECK_CUDA_GLOBAL(world_comm, cudaStreamDestroy(stream));
  }
}

TEST(CudaGraphContractTest, CapturedAutomaticWorkspaceCannotGrow) {
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "automatic workspace freeze test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm, true);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);

  auto config = makeConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t data_elements = std::max(x_pinfo.size, y_pinfo.size);
  std::complex<double>* input = nullptr;
  std::complex<double>* output = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&input, data_elements * sizeof(*input)));
  cudecomp_test::cudaBufferGuard input_guard(input);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&output, data_elements * sizeof(*output)));
  cudecomp_test::cudaBufferGuard output_guard(output);

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  void* captured_workspace = handle->ordinary_workspace.ptr;
  const size_t captured_size = handle->ordinary_workspace.size;

  cudaStream_t stream = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, stream));
  cudaGraph_t graph = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamEndCapture(stream, &graph));
  ASSERT_NE(graph, nullptr);
  CHECK_CUDA_GLOBAL(world_comm, cudaGraphDestroy(graph));
  CHECK_CUDA_GLOBAL(world_comm, cudaStreamDestroy(stream));
  EXPECT_TRUE(handle->ordinary_workspace.capture_frozen);

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED,
            cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO, CUDECOMP_DOUBLE_COMPLEX,
                                  nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(handle->ordinary_workspace.ptr, captured_workspace);
  EXPECT_EQ(handle->ordinary_workspace.size, captured_size);
}
