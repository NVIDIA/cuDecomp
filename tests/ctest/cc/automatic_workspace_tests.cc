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
constexpr std::array<int32_t, 3> kHaloExtents{0, 1, 0};
constexpr std::array<bool, 3> kHaloPeriods{true, true, true};

cudecompGridDescConfig_t makeConfig(cudecompTransposeCommBackend_t transpose_backend,
                                    cudecompHaloCommBackend_t halo_backend) {
  cudecompGridDescConfig_t config;
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));
  std::copy(kGdims.begin(), kGdims.end(), config.gdims);
  std::copy(kPdims.begin(), kPdims.end(), config.pdims);
  config.transpose_comm_backend = transpose_backend;
  config.halo_comm_backend = halo_backend;
  return config;
}

} // namespace

TEST(AutomaticWorkspaceTest, GrowsAcrossHaloAndDtypeRequirements) {
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "automatic workspace growth test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);

  auto config = makeConfig(CUDECOMP_TRANSPOSE_COMM_MPI_P2P, CUDECOMP_HALO_COMM_MPI);
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);

  cudecompPencilInfo_t halo_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompGetPencilInfo(handle, grid_desc, &halo_pinfo, 0, kHaloExtents.data(), nullptr));
  float* halo_data = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&halo_data, halo_pinfo.size * sizeof(*halo_data)));
  cudecomp_test::cudaBufferGuard halo_data_guard(halo_data);
  CHECK_CUDA_GLOBAL(world_comm, cudaMemset(halo_data, 0, halo_pinfo.size * sizeof(*halo_data)));

  int64_t halo_workspace_elements = 0;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetHaloWorkspaceSize(handle, grid_desc, 0, kHaloExtents.data(),
                                                                 &halo_workspace_elements));
  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompUpdateHalosX(handle, grid_desc, halo_data, CUDECOMP_WORKSPACE_AUTO, CUDECOMP_FLOAT,
                                             kHaloExtents.data(), kHaloPeriods.data(), 1, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());

  ASSERT_NE(handle->ordinary_workspace.ptr, nullptr);
  const size_t initial_size = handle->ordinary_workspace.size;
  EXPECT_EQ(initial_size, static_cast<size_t>(halo_workspace_elements) * sizeof(float));

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t transpose_data_elements = std::max(x_pinfo.size, y_pinfo.size);
  std::complex<double>* transpose_input = nullptr;
  std::complex<double>* transpose_output = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&transpose_input, transpose_data_elements * sizeof(*transpose_input)));
  cudecomp_test::cudaBufferGuard transpose_input_guard(transpose_input);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&transpose_output, transpose_data_elements * sizeof(*transpose_output)));
  cudecomp_test::cudaBufferGuard transpose_output_guard(transpose_output);
  CHECK_CUDA_GLOBAL(world_comm, cudaMemset(transpose_input, 0, transpose_data_elements * sizeof(*transpose_input)));

  int64_t transpose_workspace_elements = 0;
  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_workspace_elements));
  const size_t required_transpose_size = static_cast<size_t>(transpose_workspace_elements) * sizeof(*transpose_input);
  ASSERT_GT(required_transpose_size, initial_size);

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, transpose_input, transpose_output,
                                                          CUDECOMP_WORKSPACE_AUTO, CUDECOMP_DOUBLE_COMPLEX, nullptr,
                                                          nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());

  ASSERT_NE(handle->ordinary_workspace.ptr, nullptr);
  EXPECT_EQ(handle->ordinary_workspace.size, required_transpose_size);
  void* grown_workspace = handle->ordinary_workspace.ptr;

  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompUpdateHalosX(handle, grid_desc, halo_data, CUDECOMP_WORKSPACE_AUTO, CUDECOMP_FLOAT,
                                             kHaloExtents.data(), kHaloPeriods.data(), 1, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  EXPECT_EQ(handle->ordinary_workspace.ptr, grown_workspace);
  EXPECT_EQ(handle->ordinary_workspace.size, required_transpose_size);
}

TEST(AutomaticWorkspaceTest, RegistersCacheReusedByLaterDescriptor) {
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 19, 0)
  GTEST_SKIP() << "NCCL user buffer registration requires NCCL 2.19 or newer";
#else
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "NCCL workspace registration test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm, true);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);
  ASSERT_TRUE(handle->nccl_enable_ubr);

  auto mpi_config = makeConfig(CUDECOMP_TRANSPOSE_COMM_MPI_P2P, CUDECOMP_HALO_COMM_MPI);
  cudecompGridDesc_t mpi_grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &mpi_grid_desc, &mpi_config, nullptr));
  cudecomp_test::gridDescGuard mpi_grid_desc_guard(handle, mpi_grid_desc);

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, mpi_grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, mpi_grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t data_elements = std::max(x_pinfo.size, y_pinfo.size);
  float* input = nullptr;
  float* output = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&input, data_elements * sizeof(*input)));
  cudecomp_test::cudaBufferGuard input_guard(input);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&output, data_elements * sizeof(*output)));
  cudecomp_test::cudaBufferGuard output_guard(output);
  CHECK_CUDA_GLOBAL(world_comm, cudaMemset(input, 0, data_elements * sizeof(*input)));

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, mpi_grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  void* cached_workspace = handle->ordinary_workspace.ptr;
  ASSERT_NE(cached_workspace, nullptr);
  EXPECT_EQ(handle->nccl_ubr_handles.count(cached_workspace), 0);

  auto nccl_config = makeConfig(CUDECOMP_TRANSPOSE_COMM_NCCL, CUDECOMP_HALO_COMM_MPI);
  cudecompGridDesc_t nccl_grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &nccl_grid_desc, &nccl_config, nullptr));
  cudecomp_test::gridDescGuard nccl_grid_desc_guard(handle, nccl_grid_desc);

  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompTransposeXToY(handle, nccl_grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                                              CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  EXPECT_EQ(handle->ordinary_workspace.ptr, cached_workspace);

  auto entry = handle->nccl_ubr_handles.find(cached_workspace);
  ASSERT_NE(entry, handle->nccl_ubr_handles.end());
  auto registered_with = [&](const cudecomp::ncclComm& comm) {
    return !comm || std::any_of(entry->second.begin(), entry->second.end(),
                                [&](const auto& registration) { return registration.first.get() == comm.get(); });
  };
  EXPECT_TRUE(registered_with(nccl_grid_desc->nccl_comm));
  EXPECT_TRUE(registered_with(nccl_grid_desc->nccl_local_comm));
#endif
}

#ifdef ENABLE_NVSHMEM
TEST(AutomaticWorkspaceTest, KeepsAllocationDomainsSeparate) {
  const auto world_comm = cudecomp_test::MpiTestComm::world();
  if (world_comm.size() != 4) { GTEST_SKIP() << "NVSHMEM allocation-domain test requires exactly four ranks"; }

  const auto setup_decision = cudecomp_test::initializeGpuForTest(world_comm);
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  cudecompHandle_t handle = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompInit(&handle, world_comm.mpiComm()));
  cudecomp_test::cudecompHandleGuard handle_guard(handle);

  auto config = makeConfig(CUDECOMP_TRANSPOSE_COMM_MPI_P2P, CUDECOMP_HALO_COMM_NVSHMEM);
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);

  cudecompPencilInfo_t x_pinfo;
  cudecompPencilInfo_t y_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &x_pinfo, 0, nullptr, nullptr));
  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompGetPencilInfo(handle, grid_desc, &y_pinfo, 1, nullptr, nullptr));
  const int64_t transpose_data_elements = std::max(x_pinfo.size, y_pinfo.size);
  float* x_data = nullptr;
  float* y_data = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&x_data, transpose_data_elements * sizeof(*x_data)));
  cudecomp_test::cudaBufferGuard x_data_guard(x_data);
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&y_data, transpose_data_elements * sizeof(*y_data)));
  cudecomp_test::cudaBufferGuard y_data_guard(y_data);
  CHECK_CUDA_GLOBAL(world_comm, cudaMemset(x_data, 0, transpose_data_elements * sizeof(*x_data)));

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeXToY(handle, grid_desc, x_data, y_data, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  void* ordinary_workspace = handle->ordinary_workspace.ptr;
  ASSERT_NE(ordinary_workspace, nullptr);
  EXPECT_EQ(handle->nvshmem_workspace.ptr, nullptr);

  cudecompPencilInfo_t halo_pinfo;
  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompGetPencilInfo(handle, grid_desc, &halo_pinfo, 0, kHaloExtents.data(), nullptr));
  float* halo_data = nullptr;
  CHECK_CUDA_GLOBAL(world_comm, cudaMalloc(&halo_data, halo_pinfo.size * sizeof(*halo_data)));
  cudecomp_test::cudaBufferGuard halo_data_guard(halo_data);
  CHECK_CUDA_GLOBAL(world_comm, cudaMemset(halo_data, 0, halo_pinfo.size * sizeof(*halo_data)));

  CHECK_CUDECOMP_GLOBAL(world_comm,
                        cudecompUpdateHalosX(handle, grid_desc, halo_data, CUDECOMP_WORKSPACE_AUTO, CUDECOMP_FLOAT,
                                             kHaloExtents.data(), kHaloPeriods.data(), 1, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  void* nvshmem_workspace = handle->nvshmem_workspace.ptr;
  ASSERT_NE(nvshmem_workspace, nullptr);
  EXPECT_EQ(handle->ordinary_workspace.ptr, ordinary_workspace);
  EXPECT_NE(nvshmem_workspace, ordinary_workspace);

  CHECK_CUDECOMP_GLOBAL(world_comm, cudecompTransposeYToX(handle, grid_desc, y_data, x_data, CUDECOMP_WORKSPACE_AUTO,
                                                          CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, nullptr));
  CHECK_CUDA_GLOBAL(world_comm, cudaDeviceSynchronize());
  EXPECT_EQ(handle->ordinary_workspace.ptr, ordinary_workspace);
  EXPECT_EQ(handle->nvshmem_workspace.ptr, nvshmem_workspace);
}

#endif
