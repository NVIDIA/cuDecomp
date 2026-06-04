/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <memory>

#include <mpi.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudecomp.h"

#include "gpu_test_utils.h"
#include "mpi_test_utils.h"
#include "test_utils.h"

namespace {

constexpr int kApiTestRanks = 4;
constexpr std::array<int32_t, 3> kGdims{9, 10, 11};
constexpr std::array<int32_t, 3> kGdimsDist{8, 9, 10};
constexpr std::array<int32_t, 2> kPdims{2, 2};
constexpr std::array<int32_t, 3> kHaloExtents{1, 2, 1};
constexpr std::array<int32_t, 3> kPadding{1, 0, 2};
constexpr std::array<bool, 3> kHaloPeriods{false, true, false};

struct ExpectedPencilInfo {
  std::array<int32_t, 3> shape;
  std::array<int32_t, 3> lo;
  std::array<int32_t, 3> hi;
  std::array<int32_t, 3> order;
  std::array<int32_t, 3> halo_extents;
  std::array<int32_t, 3> padding;
  int64_t size;
};

constexpr ExpectedPencilInfo kExpectedDefaultPencilInfo[3][kApiTestRanks] = {
    {
        {{12, 9, 10}, {0, 0, 0}, {8, 4, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{12, 9, 9}, {0, 0, 6}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 972},
        {{12, 9, 10}, {0, 5, 0}, {8, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{12, 9, 9}, {0, 5, 6}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 972},
    },
    {
        {{8, 14, 10}, {0, 0, 0}, {4, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1120},
        {{8, 14, 9}, {0, 0, 6}, {4, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1008},
        {{7, 14, 10}, {5, 0, 0}, {8, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 980},
        {{7, 14, 9}, {5, 0, 6}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 882},
    },
    {
        {{8, 9, 15}, {0, 0, 0}, {4, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{8, 9, 15}, {0, 5, 0}, {4, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{7, 9, 15}, {5, 0, 0}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
        {{7, 9, 15}, {5, 5, 0}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
    },
};

constexpr ExpectedPencilInfo kExpectedColumnMajorPencilInfo[3][kApiTestRanks] = {
    {
        {{12, 9, 10}, {0, 0, 0}, {8, 4, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{12, 9, 10}, {0, 5, 0}, {8, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{12, 9, 9}, {0, 0, 6}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 972},
        {{12, 9, 9}, {0, 5, 6}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 972},
    },
    {
        {{8, 14, 10}, {0, 0, 0}, {4, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 1120},
        {{7, 14, 10}, {5, 0, 0}, {8, 9, 5}, {0, 1, 2}, kHaloExtents, kPadding, 980},
        {{8, 14, 9}, {0, 0, 6}, {4, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1008},
        {{7, 14, 9}, {5, 0, 6}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 882},
    },
    {
        {{8, 9, 15}, {0, 0, 0}, {4, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{7, 9, 15}, {5, 0, 0}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
        {{8, 9, 15}, {0, 5, 0}, {4, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{7, 9, 15}, {5, 5, 0}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
    },
};

constexpr ExpectedPencilInfo kExpectedGdimsDistPencilInfo[3][kApiTestRanks] = {
    {
        {{12, 9, 9}, {0, 0, 0}, {8, 4, 4}, {0, 1, 2}, kHaloExtents, kPadding, 972},
        {{12, 9, 10}, {0, 0, 5}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{12, 9, 9}, {0, 5, 0}, {8, 9, 4}, {0, 1, 2}, kHaloExtents, kPadding, 972},
        {{12, 9, 10}, {0, 5, 5}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
    },
    {
        {{7, 14, 9}, {0, 0, 0}, {3, 9, 4}, {0, 1, 2}, kHaloExtents, kPadding, 882},
        {{7, 14, 10}, {0, 0, 5}, {3, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 980},
        {{8, 14, 9}, {4, 0, 0}, {8, 9, 4}, {0, 1, 2}, kHaloExtents, kPadding, 1008},
        {{8, 14, 10}, {4, 0, 5}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1120},
    },
    {
        {{7, 9, 15}, {0, 0, 0}, {3, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
        {{7, 9, 15}, {0, 5, 0}, {3, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 945},
        {{8, 9, 15}, {4, 0, 0}, {8, 4, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
        {{8, 9, 15}, {4, 5, 0}, {8, 9, 10}, {0, 1, 2}, kHaloExtents, kPadding, 1080},
    },
};

void setDistributedConfig(cudecompGridDescConfig_t& config) {
  config.gdims[0] = kGdims[0];
  config.gdims[1] = kGdims[1];
  config.gdims[2] = kGdims[2];
  config.pdims[0] = kPdims[0];
  config.pdims[1] = kPdims[1];
}

void setMemOrder(cudecompGridDescConfig_t& config, const std::array<int32_t, 3>& order) {
  for (int axis = 0; axis < 3; ++axis) {
    for (int i = 0; i < 3; ++i) {
      config.transpose_mem_order[axis][i] = order[i];
    }
  }
}

void setGdimsDist(cudecompGridDescConfig_t& config) {
  config.gdims_dist[0] = kGdimsDist[0];
  config.gdims_dist[1] = kGdimsDist[1];
  config.gdims_dist[2] = kGdimsDist[2];
}

void expectPencilInfoEquals(const cudecompPencilInfo_t& actual, const ExpectedPencilInfo& expected) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(expected.shape[i], actual.shape[i]);
    EXPECT_EQ(expected.lo[i], actual.lo[i]);
    EXPECT_EQ(expected.hi[i], actual.hi[i]);
    EXPECT_EQ(expected.order[i], actual.order[i]);
    EXPECT_EQ(expected.halo_extents[i], actual.halo_extents[i]);
    EXPECT_EQ(expected.padding[i], actual.padding[i]);
  }
  EXPECT_EQ(expected.size, actual.size);
}

void expectGridDescConfigEquals(const cudecompGridDescConfig_t& actual, const cudecompGridDescConfig_t& expected) {
  EXPECT_EQ(expected.rank_order, actual.rank_order);
  EXPECT_EQ(expected.transpose_comm_backend, actual.transpose_comm_backend);
  EXPECT_EQ(expected.halo_comm_backend, actual.halo_comm_backend);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(expected.pdims[i], actual.pdims[i]);
  }
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(expected.gdims[i], actual.gdims[i]);
    EXPECT_EQ(expected.gdims_dist[i], actual.gdims_dist[i]);
    EXPECT_EQ(expected.transpose_axis_contiguous[i], actual.transpose_axis_contiguous[i]);
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(expected.transpose_mem_order[i][j], actual.transpose_mem_order[i][j]);
    }
  }
}

bool isMpiTransposeBackend(cudecompTransposeCommBackend_t backend) {
  return backend == CUDECOMP_TRANSPOSE_COMM_MPI_P2P || backend == CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL ||
         backend == CUDECOMP_TRANSPOSE_COMM_MPI_A2A;
}

bool isMpiHaloBackend(cudecompHaloCommBackend_t backend) {
  return backend == CUDECOMP_HALO_COMM_MPI || backend == CUDECOMP_HALO_COMM_MPI_BLOCKING;
}

void expectShiftedRanks(const cudecomp_test::MpiTestComm& comm, cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                        int axis, int dim, int displacement, bool periodic,
                        const std::array<int32_t, kApiTestRanks>& expected_ranks) {
  int32_t shifted_rank = -2;
  CHECK_CUDECOMP_GLOBAL(comm,
                        cudecompGetShiftedRank(handle, grid_desc, axis, dim, displacement, periodic, &shifted_rank));
  EXPECT_EQ(expected_ranks[comm.rank()], shifted_rank);
}

TEST(ApiGridDescConfigSetDefaultsTest, SetsDocumentedDefaults) {
  cudecompGridDescConfig_t config;
  ASSERT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));

  EXPECT_EQ(CUDECOMP_RANK_ORDER_DEFAULT, config.rank_order);
  EXPECT_EQ(CUDECOMP_TRANSPOSE_COMM_MPI_P2P, config.transpose_comm_backend);
  EXPECT_EQ(CUDECOMP_HALO_COMM_MPI, config.halo_comm_backend);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(0, config.pdims[i]);
  }
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(0, config.gdims[i]);
    EXPECT_EQ(0, config.gdims_dist[i]);
    EXPECT_FALSE(config.transpose_axis_contiguous[i]);
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(-1, config.transpose_mem_order[i][j]);
    }
  }
}

TEST(ApiGridDescConfigSetDefaultsTest, RejectsInvalidArguments) {
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescConfigSetDefaults(nullptr));
}

TEST(ApiGridDescAutotuneOptionsSetDefaultsTest, SetsDocumentedDefaults) {
  cudecompGridDescAutotuneOptions_t options;
  ASSERT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescAutotuneOptionsSetDefaults(&options));

  EXPECT_EQ(3, options.n_warmup_trials);
  EXPECT_EQ(5, options.n_trials);
  EXPECT_EQ(CUDECOMP_AUTOTUNE_GRID_TRANSPOSE, options.grid_mode);
  EXPECT_EQ(CUDECOMP_DOUBLE, options.dtype);
  EXPECT_TRUE(options.allow_uneven_decompositions);
  EXPECT_FALSE(options.disable_nccl_backends);
  EXPECT_FALSE(options.disable_nvshmem_backends);
  EXPECT_EQ(0.0, options.skip_threshold);
  EXPECT_FALSE(options.autotune_transpose_backend);
  EXPECT_FALSE(options.autotune_halo_backend);
  EXPECT_EQ(0, options.halo_axis);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(options.transpose_use_inplace_buffers[i]);
    EXPECT_EQ(1.0, options.transpose_op_weights[i]);
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(0, options.transpose_input_halo_extents[i][j]);
      EXPECT_EQ(0, options.transpose_output_halo_extents[i][j]);
      EXPECT_EQ(0, options.transpose_input_padding[i][j]);
      EXPECT_EQ(0, options.transpose_output_padding[i][j]);
    }
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(0, options.halo_extents[i]);
    EXPECT_FALSE(options.halo_periods[i]);
    EXPECT_EQ(0, options.halo_padding[i]);
  }
}

TEST(ApiGridDescAutotuneOptionsSetDefaultsTest, RejectsInvalidArguments) {
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescAutotuneOptionsSetDefaults(nullptr));
}

TEST(ApiGetDataTypeSizeTest, ReturnsSupportedTypeSizes) {
  int64_t dtype_size = 0;
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));
  EXPECT_EQ(4, dtype_size);
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGetDataTypeSize(CUDECOMP_DOUBLE, &dtype_size));
  EXPECT_EQ(8, dtype_size);
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGetDataTypeSize(CUDECOMP_FLOAT_COMPLEX, &dtype_size));
  EXPECT_EQ(8, dtype_size);
  EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGetDataTypeSize(CUDECOMP_DOUBLE_COMPLEX, &dtype_size));
  EXPECT_EQ(16, dtype_size);
}

TEST(ApiGetDataTypeSizeTest, RejectsInvalidArguments) {
  int64_t dtype_size = 0;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetDataTypeSize(CUDECOMP_FLOAT, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetDataTypeSize(static_cast<cudecompDataType_t>(999), &dtype_size));
}

TEST(ApiTransposeCommBackendToStringTest, ReturnsBackendNames) {
  EXPECT_STREQ("MPI_P2P", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_MPI_P2P));
  EXPECT_STREQ("MPI_P2P (pipelined)", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL));
  EXPECT_STREQ("MPI_A2A", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_MPI_A2A));
  EXPECT_STREQ("NCCL", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_NCCL));
  EXPECT_STREQ("NCCL (pipelined)", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_NCCL_PL));
  EXPECT_STREQ("NVSHMEM", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_NVSHMEM));
  EXPECT_STREQ("NVSHMEM (pipelined)", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL));
  EXPECT_STREQ("NVSHMEM_SM", cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_NVSHMEM_SM));
}

TEST(ApiTransposeCommBackendToStringTest, ReturnsErrorForInvalidBackend) {
  EXPECT_STREQ("ERROR", cudecompTransposeCommBackendToString(static_cast<cudecompTransposeCommBackend_t>(999)));
}

TEST(ApiHaloCommBackendToStringTest, ReturnsBackendNames) {
  EXPECT_STREQ("MPI", cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_MPI));
  EXPECT_STREQ("MPI (blocking)", cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_MPI_BLOCKING));
  EXPECT_STREQ("NCCL", cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_NCCL));
  EXPECT_STREQ("NVSHMEM", cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_NVSHMEM));
  EXPECT_STREQ("NVSHMEM (blocking)", cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING));
}

TEST(ApiHaloCommBackendToStringTest, ReturnsErrorForInvalidBackend) {
  EXPECT_STREQ("ERROR", cudecompHaloCommBackendToString(static_cast<cudecompHaloCommBackend_t>(999)));
}

class ApiMpiTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    auto world_comm = cudecomp_test::MpiTestComm::world();
    if (world_comm.size() < kApiTestRanks) {
      GTEST_SKIP() << "API tests require " << kApiTestRanks << " ranks, launched with " << world_comm.size();
    }

    active_comm_ = cudecomp_test::MpiTestComm::split(world_comm, kApiTestRanks);
    if (!active_comm_.valid()) { GTEST_SKIP() << "inactive rank for " << kApiTestRanks << "-rank API case"; }

    const auto setup_decision = cudecomp_test::initializeGpuForTest(active_comm_);
    ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
    if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

    const cudecompResult_t init_result = cudecompInit(&handle_, active_comm_.mpiComm());
    handle_guard_ = std::make_unique<cudecomp_test::cudecompHandleGuard>(handle_);
    CHECK_CUDECOMP_GLOBAL(active_comm_, init_result);
  }

  void TearDown() override {
    handle_guard_.reset();
    active_comm_.reset();
  }

  cudecompGridDescConfig_t distributedConfig() {
    cudecompGridDescConfig_t config;
    EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));
    setDistributedConfig(config);
    return config;
  }

  cudecompGridDescConfig_t emptyPencilConfig() {
    auto config = distributedConfig();
    config.gdims_dist[0] = kGdims[0];
    config.gdims_dist[1] = 1;
    config.gdims_dist[2] = kGdims[2];
    return config;
  }

  cudecompGridDescAutotuneOptions_t fastAutotuneOptions() {
    cudecompGridDescAutotuneOptions_t options;
    EXPECT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescAutotuneOptionsSetDefaults(&options));
    options.n_warmup_trials = 0;
    options.n_trials = 1;
    options.dtype = CUDECOMP_FLOAT;
    options.disable_nccl_backends = true;
    options.disable_nvshmem_backends = true;
    return options;
  }

  void expectGridDescCreateInvalid(cudecompGridDescConfig_t config) {
    cudecompGridDesc_t grid_desc = nullptr;
    const cudecompResult_t result = cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr);
    EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, result);
    if (grid_desc) { (void)cudecompGridDescDestroy(handle_, grid_desc); }
  }

  cudecomp_test::MpiTestComm active_comm_;
  cudecompHandle_t handle_ = nullptr;
  std::unique_ptr<cudecomp_test::cudecompHandleGuard> handle_guard_;
};

class ApiInitTest : public ApiMpiTestBase {};
class ApiFinalizeTest : public ApiMpiTestBase {};
class ApiGridDescCreateTest : public ApiMpiTestBase {};
class ApiGridDescDestroyTest : public ApiMpiTestBase {};
class ApiGetGridDescConfigTest : public ApiMpiTestBase {};
class ApiGetPencilInfoTest : public ApiMpiTestBase {};
class ApiGetTransposeWorkspaceSizeTest : public ApiMpiTestBase {};
class ApiGetHaloWorkspaceSizeTest : public ApiMpiTestBase {};
class ApiGetShiftedRankTest : public ApiMpiTestBase {};
class ApiMallocTest : public ApiMpiTestBase {};
class ApiFreeTest : public ApiMpiTestBase {};
class ApiTransposeTest : public ApiMpiTestBase {};
class ApiHaloTest : public ApiMpiTestBase {};

TEST_F(ApiInitTest, RejectsInvalidArguments) {
  cudecompHandle_t second_handle = nullptr;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompInit(&second_handle, active_comm_.mpiComm()));
  EXPECT_EQ(nullptr, second_handle);
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompInit(nullptr, active_comm_.mpiComm()));
}

TEST_F(ApiFinalizeTest, RejectsInvalidArguments) {
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompFinalize(nullptr));
}

TEST_F(ApiGridDescCreateTest, RejectsInvalidConfigs) {
  auto config = distributedConfig();
  config.pdims[0] = 1;
  config.pdims[1] = 1;
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.pdims[0] = 0;
  config.pdims[1] = 1;
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.rank_order = static_cast<cudecompRankOrder_t>(999);
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(999);
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(999);
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.transpose_mem_order[0][0] = 0;
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  setMemOrder(config, {0, 1, 1});
  expectGridDescCreateInvalid(config);

  config = distributedConfig();
  config.gdims_dist[0] = kGdims[0] + 1;
  config.gdims_dist[1] = kGdims[1];
  config.gdims_dist[2] = kGdims[2];
  expectGridDescCreateInvalid(config);
}

TEST_F(ApiGridDescCreateTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t unused_grid_desc = nullptr;

  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescCreate(nullptr, &unused_grid_desc, &config, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescCreate(handle_, nullptr, &config, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescCreate(handle_, &unused_grid_desc, nullptr, nullptr));
}

TEST_F(ApiGridDescCreateTest, RejectsInvalidAutotuneInputs) {
  auto config = distributedConfig();
  config.pdims[0] = 0;
  config.pdims[1] = 0;
  cudecompGridDesc_t unused_grid_desc = nullptr;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescCreate(handle_, &unused_grid_desc, &config, nullptr));

  config = distributedConfig();
  auto options = fastAutotuneOptions();
  options.grid_mode = static_cast<cudecompAutotuneGridMode_t>(999);
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescCreate(handle_, &unused_grid_desc, &config, &options));
}

TEST_F(ApiGridDescDestroyTest, RejectsInvalidArguments) {
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGridDescDestroy(handle_, nullptr));
}

TEST_F(ApiGetGridDescConfigTest, CreatePreservesConfigSettings) {
  auto config = distributedConfig();
  setGdimsDist(config);
  config.rank_order = CUDECOMP_RANK_ORDER_COL_MAJOR;
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A;
  config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI_BLOCKING;
  for (int i = 0; i < 3; ++i) {
    config.transpose_axis_contiguous[i] = true;
  }

  auto expected_config = config;

  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  cudecompGridDescConfig_t queried_config;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetGridDescConfig(handle_, grid_desc, &queried_config));
  expectGridDescConfigEquals(config, expected_config);
  expectGridDescConfigEquals(queried_config, expected_config);
}

TEST_F(ApiGetGridDescConfigTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  cudecompGridDescConfig_t queried_config;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetGridDescConfig(handle_, grid_desc, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetGridDescConfig(handle_, nullptr, &queried_config));
}

TEST_F(ApiGridDescCreateTest, AutotunesTransposeConfig) {
  auto config = distributedConfig();
  config.pdims[0] = 0;
  config.pdims[1] = 0;
  auto options = fastAutotuneOptions();
  options.grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;
  options.autotune_transpose_backend = true;

  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, &options));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  EXPECT_GT(config.pdims[0], 0);
  EXPECT_GT(config.pdims[1], 0);
  EXPECT_EQ(kApiTestRanks, config.pdims[0] * config.pdims[1]);
  EXPECT_TRUE(isMpiTransposeBackend(config.transpose_comm_backend))
      << cudecompTransposeCommBackendToString(config.transpose_comm_backend);
  EXPECT_EQ(CUDECOMP_HALO_COMM_MPI, config.halo_comm_backend);

  cudecompGridDescConfig_t queried_config;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetGridDescConfig(handle_, grid_desc, &queried_config));
  expectGridDescConfigEquals(queried_config, config);
}

TEST_F(ApiGridDescCreateTest, AutotunesHaloConfig) {
  auto config = distributedConfig();
  config.pdims[0] = 0;
  config.pdims[1] = 0;
  auto options = fastAutotuneOptions();
  options.grid_mode = CUDECOMP_AUTOTUNE_GRID_HALO;
  options.autotune_halo_backend = true;
  for (int i = 0; i < 3; ++i) {
    options.halo_extents[i] = kHaloExtents[i];
    options.halo_periods[i] = kHaloPeriods[i];
    options.halo_padding[i] = kPadding[i];
  }

  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, &options));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  EXPECT_GT(config.pdims[0], 0);
  EXPECT_GT(config.pdims[1], 0);
  EXPECT_EQ(kApiTestRanks, config.pdims[0] * config.pdims[1]);
  EXPECT_EQ(CUDECOMP_TRANSPOSE_COMM_MPI_P2P, config.transpose_comm_backend);
  EXPECT_TRUE(isMpiHaloBackend(config.halo_comm_backend)) << cudecompHaloCommBackendToString(config.halo_comm_backend);

  cudecompGridDescConfig_t queried_config;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetGridDescConfig(handle_, grid_desc, &queried_config));
  expectGridDescConfigEquals(queried_config, config);
}

TEST_F(ApiGetPencilInfoTest, MatchesExpectedDefaultDecomposition) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  for (int axis = 0; axis < 3; ++axis) {
    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetPencilInfo(handle_, grid_desc, &pinfo, axis, kHaloExtents.data(),
                                                              kPadding.data()));
    expectPencilInfoEquals(pinfo, kExpectedDefaultPencilInfo[axis][active_comm_.rank()]);
  }
}

TEST_F(ApiGetPencilInfoTest, MatchesExpectedColumnMajorDecomposition) {
  auto config = distributedConfig();
  config.rank_order = CUDECOMP_RANK_ORDER_COL_MAJOR;
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  for (int axis = 0; axis < 3; ++axis) {
    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetPencilInfo(handle_, grid_desc, &pinfo, axis, kHaloExtents.data(),
                                                              kPadding.data()));
    expectPencilInfoEquals(pinfo, kExpectedColumnMajorPencilInfo[axis][active_comm_.rank()]);
  }
}

TEST_F(ApiGetPencilInfoTest, MatchesExpectedGdimsDistDecomposition) {
  auto config = distributedConfig();
  setGdimsDist(config);
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  for (int axis = 0; axis < 3; ++axis) {
    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetPencilInfo(handle_, grid_desc, &pinfo, axis, kHaloExtents.data(),
                                                              kPadding.data()));
    expectPencilInfoEquals(pinfo, kExpectedGdimsDistPencilInfo[axis][active_comm_.rank()]);
  }
}

TEST_F(ApiGetPencilInfoTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  cudecompPencilInfo_t pinfo;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetPencilInfo(handle_, grid_desc, nullptr, 0, nullptr, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetPencilInfo(handle_, grid_desc, &pinfo, -1, nullptr, nullptr));
}

TEST_F(ApiGetTransposeWorkspaceSizeTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int64_t workspace_size = 0;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetTransposeWorkspaceSize(handle_, grid_desc, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetTransposeWorkspaceSize(handle_, nullptr, &workspace_size));
}

TEST_F(ApiGetHaloWorkspaceSizeTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int64_t workspace_size = 0;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompGetHaloWorkspaceSize(handle_, grid_desc, 0, nullptr, &workspace_size));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompGetHaloWorkspaceSize(handle_, grid_desc, 0, kHaloExtents.data(), nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompGetHaloWorkspaceSize(handle_, grid_desc, 3, kHaloExtents.data(), &workspace_size));
}

TEST_F(ApiGetShiftedRankTest, ReturnsExpectedRanksForRowMajorLayout) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, 1, false, {2, 3, -1, -1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, -1, false, {-1, -1, 0, 1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, 1, true, {2, 3, 0, 1});

  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, 1, false, {1, -1, 3, -1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, -1, false, {-1, 0, -1, 2});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, 1, true, {1, 0, 3, 2});
}

TEST_F(ApiGetShiftedRankTest, ReturnsExpectedRanksForColumnMajorLayout) {
  auto config = distributedConfig();
  config.rank_order = CUDECOMP_RANK_ORDER_COL_MAJOR;
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, 1, false, {1, -1, 3, -1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, -1, false, {-1, 0, -1, 2});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 1, 1, true, {1, 0, 3, 2});

  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, 1, false, {2, 3, -1, -1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, -1, false, {-1, -1, 0, 1});
  expectShiftedRanks(active_comm_, handle_, grid_desc, 0, 2, 1, true, {2, 3, 0, 1});
}

TEST_F(ApiGetShiftedRankTest, HandlesAxisAlignedAndZeroDisplacements) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int32_t shifted_rank = -2;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetShiftedRank(handle_, grid_desc, 0, 1, 0, false, &shifted_rank));
  EXPECT_EQ(active_comm_.rank(), shifted_rank);

  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetShiftedRank(handle_, grid_desc, 0, 0, 1, false, &shifted_rank));
  EXPECT_EQ(-1, shifted_rank);

  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetShiftedRank(handle_, grid_desc, 0, 0, 1, true, &shifted_rank));
  EXPECT_EQ(active_comm_.rank(), shifted_rank);

  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGetShiftedRank(handle_, grid_desc, 0, 1, kPdims[0], true, &shifted_rank));
  EXPECT_EQ(active_comm_.rank(), shifted_rank);

  CHECK_CUDECOMP_GLOBAL(active_comm_,
                        cudecompGetShiftedRank(handle_, grid_desc, 0, 1, kPdims[0], false, &shifted_rank));
  EXPECT_EQ(-1, shifted_rank);
}

TEST_F(ApiGetShiftedRankTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int32_t shifted_rank = 0;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetShiftedRank(handle_, grid_desc, 0, 1, 1, false, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetShiftedRank(handle_, grid_desc, 3, 1, 1, false, &shifted_rank));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompGetShiftedRank(handle_, grid_desc, 0, 3, 1, false, &shifted_rank));
}

TEST_F(ApiMallocTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  void* buffer = nullptr;
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompMalloc(handle_, grid_desc, nullptr, 16));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompMalloc(handle_, grid_desc, &buffer, 0));
}

TEST_F(ApiFreeTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompFree(nullptr, grid_desc, nullptr));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE, cudecompFree(handle_, nullptr, nullptr));
}

TEST_F(ApiTransposeTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int placeholder = 0;
  void* valid_pointer = &placeholder;

  // X-to-Y is a representative placeholder for all transpose APIs; the directional variants route to the same
  // generic transpose implementation.
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompTransposeXToY(handle_, grid_desc, nullptr, valid_pointer, valid_pointer, CUDECOMP_FLOAT, nullptr,
                                  nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompTransposeXToY(handle_, grid_desc, valid_pointer, nullptr, valid_pointer, CUDECOMP_FLOAT, nullptr,
                                  nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompTransposeXToY(handle_, grid_desc, valid_pointer, valid_pointer, nullptr, CUDECOMP_FLOAT, nullptr,
                                  nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompTransposeXToY(handle_, grid_desc, valid_pointer, valid_pointer, valid_pointer,
                                  static_cast<cudecompDataType_t>(999), nullptr, nullptr, nullptr, nullptr, 0));
}

TEST_F(ApiTransposeTest, RejectsEmptyPencilDecomposition) {
  auto config = emptyPencilConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int input = 0;
  int output = 0;
  int work = 0;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED,
            cudecompTransposeXToY(handle_, grid_desc, &input, &output, &work, CUDECOMP_FLOAT, nullptr, nullptr, nullptr,
                                  nullptr, 0));
}

TEST_F(ApiHaloTest, RejectsInvalidArguments) {
  auto config = distributedConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int placeholder = 0;
  void* valid_pointer = &placeholder;

  // X-halo updates are a representative placeholder for all halo APIs; the directional variants route to the same
  // generic halo implementation.
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, valid_pointer, valid_pointer, CUDECOMP_FLOAT, nullptr,
                                 kHaloPeriods.data(), 0, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, valid_pointer, valid_pointer, CUDECOMP_FLOAT, kHaloExtents.data(),
                                 nullptr, 0, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, nullptr, valid_pointer, CUDECOMP_FLOAT, kHaloExtents.data(),
                                 kHaloPeriods.data(), 0, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, valid_pointer, nullptr, CUDECOMP_FLOAT, kHaloExtents.data(),
                                 kHaloPeriods.data(), 0, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, valid_pointer, valid_pointer, CUDECOMP_FLOAT, kHaloExtents.data(),
                                 kHaloPeriods.data(), 3, nullptr, 0));
  EXPECT_EQ(CUDECOMP_RESULT_INVALID_USAGE,
            cudecompUpdateHalosX(handle_, grid_desc, valid_pointer, valid_pointer, static_cast<cudecompDataType_t>(999),
                                 kHaloExtents.data(), kHaloPeriods.data(), 0, nullptr, 0));
}

TEST_F(ApiHaloTest, RejectsEmptyPencilDecomposition) {
  auto config = emptyPencilConfig();
  cudecompGridDesc_t grid_desc = nullptr;
  CHECK_CUDECOMP_GLOBAL(active_comm_, cudecompGridDescCreate(handle_, &grid_desc, &config, nullptr));
  cudecomp_test::gridDescGuard grid_desc_guard(handle_, grid_desc);

  int input = 0;
  int work = 0;
  EXPECT_EQ(CUDECOMP_RESULT_NOT_SUPPORTED,
            cudecompUpdateHalosX(handle_, grid_desc, &input, &work, CUDECOMP_FLOAT, kHaloExtents.data(),
                                 kHaloPeriods.data(), 0, nullptr, 0));
}

} // namespace
