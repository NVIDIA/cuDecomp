/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cctype>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudecomp.h"

#include "backend_test_context.h"
#include "backend_utils.h"
#include "gpu_test_utils.h"
#include "mpi_test_utils.h"
#include "test_utils.h"

namespace {

constexpr std::array<int32_t, 3> kBaselineGdims{9, 10, 11};
constexpr std::array<int32_t, 3> kZeroExtents{0, 0, 0};
constexpr std::array<int32_t, 3> kBaselineHaloExtents{1, 3, 2};
constexpr std::array<int32_t, 3> kNonzeroPadding{1, 2, 1};
constexpr std::array<bool, 3> kDefaultAxisContiguous{false, false, false};
constexpr std::array<bool, 3> kAllAxisContiguous{true, true, true};
constexpr std::array<std::array<int32_t, 3>, 3> kDefaultMemOrder{{{{-1, -1, -1}}, {{-1, -1, -1}}, {{-1, -1, -1}}}};
constexpr std::array<bool, 3> kPeriodicHalos{true, true, true};
constexpr std::array<bool, 3> kNonPeriodicHalos{false, false, false};

struct HaloCase {
  cudecomp_test::HaloBackend backend;
  const char* scenario;
  int axis;
  std::array<int32_t, 3> gdims;
  std::array<int32_t, 2> pdims;
  cudecompDataType_t dtype;
  std::array<bool, 3> axis_contiguous;
  std::array<std::array<int32_t, 3>, 3> mem_order;
  std::array<int32_t, 3> halo_extents;
  std::array<bool, 3> halo_periods;
  std::array<int32_t, 3> padding;
  cudecompRankOrder_t rank_order;
};

const char* axisName(int axis) {
  switch (axis) {
  case 0: return "X";
  case 1: return "Y";
  case 2: return "Z";
  }
  return "UnknownAxis";
}

std::string sanitizeParamName(const std::string& value) {
  std::string result;
  for (char ch : value) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      result.push_back(ch);
    } else {
      result.push_back('_');
    }
  }
  return result;
}

const char* dtypeName(cudecompDataType_t dtype) {
  switch (dtype) {
  case CUDECOMP_FLOAT: return "R32";
  case CUDECOMP_FLOAT_COMPLEX: return "C32";
  case CUDECOMP_DOUBLE: return "R64";
  case CUDECOMP_DOUBLE_COMPLEX: return "C64";
  }
  return "UnknownDtype";
}

std::string paramName(const testing::TestParamInfo<HaloCase>& info) {
  const auto& test_case = info.param;
  return sanitizeParamName(test_case.scenario) + "_Axis" + axisName(test_case.axis) + "_" +
         sanitizeParamName(test_case.backend.name) + "_" + dtypeName(test_case.dtype) + "_P" +
         std::to_string(test_case.pdims[0]) + "x" + std::to_string(test_case.pdims[1]);
}

HaloCase makeCase(cudecomp_test::HaloBackend backend, const char* scenario, int axis,
                  std::array<int32_t, 3> gdims = kBaselineGdims, std::array<int32_t, 2> pdims = {2, 2},
                  cudecompDataType_t dtype = CUDECOMP_FLOAT,
                  std::array<bool, 3> axis_contiguous = kDefaultAxisContiguous,
                  std::array<std::array<int32_t, 3>, 3> mem_order = kDefaultMemOrder,
                  std::array<int32_t, 3> halo_extents = kBaselineHaloExtents,
                  std::array<bool, 3> halo_periods = kPeriodicHalos, std::array<int32_t, 3> padding = kZeroExtents,
                  cudecompRankOrder_t rank_order = CUDECOMP_RANK_ORDER_DEFAULT) {
  return {backend,         scenario,  axis,         gdims,        pdims,   dtype,
          axis_contiguous, mem_order, halo_extents, halo_periods, padding, rank_order};
}

void appendBaselineCases(std::vector<HaloCase>& cases, const cudecomp_test::HaloBackend& backend) {
  // Baseline cases are the common halo sweep: all axis-aligned pencil layouts, default and all-axis-contiguous storage,
  // periodic and non-periodic boundary behavior, nonuniform halo extents, and single-precision real/complex data.
  struct LayoutCase {
    const char* periodic_scenario;
    const char* non_periodic_scenario;
    std::array<bool, 3> axis_contiguous;
  };

  for (const auto& layout : {LayoutCase{"BaselineDefaultLayoutPeriodicBoundaries",
                                        "BaselineDefaultLayoutNonPeriodicBoundaries", kDefaultAxisContiguous},
                             LayoutCase{"BaselineAxisContiguousPeriodicBoundaries",
                                        "BaselineAxisContiguousNonPeriodicBoundaries", kAllAxisContiguous}}) {
    for (int axis = 0; axis < 3; ++axis) {
      for (const auto dtype : {CUDECOMP_FLOAT, CUDECOMP_FLOAT_COMPLEX}) {
        cases.push_back(makeCase(backend, layout.periodic_scenario, axis, kBaselineGdims, {2, 2}, dtype,
                                 layout.axis_contiguous, kDefaultMemOrder, kBaselineHaloExtents, kPeriodicHalos));
        cases.push_back(makeCase(backend, layout.non_periodic_scenario, axis, kBaselineGdims, {2, 2}, dtype,
                                 layout.axis_contiguous, kDefaultMemOrder, kBaselineHaloExtents, kNonPeriodicHalos));
      }
    }
  }
}

void appendCoverageCases(std::vector<HaloCase>& cases, const cudecomp_test::HaloBackend& backend) {
  // Coverage cases target halo paths not guaranteed by the baseline sweep. They stay in the MPI collection because
  // these behaviors are shared halo logic rather than backend-specific communication coverage.
  cases.push_back(makeCase(backend, "NonzeroPadding", 0, kBaselineGdims, {2, 2}, CUDECOMP_FLOAT, kDefaultAxisContiguous,
                           kDefaultMemOrder, kBaselineHaloExtents, kPeriodicHalos, kNonzeroPadding));

  cases.push_back(makeCase(backend, "ColumnMajorRankOrder", 0, kBaselineGdims, {2, 2}, CUDECOMP_FLOAT,
                           kDefaultAxisContiguous, kDefaultMemOrder, kBaselineHaloExtents, kPeriodicHalos, kZeroExtents,
                           CUDECOMP_RANK_ORDER_COL_MAJOR));

  cases.push_back(makeCase(backend, "InteriorNonPeriodicNeighbors", 0, kBaselineGdims, {3, 1}, CUDECOMP_FLOAT,
                           kDefaultAxisContiguous, kDefaultMemOrder, kBaselineHaloExtents, kNonPeriodicHalos));

  cases.push_back(makeCase(backend, "DtypeWorkspacePadding", 0, kBaselineGdims, {2, 2}, CUDECOMP_DOUBLE,
                           kDefaultAxisContiguous, kDefaultMemOrder, kBaselineHaloExtents, kPeriodicHalos,
                           kNonzeroPadding));
  cases.push_back(makeCase(backend, "DtypeWorkspacePadding", 0, kBaselineGdims, {2, 2}, CUDECOMP_DOUBLE_COMPLEX,
                           kDefaultAxisContiguous, kDefaultMemOrder, kBaselineHaloExtents, kPeriodicHalos,
                           kNonzeroPadding));
}

std::vector<HaloCase> haloCasesForLabel(const char* label) {
  std::vector<HaloCase> cases;
  for (const auto& backend : cudecomp_test::haloBackends()) {
    if (std::string(backend.label) != label) continue;
    appendBaselineCases(cases, backend);
    if (std::string(label) == "mpi") { appendCoverageCases(cases, backend); }
  }
  return cases;
}

bool isInternal(const cudecompPencilInfo_t& pinfo, const std::array<int64_t, 3>& local) {
  return local[0] >= pinfo.halo_extents[pinfo.order[0]] &&
         local[0] < pinfo.shape[0] - pinfo.halo_extents[pinfo.order[0]] - pinfo.padding[pinfo.order[0]] &&
         local[1] >= pinfo.halo_extents[pinfo.order[1]] &&
         local[1] < pinfo.shape[1] - pinfo.halo_extents[pinfo.order[1]] - pinfo.padding[pinfo.order[1]] &&
         local[2] >= pinfo.halo_extents[pinfo.order[2]] &&
         local[2] < pinfo.shape[2] - pinfo.halo_extents[pinfo.order[2]] - pinfo.padding[pinfo.order[2]];
}

bool isPadding(const cudecompPencilInfo_t& pinfo, const std::array<int64_t, 3>& local) {
  return local[0] >= pinfo.shape[0] - pinfo.padding[pinfo.order[0]] ||
         local[1] >= pinfo.shape[1] - pinfo.padding[pinfo.order[1]] ||
         local[2] >= pinfo.shape[2] - pinfo.padding[pinfo.order[2]];
}

std::array<int64_t, 3> localCoordinate(int64_t index, const cudecompPencilInfo_t& pinfo) {
  std::array<int64_t, 3> local{};
  local[0] = index % pinfo.shape[0];
  local[1] = index / pinfo.shape[0] % pinfo.shape[1];
  local[2] = index / (pinfo.shape[0] * pinfo.shape[1]);
  return local;
}

std::array<int64_t, 3> globalCoordinate(const cudecompPencilInfo_t& pinfo, const std::array<int64_t, 3>& local) {
  std::array<int64_t, 3> global{};
  global[pinfo.order[0]] = local[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]];
  global[pinfo.order[1]] = local[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]];
  global[pinfo.order[2]] = local[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]];
  return global;
}

int64_t wrapIndex(int64_t index, int64_t size) {
  const int64_t wrapped = index % size;
  return (wrapped < 0) ? wrapped + size : wrapped;
}

template <typename T> T unsetValue() { return static_cast<T>(-1); }

template <> std::complex<float> unsetValue<std::complex<float>>() { return {-1.0f, 0.0f}; }

template <> std::complex<double> unsetValue<std::complex<double>>() { return {-1.0, 0.0}; }

template <typename T> T pencilValue(int64_t global_index) { return static_cast<T>(global_index); }

template <> std::complex<float> pencilValue<std::complex<float>>(int64_t global_index) {
  return {static_cast<float>(global_index), -static_cast<float>(global_index)};
}

template <> std::complex<double> pencilValue<std::complex<double>>(int64_t global_index) {
  return {static_cast<double>(global_index), -static_cast<double>(global_index)};
}

int64_t globalLinearIndex(const std::array<int64_t, 3>& global, const std::array<int32_t, 3>& gdims) {
  return global[0] + gdims[0] * (global[1] + global[2] * gdims[1]);
}

template <typename T>
std::vector<T> initializePencil(const cudecompPencilInfo_t& pinfo, const std::array<int32_t, 3>& gdims) {
  std::vector<T> data(pinfo.size, unsetValue<T>());

  for (int64_t i = 0; i < pinfo.size; ++i) {
    const auto local = localCoordinate(i, pinfo);
    if (!isInternal(pinfo, local)) continue;

    const auto global = globalCoordinate(pinfo, local);
    data[i] = pencilValue<T>(globalLinearIndex(global, gdims));
  }

  return data;
}

template <typename T>
std::vector<T> initializeReference(const cudecompPencilInfo_t& pinfo, const std::array<int32_t, 3>& gdims,
                                   const std::array<bool, 3>& halo_periods) {
  std::vector<T> data(pinfo.size, unsetValue<T>());

  for (int64_t i = 0; i < pinfo.size; ++i) {
    const auto local = localCoordinate(i, pinfo);
    auto global = globalCoordinate(pinfo, local);
    bool unset = isPadding(pinfo, local);

    for (int dim = 0; dim < 3; ++dim) {
      if (global[dim] >= 0 && global[dim] < gdims[dim]) continue;

      if (halo_periods[dim]) {
        global[dim] = wrapIndex(global[dim], gdims[dim]);
      } else {
        unset = true;
      }
    }

    if (!unset) { data[i] = pencilValue<T>(globalLinearIndex(global, gdims)); }
  }

  return data;
}

template <typename T>
testing::AssertionResult pencilMatches(const std::vector<T>& expected, const std::vector<T>& actual,
                                       const cudecompPencilInfo_t& pinfo) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure() << "size mismatch: expected " << expected.size() << ", got " << actual.size();
  }

  for (int64_t i = 0; i < pinfo.size; ++i) {
    if (expected[i] == actual[i]) continue;

    const auto local = localCoordinate(i, pinfo);
    return testing::AssertionFailure() << "mismatch at local index " << i << " coordinate (" << local[0] << ", "
                                       << local[1] << ", " << local[2] << "): expected " << expected[i] << ", got "
                                       << actual[i];
  }

  return testing::AssertionSuccess();
}

template <typename T>
cudecompResult_t runHalo(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, int axis, T* input, void* work,
                         cudecompDataType_t dtype, const cudecompPencilInfo_t& pinfo,
                         const std::array<bool, 3>& halo_periods, int dim) {
  switch (axis) {
  case 0:
    return cudecompUpdateHalosX(handle, grid_desc, input, work, dtype, pinfo.halo_extents, halo_periods.data(), dim,
                                pinfo.padding, 0);
  case 1:
    return cudecompUpdateHalosY(handle, grid_desc, input, work, dtype, pinfo.halo_extents, halo_periods.data(), dim,
                                pinfo.padding, 0);
  case 2:
    return cudecompUpdateHalosZ(handle, grid_desc, input, work, dtype, pinfo.halo_extents, halo_periods.data(), dim,
                                pinfo.padding, 0);
  }
  return CUDECOMP_RESULT_INVALID_USAGE;
}

} // namespace

class HaloCorrectnessTest : public ::testing::TestWithParam<HaloCase> {};

template <typename T> void runHaloCase(const HaloCase& test_case) {
  const int active_ranks = test_case.pdims[0] * test_case.pdims[1];
  const auto world_comm = cudecomp_test::MpiTestComm::world();

  if (world_comm.size() < active_ranks) {
    GTEST_SKIP() << "axis " << axisName(test_case.axis) << " halo case with pdims " << test_case.pdims[0] << "x"
                 << test_case.pdims[1] << " requires " << active_ranks << " ranks, launched with " << world_comm.size();
  }

  cudecompGridDescConfig_t config;
  ASSERT_EQ(CUDECOMP_RESULT_SUCCESS, cudecompGridDescConfigSetDefaults(&config));
  config.gdims[0] = test_case.gdims[0];
  config.gdims[1] = test_case.gdims[1];
  config.gdims[2] = test_case.gdims[2];
  config.pdims[0] = test_case.pdims[0];
  config.pdims[1] = test_case.pdims[1];
  config.rank_order = test_case.rank_order;
  config.halo_comm_backend = test_case.backend.backend;
  config.transpose_axis_contiguous[0] = test_case.axis_contiguous[0];
  config.transpose_axis_contiguous[1] = test_case.axis_contiguous[1];
  config.transpose_axis_contiguous[2] = test_case.axis_contiguous[2];
  for (int axis = 0; axis < 3; ++axis) {
    for (int i = 0; i < 3; ++i) {
      config.transpose_mem_order[axis][i] = test_case.mem_order[axis][i];
    }
  }

  cudecomp_test::BackendTestContext test_context;
  cudecomp_test::TestSetupDecision setup_decision;
  ASSERT_TRUE(test_context.initialize(world_comm, active_ranks, test_case.backend.label,
                                      std::string(test_case.backend.label) == "nccl", config, &setup_decision));
  ASSERT_FALSE(setup_decision.fail) << setup_decision.reason;
  if (setup_decision.skip) { GTEST_SKIP() << setup_decision.reason; }

  const auto& active_comm = test_context.comm();
  cudecompHandle_t handle = test_context.handle();
  ASSERT_NE(handle, nullptr);

  cudecompGridDesc_t grid_desc = nullptr;
  const cudecompResult_t grid_desc_create_result = cudecompGridDescCreate(handle, &grid_desc, &config, nullptr);
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);
  CHECK_CUDECOMP_GLOBAL(active_comm, grid_desc_create_result);

  cudecompPencilInfo_t pinfo;
  CHECK_CUDECOMP_GLOBAL(active_comm, cudecompGetPencilInfo(handle, grid_desc, &pinfo, test_case.axis,
                                                           test_case.halo_extents.data(), test_case.padding.data()));

  int64_t workspace_num_elements = 0;
  CHECK_CUDECOMP_GLOBAL(active_comm,
                        cudecompGetHaloWorkspaceSize(handle, grid_desc, test_case.axis, test_case.halo_extents.data(),
                                                     &workspace_num_elements));

  int64_t dtype_size = 0;
  CHECK_CUDECOMP_GLOBAL(active_comm, cudecompGetDataTypeSize(test_case.dtype, &dtype_size));

  const auto initial = initializePencil<T>(pinfo, test_case.gdims);
  const auto expected = initializeReference<T>(pinfo, test_case.gdims, test_case.halo_periods);

  T* data_d = nullptr;
  const cudaError_t data_alloc_result = cudaMalloc(&data_d, pinfo.size * sizeof(*data_d));
  cudecomp_test::cudaBufferGuard data_buffer(data_d);
  CHECK_CUDA_GLOBAL(active_comm, data_alloc_result);

  void* work_d = nullptr;
  const cudecompResult_t work_alloc_result =
      cudecompMalloc(handle, grid_desc, &work_d, workspace_num_elements * dtype_size);
  cudecomp_test::cudecompBufferGuard work_buffer(handle, grid_desc, work_d);
  CHECK_CUDECOMP_GLOBAL(active_comm, work_alloc_result);

  CHECK_CUDA_GLOBAL(active_comm, cudaMemset(data_d, 0, pinfo.size * sizeof(*data_d)));
  CHECK_CUDA_GLOBAL(active_comm,
                    cudaMemcpy(data_d, initial.data(), initial.size() * sizeof(*data_d), cudaMemcpyHostToDevice));
  CHECK_CUDA_GLOBAL(active_comm, cudaMemset(work_d, 0, workspace_num_elements * dtype_size));

  for (int dim = 0; dim < 3; ++dim) {
    CHECK_CUDECOMP_GLOBAL(active_comm, runHalo(handle, grid_desc, test_case.axis, data_d, work_d, test_case.dtype,
                                               pinfo, test_case.halo_periods, dim));
  }

  std::vector<T> actual(expected.size(), unsetValue<T>());
  CHECK_CUDA_GLOBAL(active_comm,
                    cudaMemcpy(actual.data(), data_d, actual.size() * sizeof(*data_d), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(pencilMatches(expected, actual, pinfo));
}

TEST_P(HaloCorrectnessTest, UpdateHalos) {
  const auto test_case = GetParam();
  switch (test_case.dtype) {
  case CUDECOMP_FLOAT: runHaloCase<float>(test_case); break;
  case CUDECOMP_FLOAT_COMPLEX: runHaloCase<std::complex<float>>(test_case); break;
  case CUDECOMP_DOUBLE: runHaloCase<double>(test_case); break;
  case CUDECOMP_DOUBLE_COMPLEX: runHaloCase<std::complex<double>>(test_case); break;
  default: FAIL() << "unsupported test dtype " << test_case.dtype;
  }
}

INSTANTIATE_TEST_SUITE_P(MpiBackends, HaloCorrectnessTest, ::testing::ValuesIn(haloCasesForLabel("mpi")), paramName);
INSTANTIATE_TEST_SUITE_P(NcclBackends, HaloCorrectnessTest, ::testing::ValuesIn(haloCasesForLabel("nccl")), paramName);
INSTANTIATE_TEST_SUITE_P(NvshmemBackends, HaloCorrectnessTest, ::testing::ValuesIn(haloCasesForLabel("nvshmem")),
                         paramName);
