/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudecomp.h"
#include "internal/common.h"

#include "backend_test_context.h"
#include "backend_utils.h"
#include "gpu_test_utils.h"
#include "mpi_test_utils.h"
#include "test_utils.h"

namespace {

enum class TransposeOperation { XToY, YToX, YToZ, ZToY };

constexpr std::array<TransposeOperation, 4> kTransposeOperations{TransposeOperation::XToY, TransposeOperation::YToX,
                                                                 TransposeOperation::YToZ, TransposeOperation::ZToY};
constexpr std::array<int32_t, 3> kBaselineGdims{9, 10, 11};
constexpr std::array<int32_t, 3> kZeroExtents{0, 0, 0};
constexpr std::array<int32_t, 3> kInputHaloExtents{1, 2, 1};
constexpr std::array<int32_t, 3> kOutputHaloExtents{2, 1, 1};
constexpr std::array<int32_t, 3> kInputPadding{1, 1, 2};
constexpr std::array<int32_t, 3> kOutputPadding{2, 1, 1};
constexpr std::array<bool, 3> kDefaultAxisContiguous{false, false, false};
constexpr std::array<bool, 3> kAllAxisContiguous{true, true, true};
constexpr std::array<std::array<int32_t, 3>, 3> kDefaultMemOrder{{{{-1, -1, -1}}, {{-1, -1, -1}}, {{-1, -1, -1}}}};

struct TransposeCase {
  cudecomp_test::TransposeBackend backend;
  const char* scenario;
  TransposeOperation operation;
  std::array<int32_t, 3> gdims;
  std::array<int32_t, 2> pdims;
  cudecompDataType_t dtype;
  bool out_of_place;
  std::array<bool, 3> axis_contiguous;
  std::array<std::array<int32_t, 3>, 3> mem_order;
  std::array<int32_t, 3> input_halo_extents;
  std::array<int32_t, 3> output_halo_extents;
  std::array<int32_t, 3> input_padding;
  std::array<int32_t, 3> output_padding;
  cudecompRankOrder_t rank_order;
  std::vector<int32_t> synthetic_host_groups;
};

const char* operationName(TransposeOperation operation) {
  switch (operation) {
  case TransposeOperation::XToY: return "XToY";
  case TransposeOperation::YToX: return "YToX";
  case TransposeOperation::YToZ: return "YToZ";
  case TransposeOperation::ZToY: return "ZToY";
  }
  return "Unknown";
}

int inputAxis(TransposeOperation operation) {
  switch (operation) {
  case TransposeOperation::XToY: return 0;
  case TransposeOperation::YToX: return 1;
  case TransposeOperation::YToZ: return 1;
  case TransposeOperation::ZToY: return 2;
  }
  return 0;
}

int outputAxis(TransposeOperation operation) {
  switch (operation) {
  case TransposeOperation::XToY: return 1;
  case TransposeOperation::YToX: return 0;
  case TransposeOperation::YToZ: return 2;
  case TransposeOperation::ZToY: return 1;
  }
  return 0;
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

std::string paramName(const testing::TestParamInfo<TransposeCase>& info) {
  const auto& test_case = info.param;
  return sanitizeParamName(test_case.scenario) + "_" + operationName(test_case.operation) + "_" +
         sanitizeParamName(test_case.backend.name) + "_" + dtypeName(test_case.dtype) + "_P" +
         std::to_string(test_case.pdims[0]) + "x" + std::to_string(test_case.pdims[1]) + "_" +
         (test_case.out_of_place ? "OutOfPlace" : "InPlace");
}

TransposeCase makeCase(cudecomp_test::TransposeBackend backend, const char* scenario, TransposeOperation operation,
                       std::array<int32_t, 3> gdims = kBaselineGdims, std::array<int32_t, 2> pdims = {2, 2},
                       cudecompDataType_t dtype = CUDECOMP_FLOAT, bool out_of_place = false,
                       std::array<bool, 3> axis_contiguous = kDefaultAxisContiguous,
                       std::array<std::array<int32_t, 3>, 3> mem_order = kDefaultMemOrder,
                       std::array<int32_t, 3> input_halo_extents = kZeroExtents,
                       std::array<int32_t, 3> output_halo_extents = kZeroExtents,
                       std::array<int32_t, 3> input_padding = kZeroExtents,
                       std::array<int32_t, 3> output_padding = kZeroExtents,
                       cudecompRankOrder_t rank_order = CUDECOMP_RANK_ORDER_DEFAULT) {
  return {backend,
          scenario,
          operation,
          gdims,
          pdims,
          dtype,
          out_of_place,
          axis_contiguous,
          mem_order,
          input_halo_extents,
          output_halo_extents,
          input_padding,
          output_padding,
          rank_order,
          {}};
}

TransposeCase withSyntheticHostGroups(TransposeCase test_case, std::vector<int32_t> synthetic_host_groups) {
  test_case.synthetic_host_groups = std::move(synthetic_host_groups);
  return test_case;
}

TransposeCase withHaloAndPadding(TransposeCase test_case) {
  test_case.input_halo_extents = kInputHaloExtents;
  test_case.output_halo_extents = kOutputHaloExtents;
  test_case.input_padding = kInputPadding;
  test_case.output_padding = kOutputPadding;
  return test_case;
}

void appendBaselineCases(std::vector<TransposeCase>& cases, const cudecomp_test::TransposeBackend& backend) {
  // Baseline cases are the common transpose sweep: all direct transpose operations, in-place and out-of-place, default
  // layout and all-axis-contiguous layout, and single-precision real/complex data. MPI also keeps P1x1 coverage for
  // the single-rank local special cases. NCCL and NVSHMEM skip P1x1 because it bypasses backend communication; for
  // NVSHMEM, it also avoids changing the PE count across init/finalize cycles inside one CTest process set.
  struct LayoutCase {
    const char* scenario;
    std::array<bool, 3> axis_contiguous;
  };

  for (const auto& layout : {LayoutCase{"BaselineDefaultLayout", kDefaultAxisContiguous},
                             LayoutCase{"BaselineAxisContiguous", kAllAxisContiguous}}) {
    for (const auto pdims : {std::array<int32_t, 2>{1, 1}, std::array<int32_t, 2>{2, 2}}) {
      if (pdims == std::array<int32_t, 2>{1, 1} && std::string(backend.label) != "mpi") continue;

      for (const auto dtype : {CUDECOMP_FLOAT, CUDECOMP_FLOAT_COMPLEX}) {
        for (const auto operation : kTransposeOperations) {
          for (const bool out_of_place : {false, true}) {
            cases.push_back(makeCase(backend, layout.scenario, operation, kBaselineGdims, pdims, dtype, out_of_place,
                                     layout.axis_contiguous));
          }
        }
      }
    }
  }
}

void appendNcclNativeAlltoAllCases(std::vector<TransposeCase>& cases, const cudecomp_test::TransposeBackend& backend) {
  if (backend.backend != CUDECOMP_TRANSPOSE_COMM_NCCL) return;

  cases.push_back(
      makeCase(backend, "NativeAlltoAllFastPath", TransposeOperation::YToZ, {8, 8, 8}, {1, 4}, CUDECOMP_FLOAT, true));
}

void appendCoverageCases(std::vector<TransposeCase>& cases, const cudecomp_test::TransposeBackend& backend) {
  // Coverage cases select explicit memory orders, nonzero halo/padding, dtypes, rank order, and rank counts to reach
  // transpose paths not guaranteed by the baseline sweep. These are not inherently MPI-only, but running them in the
  // MPI collection keeps NCCL/NVSHMEM focused on backend baseline coverage while still exercising shared transpose
  // logic.
  constexpr std::array<std::array<int32_t, 3>, 3> unpack_mem_order{{{{0, 1, 2}}, {{0, 1, 2}}, {{0, 1, 2}}}};
  constexpr std::array<std::array<int32_t, 3>, 3> transpose_unpack_mem_order{{{{0, 2, 1}}, {{0, 1, 2}}, {{0, 1, 2}}}};
  // These last-axis choices force the multi-rank split transpose/unpack path for each direct operation.
  constexpr std::array<std::array<int32_t, 3>, 3> split_unpack_mem_order{{{{0, 1, 2}}, {{0, 2, 1}}, {{1, 2, 0}}}};

  cases.push_back(
      withHaloAndPadding(makeCase(backend, "ExplicitMemOrderUnpack", TransposeOperation::XToY, kBaselineGdims, {2, 2},
                                  CUDECOMP_FLOAT, true, kDefaultAxisContiguous, unpack_mem_order)));
  cases.push_back(
      withHaloAndPadding(makeCase(backend, "ExplicitMemOrderTransposeUnpack", TransposeOperation::XToY, kBaselineGdims,
                                  {2, 2}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous, transpose_unpack_mem_order)));
  for (const auto operation : kTransposeOperations) {
    cases.push_back(
        withHaloAndPadding(makeCase(backend, "ExplicitMemOrderSplitUnpack", operation, kBaselineGdims, {2, 2},
                                    CUDECOMP_FLOAT, true, kDefaultAxisContiguous, split_unpack_mem_order)));
  }

  cases.push_back(makeCase(backend, "ColumnMajorRankOrder", TransposeOperation::XToY, kBaselineGdims, {2, 2},
                           CUDECOMP_FLOAT, false, kDefaultAxisContiguous, kDefaultMemOrder, kZeroExtents, kZeroExtents,
                           kZeroExtents, kZeroExtents, CUDECOMP_RANK_ORDER_COL_MAJOR));

  cases.push_back(
      withHaloAndPadding(makeCase(backend, "NonPowerOfTwoCommunicator", TransposeOperation::XToY, kBaselineGdims,
                                  {3, 1}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous, unpack_mem_order)));

  cases.push_back(
      withHaloAndPadding(makeCase(backend, "DtypeWorkspacePadding", TransposeOperation::XToY, kBaselineGdims, {2, 2},
                                  CUDECOMP_DOUBLE, true, kDefaultAxisContiguous, split_unpack_mem_order)));
  cases.push_back(
      withHaloAndPadding(makeCase(backend, "DtypeWorkspacePadding", TransposeOperation::YToZ, kBaselineGdims, {2, 2},
                                  CUDECOMP_DOUBLE_COMPLEX, true, kDefaultAxisContiguous, split_unpack_mem_order)));

  if (backend.backend == CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL) {
    // Pipelined MPI has additional paths for direct offset handling and inter-group scheduling. The synthetic host
    // groups below make a single-node test look like a multi-host communicator so those inter-group paths are covered.
    constexpr std::array<int32_t, 2> inter_group_pdims{3, 1};
    const std::vector<int32_t> inter_group_hosts{0, 1, 2};
    constexpr std::array<std::array<int32_t, 3>, 3> transpose_pack_offset_mem_order{
        {{{1, 0, 2}}, {{1, 2, 0}}, {{0, 1, 2}}}};
    constexpr std::array<std::array<int32_t, 3>, 3> direct_transpose_pack_offset_mem_order{
        {{{1, 0, 2}}, {{2, 1, 0}}, {{0, 1, 2}}}};
    constexpr std::array<std::array<int32_t, 3>, 3> direct_transpose_unpack_offset_mem_order{
        {{{1, 0, 2}}, {{0, 1, 2}}, {{0, 1, 2}}}};

    cases.push_back(withHaloAndPadding(makeCase(backend, "ExplicitMemOrderTransposePackOffset",
                                                TransposeOperation::XToY, kBaselineGdims, {2, 2}, CUDECOMP_FLOAT, true,
                                                kDefaultAxisContiguous, transpose_pack_offset_mem_order)));
    cases.push_back(withHaloAndPadding(makeCase(backend, "DirectTransposePackOffset", TransposeOperation::XToY,
                                                kBaselineGdims, {1, 1}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous,
                                                direct_transpose_pack_offset_mem_order)));
    cases.push_back(withHaloAndPadding(makeCase(backend, "DirectTransposeUnpackOffset", TransposeOperation::XToY,
                                                kBaselineGdims, {1, 1}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous,
                                                direct_transpose_unpack_offset_mem_order)));

    cases.push_back(withSyntheticHostGroups(
        withHaloAndPadding(makeCase(backend, "SyntheticInterGroupUnpack", TransposeOperation::XToY, kBaselineGdims,
                                    inter_group_pdims, CUDECOMP_FLOAT, true, kDefaultAxisContiguous, unpack_mem_order)),
        inter_group_hosts));
    cases.push_back(
        withSyntheticHostGroups(withHaloAndPadding(makeCase(backend, "SyntheticInterGroupTransposeUnpack",
                                                            TransposeOperation::XToY, kBaselineGdims, inter_group_pdims,
                                                            CUDECOMP_FLOAT, true, kDefaultAxisContiguous,
                                                            transpose_unpack_mem_order)),
                                inter_group_hosts));
    cases.push_back(
        withSyntheticHostGroups(withHaloAndPadding(makeCase(backend, "SyntheticInterGroupSplitUnpack",
                                                            TransposeOperation::XToY, kBaselineGdims, inter_group_pdims,
                                                            CUDECOMP_FLOAT, true, kDefaultAxisContiguous,
                                                            split_unpack_mem_order)),
                                inter_group_hosts));
  }
}

std::vector<TransposeCase> transposeCasesForLabel(const char* label) {
  std::vector<TransposeCase> cases;
  for (const auto& backend : cudecomp_test::transposeBackends()) {
    if (std::string(backend.label) != label) continue;

    appendBaselineCases(cases, backend);
    if (std::string(label) == "nccl") { appendNcclNativeAlltoAllCases(cases, backend); }
    if (std::string(label) == "mpi") { appendCoverageCases(cases, backend); }
  }
  return cases;
}

std::vector<TransposeCase> cudaGraphTransposeCases() {
  std::vector<TransposeCase> cases;
  for (const auto& backend : cudecomp_test::transposeBackends()) {
    if (backend.backend != CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL) continue;

    constexpr std::array<std::array<int32_t, 3>, 3> pack_mem_order{{{{0, 1, 2}}, {{0, 1, 2}}, {{0, 1, 2}}}};
    constexpr std::array<std::array<int32_t, 3>, 3> transpose_pack_mem_order{{{{0, 1, 2}}, {{1, 2, 0}}, {{0, 1, 2}}}};

    cases.push_back(withHaloAndPadding(makeCase(backend, "CudaGraphsPack", TransposeOperation::XToY, kBaselineGdims,
                                                {2, 2}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous, pack_mem_order)));
    cases.push_back(
        withHaloAndPadding(makeCase(backend, "CudaGraphsTransposePack", TransposeOperation::XToY, kBaselineGdims,
                                    {2, 2}, CUDECOMP_FLOAT, true, kDefaultAxisContiguous, transpose_pack_mem_order)));
  }
  return cases;
}

std::vector<TransposeCase> ncclUserBufferRegistrationCases() {
  std::vector<TransposeCase> cases;
  for (const auto& backend : cudecomp_test::transposeBackends()) {
    if (backend.backend != CUDECOMP_TRANSPOSE_COMM_NCCL) continue;
    cases.push_back(makeCase(backend, "UserBufferRegistration", TransposeOperation::XToY, kBaselineGdims, {2, 2},
                             CUDECOMP_FLOAT, true));
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

template <typename T> T pencilValue(int64_t global_index) { return static_cast<T>(global_index); }

template <> std::complex<float> pencilValue<std::complex<float>>(int64_t global_index) {
  return {static_cast<float>(global_index), -static_cast<float>(global_index)};
}

template <> std::complex<double> pencilValue<std::complex<double>>(int64_t global_index) {
  return {static_cast<double>(global_index), -static_cast<double>(global_index)};
}

template <typename T>
std::vector<T> initializePencil(const cudecompPencilInfo_t& pinfo, const std::array<int32_t, 3>& gdims) {
  std::vector<T> data(pinfo.size, T{-1});

  for (int64_t i = 0; i < pinfo.size; ++i) {
    std::array<int64_t, 3> local{};
    local[0] = i % pinfo.shape[0];
    local[1] = i / pinfo.shape[0] % pinfo.shape[1];
    local[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    if (!isInternal(pinfo, local)) continue;

    std::array<int64_t, 3> global{};
    global[pinfo.order[0]] = local[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]];
    global[pinfo.order[1]] = local[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]];
    global[pinfo.order[2]] = local[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]];

    data[i] = pencilValue<T>(global[0] + gdims[0] * (global[1] + global[2] * gdims[1]));
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

    std::array<int64_t, 3> local{};
    local[0] = i % pinfo.shape[0];
    local[1] = i / pinfo.shape[0] % pinfo.shape[1];
    local[2] = i / (pinfo.shape[0] * pinfo.shape[1]);

    if (!isInternal(pinfo, local)) continue;
    return testing::AssertionFailure() << "mismatch at local index " << i << " coordinate (" << local[0] << ", "
                                       << local[1] << ", " << local[2] << "): expected " << expected[i] << ", got "
                                       << actual[i];
  }

  return testing::AssertionSuccess();
}

template <typename T>
cudecompResult_t runTranspose(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, TransposeOperation operation,
                              T* input, T* output, void* work, cudecompDataType_t dtype,
                              const cudecompPencilInfo_t& input_info, const cudecompPencilInfo_t& output_info) {
  switch (operation) {
  case TransposeOperation::XToY:
    return cudecompTransposeXToY(handle, grid_desc, input, output, work, dtype, input_info.halo_extents,
                                 output_info.halo_extents, input_info.padding, output_info.padding, 0);
  case TransposeOperation::YToX:
    return cudecompTransposeYToX(handle, grid_desc, input, output, work, dtype, input_info.halo_extents,
                                 output_info.halo_extents, input_info.padding, output_info.padding, 0);
  case TransposeOperation::YToZ:
    return cudecompTransposeYToZ(handle, grid_desc, input, output, work, dtype, input_info.halo_extents,
                                 output_info.halo_extents, input_info.padding, output_info.padding, 0);
  case TransposeOperation::ZToY:
    return cudecompTransposeZToY(handle, grid_desc, input, output, work, dtype, input_info.halo_extents,
                                 output_info.halo_extents, input_info.padding, output_info.padding, 0);
  }
  return CUDECOMP_RESULT_INVALID_USAGE;
}

template <typename T>
void runAndVerifyTranspose(const cudecomp_test::MpiTestComm& active_comm, cudecompHandle_t handle,
                           cudecompGridDesc_t grid_desc, const TransposeCase& test_case, T* input_d, T* output_d,
                           void* work_d, int64_t data_num_elements, int64_t workspace_num_elements, int64_t dtype_size,
                           const std::vector<T>& input_ref, const std::vector<T>& output_ref,
                           const cudecompPencilInfo_t& input_info, const cudecompPencilInfo_t& output_info) {
  CHECK_CUDA_GLOBAL(active_comm, cudaMemset(input_d, 0, data_num_elements * sizeof(*input_d)));
  if (output_d != input_d) {
    CHECK_CUDA_GLOBAL(active_comm, cudaMemset(output_d, 0, data_num_elements * sizeof(*output_d)));
  }
  CHECK_CUDA_GLOBAL(active_comm,
                    cudaMemcpy(input_d, input_ref.data(), input_ref.size() * sizeof(*input_d), cudaMemcpyHostToDevice));
  CHECK_CUDA_GLOBAL(active_comm, cudaMemset(work_d, 0, workspace_num_elements * dtype_size));

  CHECK_CUDECOMP_GLOBAL(active_comm, runTranspose(handle, grid_desc, test_case.operation, input_d, output_d, work_d,
                                                  test_case.dtype, input_info, output_info));

  std::vector<T> output(output_ref.size(), T{});
  CHECK_CUDA_GLOBAL(active_comm,
                    cudaMemcpy(output.data(), output_d, output.size() * sizeof(*output_d), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(pencilMatches(output_ref, output, output_info));
}

cudecomp::cudecompCommAxis communicationAxis(TransposeOperation operation) {
  const int ax_a = inputAxis(operation);
  const int ax_b = outputAxis(operation);
  return (ax_a == 2 || ax_b == 2) ? cudecomp::CUDECOMP_COMM_ROW : cudecomp::CUDECOMP_COMM_COL;
}

testing::AssertionResult applySyntheticHostnames(cudecompHandle_t handle, const std::vector<int32_t>& host_groups) {
  // Inter-group transpose paths are normally selected from MPI processor names and topology metadata. Tests override
  // the handle before grid descriptor creation so communicator setup observes deterministic host groupings on one node.
  if (host_groups.empty()) return testing::AssertionSuccess();
  if (!handle) return testing::AssertionFailure() << "cannot apply synthetic hostnames to null cuDecomp handle";

  if (host_groups.size() != handle->hostnames.size()) {
    return testing::AssertionFailure() << "synthetic hostname group count " << host_groups.size()
                                       << " does not match handle rank count " << handle->hostnames.size();
  }

  for (int rank = 0; rank < static_cast<int>(host_groups.size()); ++rank) {
    std::string hostname = "cudecomp-test-host-" + std::to_string(host_groups[rank]);
    if (hostname.size() >= handle->hostnames[rank].size()) {
      return testing::AssertionFailure() << "synthetic hostname '" << hostname << "' exceeds MPI processor name limit";
    }
    handle->hostnames[rank].fill('\0');
    std::copy(hostname.begin(), hostname.end(), handle->hostnames[rank].begin());
  }

  // Force communicator setup to use the synthetic hostnames even on systems with MNNVL topology data.
  handle->rank_to_mnnvl_info.clear();
  handle->rank_to_clique.clear();
  handle->rank_to_clique_rank = handle->rank_to_local_rank;

  return testing::AssertionSuccess();
}

testing::AssertionResult syntheticTopologyIsActive(cudecompGridDesc_t grid_desc, TransposeOperation operation) {
  const auto comm_axis = communicationAxis(operation);
  const auto& comm_info =
      (comm_axis == cudecomp::CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;

  if (comm_info.ngroups <= 1) {
    return testing::AssertionFailure() << "synthetic topology did not create inter-group communicator: ngroups="
                                       << comm_info.ngroups << ", npergroup=" << comm_info.npergroup
                                       << ", nranks=" << comm_info.nranks;
  }

  return testing::AssertionSuccess();
}

} // namespace

class TransposeCorrectnessTest : public ::testing::TestWithParam<TransposeCase> {};
class CudaGraphTransposeCorrectnessTest : public ::testing::TestWithParam<TransposeCase> {};
class NcclUserBufferRegistrationTest : public ::testing::TestWithParam<TransposeCase> {};

testing::AssertionResult ncclUserBufferRegistrationIsActive(cudecompHandle_t handle, void* buffer) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  if (!handle->nccl_enable_ubr) { return testing::AssertionFailure() << "NCCL user buffer registration is disabled"; }

  auto entry = handle->nccl_ubr_handles.find(buffer);
  if (entry == handle->nccl_ubr_handles.end() || entry->second.empty()) {
    return testing::AssertionFailure() << "NCCL user buffer registration handle was not recorded";
  }

  return testing::AssertionSuccess();
#else
  return testing::AssertionFailure() << "NCCL user buffer registration requires NCCL 2.19 or newer";
#endif
}

template <typename T>
void runTransposeCase(const TransposeCase& test_case, bool check_cuda_graph_replay = false,
                      bool check_nccl_user_buffer_registration = false) {
  const int active_ranks = test_case.pdims[0] * test_case.pdims[1];
  const auto world_comm = cudecomp_test::MpiTestComm::world();

  if (world_comm.size() < active_ranks) {
    GTEST_SKIP() << operationName(test_case.operation) << " with pdims " << test_case.pdims[0] << "x"
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
  config.transpose_comm_backend = test_case.backend.backend;
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
  ASSERT_TRUE(applySyntheticHostnames(handle, test_case.synthetic_host_groups));
  if (check_cuda_graph_replay) { ASSERT_TRUE(handle->cuda_graphs_enable); }

  cudecompGridDesc_t grid_desc = nullptr;
  const cudecompResult_t grid_desc_create_result = cudecompGridDescCreate(handle, &grid_desc, &config, nullptr);
  cudecomp_test::gridDescGuard grid_desc_guard(handle, grid_desc);
  CHECK_CUDECOMP_GLOBAL(active_comm, grid_desc_create_result);
  if (!test_case.synthetic_host_groups.empty()) {
    ASSERT_TRUE(syntheticTopologyIsActive(grid_desc, test_case.operation));
  }

  cudecompPencilInfo_t input_info;
  cudecompPencilInfo_t output_info;
  CHECK_CUDECOMP_GLOBAL(active_comm,
                        cudecompGetPencilInfo(handle, grid_desc, &input_info, inputAxis(test_case.operation),
                                              test_case.input_halo_extents.data(), test_case.input_padding.data()));
  CHECK_CUDECOMP_GLOBAL(active_comm,
                        cudecompGetPencilInfo(handle, grid_desc, &output_info, outputAxis(test_case.operation),
                                              test_case.output_halo_extents.data(), test_case.output_padding.data()));

  int64_t workspace_num_elements = 0;
  CHECK_CUDECOMP_GLOBAL(active_comm, cudecompGetTransposeWorkspaceSize(handle, grid_desc, &workspace_num_elements));

  int64_t dtype_size = 0;
  CHECK_CUDECOMP_GLOBAL(active_comm, cudecompGetDataTypeSize(test_case.dtype, &dtype_size));

  const auto input_ref = initializePencil<T>(input_info, test_case.gdims);
  const auto output_ref = initializePencil<T>(output_info, test_case.gdims);
  const int64_t data_num_elements = std::max(input_info.size, output_info.size);

  T* input_d = nullptr;
  const cudaError_t input_alloc_result = cudaMalloc(&input_d, data_num_elements * sizeof(*input_d));
  cudecomp_test::cudaBufferGuard input_buffer(input_d);
  CHECK_CUDA_GLOBAL(active_comm, input_alloc_result);

  if (check_cuda_graph_replay) {
    ASSERT_TRUE(test_case.out_of_place);

    T* output_a_d = nullptr;
    const cudaError_t output_a_alloc_result = cudaMalloc(&output_a_d, data_num_elements * sizeof(*output_a_d));
    cudecomp_test::cudaBufferGuard output_a_buffer(output_a_d);
    CHECK_CUDA_GLOBAL(active_comm, output_a_alloc_result);

    T* output_b_d = nullptr;
    const cudaError_t output_b_alloc_result = cudaMalloc(&output_b_d, data_num_elements * sizeof(*output_b_d));
    cudecomp_test::cudaBufferGuard output_b_buffer(output_b_d);
    CHECK_CUDA_GLOBAL(active_comm, output_b_alloc_result);

    void* work_a_d = nullptr;
    const cudecompResult_t work_a_alloc_result =
        cudecompMalloc(handle, grid_desc, &work_a_d, workspace_num_elements * dtype_size);
    cudecomp_test::cudecompBufferGuard work_a_buffer(handle, grid_desc, work_a_d);
    CHECK_CUDECOMP_GLOBAL(active_comm, work_a_alloc_result);

    void* work_b_d = nullptr;
    const cudecompResult_t work_b_alloc_result =
        cudecompMalloc(handle, grid_desc, &work_b_d, workspace_num_elements * dtype_size);
    cudecomp_test::cudecompBufferGuard work_b_buffer(handle, grid_desc, work_b_d);
    CHECK_CUDECOMP_GLOBAL(active_comm, work_b_alloc_result);

    runAndVerifyTranspose(active_comm, handle, grid_desc, test_case, input_d, output_a_d, work_a_d, data_num_elements,
                          workspace_num_elements, dtype_size, input_ref, output_ref, input_info, output_info);
    runAndVerifyTranspose(active_comm, handle, grid_desc, test_case, input_d, output_a_d, work_a_d, data_num_elements,
                          workspace_num_elements, dtype_size, input_ref, output_ref, input_info, output_info);
    runAndVerifyTranspose(active_comm, handle, grid_desc, test_case, input_d, output_b_d, work_b_d, data_num_elements,
                          workspace_num_elements, dtype_size, input_ref, output_ref, input_info, output_info);
    runAndVerifyTranspose(active_comm, handle, grid_desc, test_case, input_d, output_b_d, work_b_d, data_num_elements,
                          workspace_num_elements, dtype_size, input_ref, output_ref, input_info, output_info);
    return;
  }

  T* output_d = input_d;
  cudecomp_test::cudaBufferGuard output_buffer;
  if (test_case.out_of_place) {
    T* allocated_output_d = nullptr;
    const cudaError_t output_alloc_result = cudaMalloc(&allocated_output_d, data_num_elements * sizeof(*output_d));
    output_buffer.reset(allocated_output_d);
    CHECK_CUDA_GLOBAL(active_comm, output_alloc_result);
    output_d = allocated_output_d;
  }

  void* work_d = nullptr;
  const cudecompResult_t work_alloc_result =
      cudecompMalloc(handle, grid_desc, &work_d, workspace_num_elements * dtype_size);
  cudecomp_test::cudecompBufferGuard work_buffer(handle, grid_desc, work_d);
  CHECK_CUDECOMP_GLOBAL(active_comm, work_alloc_result);
  if (check_nccl_user_buffer_registration) { ASSERT_TRUE(ncclUserBufferRegistrationIsActive(handle, work_d)); }

  runAndVerifyTranspose(active_comm, handle, grid_desc, test_case, input_d, output_d, work_d, data_num_elements,
                        workspace_num_elements, dtype_size, input_ref, output_ref, input_info, output_info);

  if (check_nccl_user_buffer_registration) {
    const cudecompResult_t work_free_result = cudecompFree(handle, grid_desc, work_d);
    if (work_free_result == CUDECOMP_RESULT_SUCCESS) { work_buffer.release(); }
    CHECK_CUDECOMP_GLOBAL(active_comm, work_free_result);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
    EXPECT_EQ(handle->nccl_ubr_handles.count(work_d), 0);
#endif
  }
}

TEST_P(TransposeCorrectnessTest, DirectOperation) {
  const auto test_case = GetParam();
  switch (test_case.dtype) {
  case CUDECOMP_FLOAT: runTransposeCase<float>(test_case); break;
  case CUDECOMP_FLOAT_COMPLEX: runTransposeCase<std::complex<float>>(test_case); break;
  case CUDECOMP_DOUBLE: runTransposeCase<double>(test_case); break;
  case CUDECOMP_DOUBLE_COMPLEX: runTransposeCase<std::complex<double>>(test_case); break;
  default: FAIL() << "unsupported test dtype " << test_case.dtype;
  }
}

TEST_P(CudaGraphTransposeCorrectnessTest, CapturesAndReplaysPackingGraph) {
  const auto test_case = GetParam();
  switch (test_case.dtype) {
  case CUDECOMP_FLOAT: runTransposeCase<float>(test_case, true); break;
  case CUDECOMP_FLOAT_COMPLEX: runTransposeCase<std::complex<float>>(test_case, true); break;
  case CUDECOMP_DOUBLE: runTransposeCase<double>(test_case, true); break;
  case CUDECOMP_DOUBLE_COMPLEX: runTransposeCase<std::complex<double>>(test_case, true); break;
  default: FAIL() << "unsupported test dtype " << test_case.dtype;
  }
}

TEST_P(NcclUserBufferRegistrationTest, DirectOperation) {
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 19, 0)
  GTEST_SKIP() << "NCCL user buffer registration requires NCCL 2.19 or newer";
#else
  const auto test_case = GetParam();
  switch (test_case.dtype) {
  case CUDECOMP_FLOAT: runTransposeCase<float>(test_case, false, true); break;
  case CUDECOMP_FLOAT_COMPLEX: runTransposeCase<std::complex<float>>(test_case, false, true); break;
  case CUDECOMP_DOUBLE: runTransposeCase<double>(test_case, false, true); break;
  case CUDECOMP_DOUBLE_COMPLEX: runTransposeCase<std::complex<double>>(test_case, false, true); break;
  default: FAIL() << "unsupported test dtype " << test_case.dtype;
  }
#endif
}

INSTANTIATE_TEST_SUITE_P(MpiBackends, TransposeCorrectnessTest, ::testing::ValuesIn(transposeCasesForLabel("mpi")),
                         paramName);
INSTANTIATE_TEST_SUITE_P(NcclBackends, TransposeCorrectnessTest, ::testing::ValuesIn(transposeCasesForLabel("nccl")),
                         paramName);
INSTANTIATE_TEST_SUITE_P(NvshmemBackends, TransposeCorrectnessTest,
                         ::testing::ValuesIn(transposeCasesForLabel("nvshmem")), paramName);
INSTANTIATE_TEST_SUITE_P(CudaGraphMpiBackends, CudaGraphTransposeCorrectnessTest,
                         ::testing::ValuesIn(cudaGraphTransposeCases()), paramName);
INSTANTIATE_TEST_SUITE_P(NcclUserBufferRegistration, NcclUserBufferRegistrationTest,
                         ::testing::ValuesIn(ncclUserBufferRegistrationCases()), paramName);
