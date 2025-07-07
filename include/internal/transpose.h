/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <array>
#include <cstdint>
#include <vector>

#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <mpi.h>

#include "internal/checks.h"
#include "internal/comm_routines.h"
#include "internal/cudecomp_kernels.h"
#include "internal/nvtx.h"
#include "internal/performance.h"
#include "internal/utils.h"

namespace cudecomp {

static inline bool isTransposeCommPipelined(cudecompTransposeCommBackend_t commType) {
  return (commType == CUDECOMP_TRANSPOSE_COMM_NCCL_PL ||
#ifdef ENABLE_NVSHMEM
          commType == CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL ||
#endif
          commType == CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL);
}

#if CUTENSOR_MAJOR >= 2
static inline cutensorDataType_t getCutensorDataType(float) { return CUTENSOR_R_32F; }
static inline cutensorDataType_t getCutensorDataType(double) { return CUTENSOR_R_64F; }
static inline cutensorDataType_t getCutensorDataType(cuda::std::complex<float>) { return CUTENSOR_C_32F; }
static inline cutensorDataType_t getCutensorDataType(cuda::std::complex<double>) { return CUTENSOR_C_64F; }
template <typename T> static inline cutensorDataType_t getCutensorDataType() { return getCutensorDataType(T(0)); }

static inline cutensorComputeDescriptor_t getCutensorComputeType(cutensorDataType_t cutensor_dtype) {
  switch (cutensor_dtype) {
  case CUTENSOR_R_32F:
  case CUTENSOR_C_32F: return CUTENSOR_COMPUTE_DESC_32F;
  case CUTENSOR_R_64F:
  case CUTENSOR_C_64F:
  default: return CUTENSOR_COMPUTE_DESC_64F;
  }
}

template <typename T> static inline uint32_t getAlignment(const T* ptr) {
  auto i_ptr = reinterpret_cast<std::uintptr_t>(ptr);
  for (uint32_t d = 16; d > 0; d >>= 1) {
    if (i_ptr % d == 0) return d;
  }
  return 1;
}

template <typename T>
static void localPermute(const cudecompHandle_t handle, const std::array<int64_t, 3>& extent_in,
                         const std::array<int32_t, 3>& order_out, const std::array<int64_t, 3>& strides_in,
                         const std::array<int64_t, 3>& strides_out, T* input, T* output, cudaStream_t stream) {
  cutensorDataType_t cutensor_type = getCutensorDataType<T>();

  std::array<int32_t, 3> order_in{0, 1, 2};
  std::array<int64_t, 3> extent_out;
  for (int i = 0; i < 3; ++i) {
    extent_out[i] = extent_in[order_out[i]];
    if (extent_out[i] == 0) return;
  }

  auto strides_in_ptr = anyNonzeros(strides_in) ? strides_in.data() : nullptr;
  auto strides_out_ptr = anyNonzeros(strides_out) ? strides_out.data() : nullptr;

  cutensorTensorDescriptor_t desc_in;
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle->cutensor_handle, &desc_in, 3, extent_in.data(), strides_in_ptr,
                                                cutensor_type, getAlignment(input)));
  cutensorTensorDescriptor_t desc_out;
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle->cutensor_handle, &desc_out, 3, extent_out.data(),
                                                strides_out_ptr, cutensor_type, getAlignment(output)));

  cutensorOperationDescriptor_t desc_op;
  CHECK_CUTENSOR(cutensorCreatePermutation(handle->cutensor_handle, &desc_op, desc_in, order_in.data(),
                                           CUTENSOR_OP_IDENTITY, desc_out, order_out.data(),
                                           getCutensorComputeType(cutensor_type)));

  cutensorPlan_t plan;
  CHECK_CUTENSOR(cutensorCreatePlan(handle->cutensor_handle, &plan, desc_op, handle->cutensor_plan_pref, 0));

  T one(1);
  CHECK_CUTENSOR(cutensorPermute(handle->cutensor_handle, plan, &one, input, output, stream));

  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(desc_in));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(desc_out));
  CHECK_CUTENSOR(cutensorDestroyOperationDescriptor(desc_op));
  CHECK_CUTENSOR(cutensorDestroyPlan(plan));
}

#else

static inline cudaDataType_t getCudaDataType(float) { return CUDA_R_32F; }
static inline cudaDataType_t getCudaDataType(double) { return CUDA_R_64F; }
static inline cudaDataType_t getCudaDataType(cuda::std::complex<float>) { return CUDA_C_32F; }
static inline cudaDataType_t getCudaDataType(cuda::std::complex<double>) { return CUDA_C_64F; }
template <typename T> static inline cudaDataType_t getCudaDataType() { return getCudaDataType(T(0)); }

template <typename T>
static void localPermute(const cudecompHandle_t handle, const std::array<int64_t, 3>& extent_in,
                         const std::array<int32_t, 3>& order_out, const std::array<int64_t, 3>& strides_in,
                         const std::array<int64_t, 3>& strides_out, T* input, T* output, cudaStream_t stream) {
  cudaDataType_t cuda_type = getCudaDataType<T>();

  std::array<int32_t, 3> order_in{0, 1, 2};
  std::array<int64_t, 3> extent_out;
  for (int i = 0; i < 3; ++i) {
    extent_out[i] = extent_in[order_out[i]];
    if (extent_out[i] == 0) return;
  }

  auto strides_in_ptr = anyNonzeros(strides_in) ? strides_in.data() : nullptr;
  auto strides_out_ptr = anyNonzeros(strides_out) ? strides_out.data() : nullptr;

  cutensorTensorDescriptor_t desc_in;
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(&handle->cutensor_handle, &desc_in, 3, extent_in.data(), strides_in_ptr,
                                              cuda_type, CUTENSOR_OP_IDENTITY));
  cutensorTensorDescriptor_t desc_out;
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(&handle->cutensor_handle, &desc_out, 3, extent_out.data(),
                                              strides_out_ptr, cuda_type, CUTENSOR_OP_IDENTITY));

  T one(1);
  CHECK_CUTENSOR(cutensorPermutation(&handle->cutensor_handle, &one, input, &desc_in, order_in.data(), output,
                                     &desc_out, order_out.data(), cuda_type, stream));
}
#endif

template <typename T>
static void cudecompTranspose_(int ax, int dir, const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                               T* input, T* output, T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                               const int32_t output_halo_extents_ptr[] = nullptr,
                               const int32_t input_padding_ptr[] = nullptr,
                               const int32_t output_padding_ptr[] = nullptr, cudaStream_t stream = 0) {

  std::array<int32_t, 3> input_halo_extents{};
  std::array<int32_t, 3> output_halo_extents{};
  if (input_halo_extents_ptr) std::copy(input_halo_extents_ptr, input_halo_extents_ptr + 3, input_halo_extents.begin());
  if (output_halo_extents_ptr)
    std::copy(output_halo_extents_ptr, output_halo_extents_ptr + 3, output_halo_extents.begin());

  std::array<int32_t, 3> input_padding{};
  std::array<int32_t, 3> output_padding{};
  if (input_padding_ptr) std::copy(input_padding_ptr, input_padding_ptr + 3, input_padding.begin());
  if (output_padding_ptr) std::copy(output_padding_ptr, output_padding_ptr + 3, output_padding.begin());

  bool fwd = dir > 0;

  bool inplace = (input == output);
  bool input_has_halos_padding = anyNonzeros(input_halo_extents) || anyNonzeros(input_padding);
  bool output_has_halos_padding = anyNonzeros(output_halo_extents) || anyNonzeros(output_padding);
  bool pipelined = isTransposeCommPipelined(grid_desc->config.transpose_comm_backend);
  int memcpy_limit = pipelined ? 1 : CUDECOMP_BATCHED_D2D_3D_PARAM_CAPACITY;

  // Set axis values
  int ax_a = ax;
  int ax_b = (fwd ? ax_a + 1 : ax_a + 2) % 3;
  int ax_c = (fwd ? ax_a + 2 : ax_a + 1) % 3;

  auto comm_axis = (ax_a == 2 || ax_b == 2) ? CUDECOMP_COMM_ROW : CUDECOMP_COMM_COL;
  const auto& comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
  int comm_rank = comm_info.rank;

  // Get splits
  auto splits_a =
      getSplits(grid_desc->config.gdims_dist[ax_a], grid_desc->config.pdims[comm_axis == CUDECOMP_COMM_COL ? 0 : 1],
                grid_desc->config.gdims[ax_a] - grid_desc->config.gdims_dist[ax_a]);
  auto splits_b =
      getSplits(grid_desc->config.gdims_dist[ax_b], grid_desc->config.pdims[comm_axis == CUDECOMP_COMM_COL ? 0 : 1],
                grid_desc->config.gdims[ax_b] - grid_desc->config.gdims_dist[ax_b]);

  // Set offsets
  std::vector<int64_t> offsets_a(splits_a.size(), 0);
  std::vector<int64_t> offsets_b(splits_b.size(), 0);
  for (int i = 0; i < splits_a.size() - 1; ++i) {
    offsets_a[i + 1] = offsets_a[i] + splits_a[i];
    offsets_b[i + 1] = offsets_b[i] + splits_b[i];
  }

  // Get pencil info
  cudecompPencilInfo_t pinfo_a, pinfo_a_h;
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_a, ax_a, nullptr, nullptr));
  CHECK_CUDECOMP(
      cudecompGetPencilInfo(handle, grid_desc, &pinfo_a_h, ax_a, input_halo_extents.data(), input_padding.data()));
  cudecompPencilInfo_t pinfo_b, pinfo_b_h;
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_b, ax_b, nullptr, nullptr));
  CHECK_CUDECOMP(
      cudecompGetPencilInfo(handle, grid_desc, &pinfo_b_h, ax_b, output_halo_extents.data(), output_padding.data()));

  // Check if input and output orders are the same
  bool orders_equal = true;
  for (int i = 0; i < 3; ++i) {
    if (pinfo_a.order[i] != pinfo_b.order[i]) orders_equal = false;
  }

  // Check if input and output halo extents and padding are the same
  bool halos_padding_equal = true;
  for (int i = 0; i < 3; ++i) {
    if (input_halo_extents[i] != output_halo_extents[i]) halos_padding_equal = false;
    if (input_padding[i] != output_padding[i]) halos_padding_equal = false;
  }

  // Get global ordered shapes
  auto shape_g_a = getShapeG(pinfo_a);
  auto shape_g_a_h = getShapeG(pinfo_a_h);
  auto shape_g_b = getShapeG(pinfo_b);
  auto shape_g_b_h = getShapeG(pinfo_b_h);

  // Set input/output pointers for each phase
  T* i1 = input;
  T* o1 = work;
  T* o2 = work + pinfo_a.size;
  T* o3 = output;

  if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend)) {
    auto max_pencil_size_a = getGlobalMaxPencilSize(handle, grid_desc, ax_a);
    o2 = work + max_pencil_size_a;
    // Record event at start of transpose op for NVSHMEM team synchronization
    CHECK_CUDA(cudaEventRecord(grid_desc->nvshmem_sync_event, stream));
  }

  cudecompTransposePerformanceSample* current_sample = nullptr;
  if (handle->performance_report_enable) {
    auto& samples = getOrCreateTransposePerformanceSamples(handle, grid_desc, createTransposeConfig(ax, dir, input, output, input_halo_extents.data(), output_halo_extents.data(), input_padding.data(), output_padding.data(), getCudecompDataType<T>()));
    current_sample = &samples.samples[samples.sample_idx];
    current_sample->alltoall_timing_count = 0;
    current_sample->alltoall_bytes = pinfo_a.size * sizeof(T);
    current_sample->valid = true;

    // Record start event
    CHECK_CUDA(cudaEventRecord(current_sample->transpose_start_event, stream));
  }

  // Adjust pointers to handle special cases
  bool direct_pack = false;
  bool direct_transpose = false;
  if (splits_a.size() == 1) {
    // Special cases for single rank communicators
    if (orders_equal) {
      if (inplace) {
        if (halos_padding_equal) {
          // Single rank, in place, Pack -> Unpack: No transpose necessary.
          if (handle->performance_report_enable) {
            // Record performance data
            CHECK_CUDA(cudaEventRecord(current_sample->transpose_end_event, stream));
            advanceTransposePerformanceSample(handle, grid_desc, createTransposeConfig(ax, dir, input, output, input_halo_extents.data(), output_halo_extents.data(), input_padding.data(), output_padding.data(), getCudecompDataType<T>()));
          }
          return;
        }
      } else {
        // Single rank, out of place, Pack -> Unpack: Pack directly to output and return
        o1 = output;
        direct_pack = true;
      }
    } else {
      if (!inplace) {
        if (pinfo_b.order[2] == ax_a) {
          // Single rank, out of place, Transpose -> Unpack: Transpose directly to
          // output and return
          o1 = output;
          direct_transpose = true;
        } else {
          // Single rank, out of place, Pack -> Transpose: Skip pack, transpose directly
          // to output
          o1 = input;
          o2 = input;
          direct_transpose = true;
        }
      }
    }
  } else {
    // Special cases that skip local phases (i.e. all-to-all directly from/to input or output)

    bool enable = true;

    if (pipelined && inplace) {
      // Note: Disabling special cases for in-place pipelined communication to avoid
      // handing more complex input/output data dependencies. Can revisit.
      enable = false;
    } else if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend)) {
      // Note: For NVSHMEM, disabling special cases to ensure communication is always staged
      // in to workspace (which should be nvshmem allocated). Can revisit support for input/output
      // arrays allocated with nvshmem.
      enable = false;
    } else if (transposeBackendRequiresMpi(grid_desc->config.transpose_comm_backend) &&
               (isManagedPointer(input) || isManagedPointer(output))) {
      // Note: For MPI, disable special cases if input or output pointers are to managed memory
      // since MPI performance directly from managed memory is not great
      enable = false;
    }

    if (enable) {
      if (pinfo_a.order[2] == ax_a && !input_has_halos_padding) {
        // Input is already packed for all to all, skip pack
        o1 = input;
        o2 = work;
      } else if (pinfo_a.order[2] == ax_b && orders_equal && !output_has_halos_padding) {
        // Output of all to all is in correct orientation, skip unpack
        o2 = output;
      }
    }
  }

  // Setup communication info
  std::vector<comm_count_t> send_offsets(splits_a.size());
  std::vector<comm_count_t> recv_offsets(splits_a.size());
  std::vector<comm_count_t> send_counts(splits_a.size());
  std::vector<comm_count_t> recv_counts(splits_a.size());

  std::vector<comm_count_t> recv_offsets_nvshmem(splits_a.size());
  size_t rank_offset = offsets_b[comm_rank];
  for (int i = 0; i < splits_a.size(); ++i) {
    send_offsets[i] = offsets_a[i] * shape_g_a[ax_b] * shape_g_a[ax_c];
    recv_offsets[i] = offsets_b[i] * shape_g_b[ax_a] * shape_g_b[ax_c];
    send_counts[i] = splits_a[i] * shape_g_a[ax_b] * shape_g_a[ax_c];
    recv_counts[i] = splits_b[i] * shape_g_b[ax_a] * shape_g_b[ax_c];

    recv_offsets_nvshmem[i] = rank_offset * shape_g_a[ax_c] * splits_a[i];
  }

  bool data_transposed = false;

  if (o1 != i1) {
    if (pinfo_b.order[2] == ax_a && !orders_equal) {
      // Transpose/Pack
      std::array<int64_t, 3> extents;
      std::array<int64_t, 3> extents_h;
      std::array<int64_t, 3> extents_h_b;
      std::array<int64_t, 3> strides_in{1, 0, 0}, strides_out{1, 0, 0};
      std::array<int, 3> order;

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (pinfo_a.order[j] == pinfo_b.order[i]) {
            order[i] = j;
            break;
          }
        }
      }

      for (int i = 0; i < 3; ++i) {
        extents[i] = shape_g_a[pinfo_a.order[i]];
        extents_h[i] = shape_g_a_h[pinfo_a.order[i]];
        extents_h_b[i] = shape_g_b_h[pinfo_a.order[i]];
      }

      for (int i = 1; i < 3; ++i) {
        strides_in[i] = strides_in[i - 1] * extents_h[i - 1];
        if (!direct_transpose) {
          strides_out[i] = strides_out[i - 1] * extents[order[i - 1]];
        } else {
          strides_out[i] = strides_out[i - 1] * extents_h_b[order[i - 1]];
        }
      }

      if (pipelined) {
        auto dtype = getCudecompDataType<T>();
        auto key = std::tie(i1, o1, ax, dir, pinfo_a_h, pinfo_b_h, dtype);

        if (handle->cuda_graphs_enable && grid_desc->graph_cache.cached(key)) {
          grid_desc->graph_cache.replay(key, stream);
        } else {
          cudaStream_t graph_stream = stream;
          if (handle->cuda_graphs_enable && splits_a.size() > 1) {
            graph_stream = grid_desc->graph_cache.startCapture(key, stream);
          }

          for (int j = 1; j < splits_a.size() + 1; ++j) {
            int src_rank, dst_rank;
            getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
            if (j == splits_a.size()) dst_rank = comm_rank;

            size_t shift = offsets_a[dst_rank];
            for (int i = 0; i < 3; ++i) {
              if (pinfo_a_h.order[i] == ax_a) break;
              shift *= shape_g_a_h[pinfo_a_h.order[i]];
            }

            T* src = i1 + shift + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
            T* dst;
            if (!direct_transpose) {
              dst = o1 + send_offsets[dst_rank];
            } else {
              size_t shift_b = offsets_b[src_rank];
              for (int i = 0; i < 3; ++i) {
                if (pinfo_b_h.order[i] == ax_b) break;
                shift *= shape_g_b_h[pinfo_b_h.order[i]];
              }

              dst = o1 + shift + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
            }

            for (int i = 0; i < 3; ++i) {
              if (ax_a == pinfo_a.order[i]) extents[i] = splits_a[dst_rank];
            }

            localPermute(handle, extents, order, strides_in, strides_out, src, dst, graph_stream);
#if CUDART_VERSION >= 11010
            cudaStreamCaptureStatus capture_status;
            CHECK_CUDA(cudaStreamIsCapturing(graph_stream, &capture_status));
            CHECK_CUDA(cudaEventRecordWithFlags(grid_desc->events[dst_rank], graph_stream,
                                                capture_status == cudaStreamCaptureStatusActive
                                                    ? cudaEventRecordExternal
                                                    : cudaEventRecordDefault));
#else
            CHECK_CUDA(cudaEventRecord((grid_desc->events[dst_rank], graph_stream));
#endif
          }

          if (handle->cuda_graphs_enable && splits_a.size() > 1) {
            grid_desc->graph_cache.endCapture(key);
            grid_desc->graph_cache.replay(key, stream);
          }
        }
      } else {
        T* src = i1 + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
        T* dst;
        if (!direct_transpose) {
          dst = o1;
        } else {
          dst = o1 + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
        }
        localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
      }

      data_transposed = true;

    } else {
      // Pack
      int memcpy_count = 0;
      cudecompBatchedD2DMemcpy3DParams<T> memcpy_params;

      auto dtype = getCudecompDataType<T>();
      auto key = std::tie(i1, o1, ax, dir, pinfo_a_h, pinfo_b_h, dtype);

      if (handle->cuda_graphs_enable && grid_desc->graph_cache.cached(key)) {
        grid_desc->graph_cache.replay(key, stream);
      } else {
        cudaStream_t graph_stream = stream;
        if (handle->cuda_graphs_enable && pipelined && splits_a.size() > 1) {
          graph_stream = grid_desc->graph_cache.startCapture(key, stream);
        }

        for (int j = 1; j < splits_a.size() + 1; ++j) {
          int src_rank, dst_rank;
          getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
          if (j == splits_a.size()) dst_rank = comm_rank;

          size_t shift = offsets_a[dst_rank];
          for (int i = 0; i < 3; ++i) {
            if (pinfo_a_h.order[i] == ax_a) break;
            shift *= shape_g_a_h[pinfo_a_h.order[i]];
          }

          T* src = i1 + shift + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
          T* dst;
          if (!direct_pack) {
            dst = o1 + send_offsets[dst_rank];
          } else {
            size_t shift_b = offsets_b[src_rank];
            for (int i = 0; i < 3; ++i) {
              if (pinfo_b_h.order[i] == ax_b) break;
              shift_b *= shape_g_b_h[pinfo_b_h.order[i]];
            }
            dst = o1 + shift_b + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
          }

          memcpy_params.src[memcpy_count] = src;
          memcpy_params.dest[memcpy_count] = dst;
          memcpy_params.src_strides[0][memcpy_count] = pinfo_a_h.shape[0] * pinfo_a_h.shape[1];
          memcpy_params.src_strides[1][memcpy_count] = pinfo_a_h.shape[0];
          if (!direct_pack) {
            memcpy_params.dest_strides[1][memcpy_count] =
                (ax_a == pinfo_a.order[0]) ? splits_a[dst_rank] : pinfo_a.shape[0];
            memcpy_params.dest_strides[0][memcpy_count] =
                memcpy_params.dest_strides[1][memcpy_count] *
                ((ax_a == pinfo_a.order[1]) ? splits_a[dst_rank] : pinfo_a.shape[1]);
          } else {
            memcpy_params.dest_strides[0][memcpy_count] = pinfo_b_h.shape[0] * pinfo_b_h.shape[1];
            memcpy_params.dest_strides[1][memcpy_count] = pinfo_b_h.shape[0];
          }
          memcpy_params.extents[2][memcpy_count] = (ax_a == pinfo_a.order[0]) ? splits_a[dst_rank] : pinfo_a.shape[0];
          memcpy_params.extents[1][memcpy_count] = (ax_a == pinfo_a.order[1]) ? splits_a[dst_rank] : pinfo_a.shape[1];
          memcpy_params.extents[0][memcpy_count] = (ax_a == pinfo_a.order[2]) ? splits_a[dst_rank] : pinfo_a.shape[2];
          memcpy_count++;
          if (memcpy_count == memcpy_limit || j == splits_a.size()) {
            memcpy_params.ncopies = memcpy_count;
            cudecomp_batched_d2d_memcpy_3d(memcpy_params, graph_stream);
            memcpy_count = 0;
          }
#if CUDART_VERSION >= 11010
          if (pipelined) {
            cudaStreamCaptureStatus capture_status;
            CHECK_CUDA(cudaStreamIsCapturing(graph_stream, &capture_status));
            CHECK_CUDA(cudaEventRecordWithFlags(grid_desc->events[dst_rank], graph_stream,
                                                capture_status == cudaStreamCaptureStatusActive
                                                    ? cudaEventRecordExternal
                                                    : cudaEventRecordDefault));
          }
#else
          if (pipelined) CHECK_CUDA(cudaEventRecord((grid_desc->events[dst_rank], graph_stream));
#endif
        }
        if (handle->cuda_graphs_enable && pipelined && splits_a.size() > 1) {
          grid_desc->graph_cache.endCapture(key);
          grid_desc->graph_cache.replay(key, stream);
        }
      }
    }

    if (o1 == output) {
      // o1 is output. Return.
      if (handle->performance_report_enable) {
        CHECK_CUDA(cudaEventRecord(current_sample->transpose_end_event, stream));
        advanceTransposePerformanceSample(handle, grid_desc, createTransposeConfig(ax, dir, input, output, input_halo_extents.data(), output_halo_extents.data(), input_padding.data(), output_padding.data(), getCudecompDataType<T>()));
      }
      return;
    }
  } else {
    // For special cases that skip packing and are pipelined, need to record events to
    // enforce input data dependency
    if (pipelined) {
      for (int j = 0; j < splits_a.size(); ++j) {
        CHECK_CUDA(cudaEventRecord(grid_desc->events[j], stream));
      }
    }
  }

  // Communicate
  if (splits_a.size() > 1) {
    if (!pipelined) {
      cudecompAlltoall(handle, grid_desc, o1, send_counts, send_offsets, o2, recv_counts, recv_offsets,
                       recv_offsets_nvshmem, comm_axis, stream, current_sample);
    }
  } else {
    o2 = o1;
  }

  // Unpack into output buffer
  if (!data_transposed && !orders_equal) {
    if (pinfo_a.order[2] == ax_b || splits_a.size() == 1) {
      // Transpose/Unpack
      // Note: routing all single cases to this branch since single rank split transpose
      // is equivalent
      std::array<int64_t, 3> extents;
      std::array<int64_t, 3> extents_h;
      std::array<int64_t, 3> extents_h_a;
      std::array<int64_t, 3> strides_in{1, 0, 0}, strides_out{1, 0, 0};
      std::array<int, 3> order;

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (pinfo_a.order[j] == pinfo_b.order[i]) {
            order[i] = j;
            break;
          }
        }

        extents[i] = shape_g_b[pinfo_a.order[i]];
        extents_h[i] = shape_g_b_h[pinfo_b_h.order[i]];
        extents_h_a[i] = shape_g_a_h[pinfo_a_h.order[i]];
        if (i > 0) {
          if (!direct_transpose) {
            strides_in[i] = strides_in[i - 1] * extents[i - 1];
          } else {
            strides_in[i] = strides_in[i - 1] * extents_h_a[i - 1];
          }
          strides_out[i] = strides_out[i - 1] * extents_h[i - 1];
        }
      }

      if (pipelined) {
        bool nvshmem_synced = false;
        for (int j = 0; j < splits_b.size(); ++j) {
          int src_rank, dst_rank;
          getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
          if (j == 0) {
            dst_rank = comm_rank;
            src_rank = comm_rank;
          }

          std::vector<int> dst_ranks{dst_rank};
          std::vector<int> src_ranks{src_rank};

          if (j != 0 && comm_info.ngroups != 1) {
            // Perform pipelining in pairs to hide intra-group comms behind inter-group transfers
            if (j % 2 == 1) {
              if (j + 1 < splits_b.size()) {
                int src_rank_next, dst_rank_next;
                getAlltoallPeerRanks(grid_desc, comm_axis, j + 1, src_rank_next, dst_rank_next);
                dst_ranks.push_back(dst_rank_next);
                src_ranks.push_back(src_rank_next);
              }
            } else {
              // Skip alltoall, this transfer was paired with previous one.
              dst_ranks.resize(0);
              src_ranks.resize(0);
            }
          }

          if (o2 != o1) {
            cudecompAlltoallPipelined(handle, grid_desc, o1, send_counts, send_offsets, o2, recv_counts, recv_offsets,
                                      recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced, current_sample);
          }

          if (o2 != o3) {
            size_t shift = offsets_b[src_rank];
            for (int i = 0; i < 3; ++i) {
              if (pinfo_b_h.order[i] == ax_b) break;
              shift *= shape_g_b_h[pinfo_b_h.order[i]];
            }

            T* src;
            if (!direct_transpose) {
              src = o2 + recv_offsets[src_rank];
            } else {
              size_t shift_a = offsets_a[dst_rank];
              for (int i = 0; i < 3; ++i) {
                if (pinfo_a_h.order[i] == ax_a) break;
                shift_a *= shape_g_a_h[pinfo_a_h.order[i]];
              }
              src = o2 + shift_a + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
            }
            T* dst = o3 + shift + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
            for (int i = 0; i < 3; ++i) {
              if (ax_b == pinfo_a.order[i]) {
                extents[i] = splits_b[src_rank];
                break;
              }
            }

            localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
          }
        }
      } else {
        if (o2 != o3) {
          T* src;
          if (!direct_transpose) {
            src = o2;
          } else {
            src = o2 + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
          }
          T* dst = o3 + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
          localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
        }
      }
    } else {
      // Split Transpose/Unpack
      std::array<int64_t, 3> extents;
      std::array<int64_t, 3> extents_h;
      std::array<int64_t, 3> strides_in{1, 0, 0}, strides_out{1, 0, 0};
      std::array<int, 3> order;

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (pinfo_a.order[j] == pinfo_b.order[i]) {
            order[i] = j;
            break;
          }
        }

        extents[i] = shape_g_b[pinfo_a.order[i]];
        extents_h[i] = shape_g_b_h[pinfo_b_h.order[i]];
        if (i > 0) { strides_out[i] = strides_out[i - 1] * extents_h[i - 1]; }
      }

      bool nvshmem_synced = false;
      for (int j = 0; j < splits_b.size(); ++j) {
        int src_rank, dst_rank;
        getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
        if (j == 0) {
          dst_rank = comm_rank;
          src_rank = comm_rank;
        }

        if (pipelined) {
          std::vector<int> dst_ranks{dst_rank};
          std::vector<int> src_ranks{src_rank};

          if (j != 0 && comm_info.ngroups != 1) {
            // Perform pipelining in pairs to hide intra-group comms behind inter-group transfers
            if (j % 2 == 1) {
              if (j + 1 < splits_b.size()) {
                int src_rank_next, dst_rank_next;
                getAlltoallPeerRanks(grid_desc, comm_axis, j + 1, src_rank_next, dst_rank_next);
                dst_ranks.push_back(dst_rank_next);
                src_ranks.push_back(src_rank_next);
              }
            } else {
              // Skip alltoall, this transfer was paired with previous one.
              dst_ranks.resize(0);
              src_ranks.resize(0);
            }
          }

          if (o2 != o1) {
            cudecompAlltoallPipelined(handle, grid_desc, o1, send_counts, send_offsets, o2, recv_counts, recv_offsets,
                                      recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced, current_sample);
          }
        }

        if (o2 != o3) {
          for (int i = 0; i < 3; ++i) {
            if (ax_b == pinfo_a.order[i]) extents[i] = splits_b[src_rank];
            if (i > 0) { strides_in[i] = strides_in[i - 1] * extents[i - 1]; }
          }

          size_t shift = offsets_b[src_rank];
          for (int i = 0; i < 3; ++i) {
            if (pinfo_b_h.order[i] == ax_b) break;
            shift *= shape_g_b_h[pinfo_b_h.order[i]];
          }

          T* src = o2 + recv_offsets[src_rank];
          T* dst = o3 + shift + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
          localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
        }
      }
    }
  } else {
    // Unpack
    bool nvshmem_synced = false;
    int memcpy_count = 0;
    cudecompBatchedD2DMemcpy3DParams<T> memcpy_params;
    for (int j = 0; j < splits_a.size(); ++j) {
      int src_rank, dst_rank;
      getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
      if (j == 0) {
        dst_rank = comm_rank;
        src_rank = comm_rank;
      }

      if (pipelined) {
        std::vector<int> dst_ranks{dst_rank};
        std::vector<int> src_ranks{src_rank};

        if (j != 0 && comm_info.ngroups != 1) {
          // Perform pipelining in pairs to hide intra-group comms behind inter-group transfers
          if (j % 2 == 1) {
            if (j + 1 < splits_b.size()) {
              int src_rank_next, dst_rank_next;
              getAlltoallPeerRanks(grid_desc, comm_axis, j + 1, src_rank_next, dst_rank_next);
              dst_ranks.push_back(dst_rank_next);
              src_ranks.push_back(src_rank_next);
            }
          } else {
            // Skip alltoall, this transfer was paired with previous one.
            dst_ranks.resize(0);
            src_ranks.resize(0);
          }
        }

        if (o2 != o1) {
          cudecompAlltoallPipelined(handle, grid_desc, o1, send_counts, send_offsets, o2, recv_counts, recv_offsets,
                                    recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced, current_sample);
        }
      }

      if (o2 != o3) {
        size_t shift = offsets_b[src_rank];
        for (int i = 0; i < 3; ++i) {
          if (pinfo_b_h.order[i] == ax_b) break;
          shift *= shape_g_b_h[pinfo_b_h.order[i]];
        }

        T* src = o2 + recv_offsets[src_rank];
        T* dest = o3 + shift + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
        memcpy_params.src[memcpy_count] = src;
        memcpy_params.dest[memcpy_count] = dest;
        memcpy_params.src_strides[1][memcpy_count] = (ax_b == pinfo_b.order[0]) ? splits_b[src_rank] : pinfo_b.shape[0];
        memcpy_params.src_strides[0][memcpy_count] =
            memcpy_params.src_strides[1][memcpy_count] *
            ((ax_b == pinfo_b.order[1]) ? splits_b[src_rank] : pinfo_b.shape[1]);
        memcpy_params.dest_strides[0][memcpy_count] = pinfo_b_h.shape[0] * pinfo_b_h.shape[1];
        memcpy_params.dest_strides[1][memcpy_count] = pinfo_b_h.shape[0];
        memcpy_params.extents[2][memcpy_count] = (ax_b == pinfo_b.order[0]) ? splits_b[src_rank] : pinfo_b.shape[0];
        memcpy_params.extents[1][memcpy_count] = (ax_b == pinfo_b.order[1]) ? splits_b[src_rank] : pinfo_b.shape[1];
        memcpy_params.extents[0][memcpy_count] = (ax_b == pinfo_b.order[2]) ? splits_b[src_rank] : pinfo_b.shape[2];
        memcpy_count++;
        if (memcpy_count == memcpy_limit || j == splits_a.size() - 1) {
          memcpy_params.ncopies = memcpy_count;
          cudecomp_batched_d2d_memcpy_3d(memcpy_params, stream);
          memcpy_count = 0;
        }
      }
    }
  }

  if (handle->performance_report_enable) {
    // Record performance data
    CHECK_CUDA(cudaEventRecord(current_sample->transpose_end_event, stream));
    advanceTransposePerformanceSample(handle, grid_desc, createTransposeConfig(ax, dir, input, output, input_halo_extents.data(), output_halo_extents.data(), input_padding.data(), output_padding.data(), getCudecompDataType<T>()));
  }

}

template <typename T>
void cudecompTransposeXToY(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr,
                           const int32_t input_padding_ptr[] = nullptr, const int32_t output_padding_ptr[] = nullptr,
                           cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeXToY");
  cudecompTranspose_(0, 1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     input_padding_ptr, output_padding_ptr, stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeYToZ(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr,
                           const int32_t input_padding_ptr[] = nullptr, const int32_t output_padding_ptr[] = nullptr,
                           cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeYToZ");
  cudecompTranspose_(1, 1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     input_padding_ptr, output_padding_ptr, stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeZToY(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr,
                           const int32_t input_padding_ptr[] = nullptr, const int32_t output_padding_ptr[] = nullptr,
                           cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeZToY");
  cudecompTranspose_(2, -1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     input_padding_ptr, output_padding_ptr, stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeYToX(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr,
                           const int32_t input_padding_ptr[] = nullptr, const int32_t output_padding_ptr[] = nullptr,
                           cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeYToX");
  cudecompTranspose_(1, -1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     input_padding_ptr, output_padding_ptr, stream);
  nvtx::rangePop();
}

} // namespace cudecomp

#endif // TRANSPOSE_H
