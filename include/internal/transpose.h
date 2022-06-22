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
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <mpi.h>

#include "internal/checks.h"
#include "internal/comm_routines.h"
#include "internal/cudecomp_kernels.h"
#include "internal/nvtx.h"

namespace cudecomp {

static inline std::vector<int64_t> getSplits(int64_t N, int nchunks, int pad) {
  std::vector<int64_t> splits(nchunks, N / nchunks);
  for (int i = 0; i < N % nchunks; ++i) { splits[i] += 1; }

  // Add padding to last populated pencil
  splits[std::min(N, (int64_t)nchunks) - 1] += pad;

  return splits;
}

static inline bool isTransposeCommPipelined(cudecompTransposeCommBackend_t commType) {
  return (commType == CUDECOMP_TRANSPOSE_COMM_NCCL_PL ||
#ifdef ENABLE_NVSHMEM
          commType == CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL ||
#endif
          commType == CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL);
}

template <typename T> static inline bool anyNonzeros(const std::array<T, 3>& arr) {
  return (arr[0] != T(0) || arr[1] != T(0) || arr[2] != T(0));
}

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

template <typename T>
static void cudecompTranspose_(int ax, int dir, const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                               T* input, T* output, T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                               const int32_t output_halo_extents_ptr[] = nullptr, cudaStream_t stream = 0) {

  std::array<int32_t, 3> input_halo_extents{};
  std::array<int32_t, 3> output_halo_extents{};
  if (input_halo_extents_ptr) std::copy(input_halo_extents_ptr, input_halo_extents_ptr + 3, input_halo_extents.begin());
  if (output_halo_extents_ptr)
    std::copy(output_halo_extents_ptr, output_halo_extents_ptr + 3, output_halo_extents.begin());

  bool fwd = dir > 0;

  bool inplace = (input == output);
  bool input_has_halos = anyNonzeros(input_halo_extents);
  bool output_has_halos = anyNonzeros(output_halo_extents);
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

  bool power_of_2 = !(splits_a.size() & splits_a.size() - 1);

  // Get pencil info
  cudecompPencilInfo_t pinfo_a, pinfo_a_h;
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_a, ax_a, nullptr));
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_a_h, ax_a, input_halo_extents.data()));
  cudecompPencilInfo_t pinfo_b, pinfo_b_h;
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_b, ax_b, nullptr));
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_b_h, ax_b, output_halo_extents.data()));

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
  }

  // Adjust pointers to handle special cases
  if (!input_has_halos && !output_has_halos) {
    if (splits_a.size() == 1) {
      if (grid_desc->config.transpose_axis_contiguous[ax_a]) {
        if (grid_desc->config.transpose_axis_contiguous[ax_b]) {
          if (fwd && !inplace) {
            // Single rank, out of place: Transpose directly to output and return
            o1 = output;
          } else if (!fwd && !inplace) {
            // Single rank, out of place: Skip pack, transpose directly to output
            o1 = input;
            o2 = input;
          }
        } else {
          if (!inplace) {
            // Single rank, out of place: Skip pack, transpose directly to output
            o1 = input;
            o2 = input;
          }
        }
      } else {
        if (grid_desc->config.transpose_axis_contiguous[ax_b]) {
          if (!inplace && fwd) {
            // Single rank, out of place: Transpose directly to output and return
            o1 = output;
          } else if (!inplace && !fwd) {
            // Single rank, out of place: Skip pack, transpose directly to output
            o1 = input;
            o2 = input;
          }
        } else {
          if (inplace) {
            // Single rank, in place: No transpose necessary.
            return;
          } else {
            // Single rank, out of place: Pack directly to output and return
            o1 = output;
          }
        }
      }
    } else {
      if (!grid_desc->config.transpose_axis_contiguous[ax_a]) {
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
          if (fwd && ax_a == 1 && !grid_desc->config.transpose_axis_contiguous[ax_b]) { // Y2Z
            // Output of all to all is in correct orientation, skip unpack
            o2 = output;
          } else if (!fwd && ax_a == 2) { // Z2Y
            // Input is already packed for all to all, skip pack
            o1 = input;
            o2 = work;
          }
        }
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

  if (o1 != i1) {
    if (fwd && grid_desc->config.transpose_axis_contiguous[ax_b]) {
      // Transpose
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
      }

      for (int i = 0; i < 3; ++i) {
        extents[i] = shape_g_a[pinfo_a.order[i]];
        extents_h[i] = shape_g_a_h[pinfo_a.order[i]];
      }

      for (int i = 1; i < 3; ++i) {
        strides_in[i] = strides_in[i - 1] * extents_h[i - 1];
        strides_out[i] = strides_out[i - 1] * extents[order[i - 1]];
      }

      if (pipelined) {
        for (int j = 1; j < splits_a.size() + 1; ++j) {
          int src_rank, dst_rank;
          getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
          if (j == splits_a.size()) dst_rank = comm_rank;

          size_t shift = offsets_a[dst_rank];
          for (int i = 0; i < 3; ++i) {
            if (pinfo_a_h.order[i] == ax_a) break;
            shift *= shape_g_a_h[i];
          }

          T* src = i1 + shift + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
          T* dst = o1 + send_offsets[dst_rank];
          for (int i = 0; i < 3; ++i) {
            if (ax_a == pinfo_a.order[i]) extents[i] = splits_a[dst_rank];
          }
          localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
          CHECK_CUDA(cudaEventRecord(grid_desc->events[dst_rank], stream));
        }
      } else {
        T* src = i1 + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
        T* dst = o1;
        localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
      }

    } else {
      // Pack
      int memcpy_count = 0;
      cudecompBatchedD2DMemcpy3DParams<T> memcpy_params;
      for (int j = 1; j < splits_a.size() + 1; ++j) {
        int src_rank, dst_rank;
        getAlltoallPeerRanks(grid_desc, comm_axis, j, src_rank, dst_rank);
        if (j == splits_a.size()) dst_rank = comm_rank;

        size_t shift = offsets_a[dst_rank];
        for (int i = 0; i < 3; ++i) {
          if (pinfo_a_h.order[i] == ax_a) break;
          shift *= shape_g_a_h[i];
        }

        T* src = i1 + shift + getPencilPtrOffset(pinfo_a_h, input_halo_extents);
        T* dst = o1 + send_offsets[dst_rank];

        memcpy_params.src[memcpy_count] = src;
        memcpy_params.dest[memcpy_count] = dst;
        memcpy_params.src_strides[0][memcpy_count] = pinfo_a_h.shape[0] * pinfo_a_h.shape[1];
        memcpy_params.src_strides[1][memcpy_count] = pinfo_a_h.shape[0];
        memcpy_params.dest_strides[1][memcpy_count] =
            (ax_a == pinfo_a.order[0]) ? splits_a[dst_rank] : pinfo_a.shape[0];
        memcpy_params.dest_strides[0][memcpy_count] =
            memcpy_params.dest_strides[1][memcpy_count] *
            ((ax_a == pinfo_a.order[1]) ? splits_a[dst_rank] : pinfo_a.shape[1]);
        memcpy_params.extents[2][memcpy_count] = (ax_a == pinfo_a.order[0]) ? splits_a[dst_rank] : pinfo_a.shape[0];
        memcpy_params.extents[1][memcpy_count] = (ax_a == pinfo_a.order[1]) ? splits_a[dst_rank] : pinfo_a.shape[1];
        memcpy_params.extents[0][memcpy_count] = (ax_a == pinfo_a.order[2]) ? splits_a[dst_rank] : pinfo_a.shape[2];
        memcpy_count++;
        if (memcpy_count == memcpy_limit || j == splits_a.size()) {
          memcpy_params.ncopies = memcpy_count;
          cudecomp_batched_d2d_memcpy_3d(memcpy_params, stream);
          memcpy_count = 0;
        }
        if (pipelined) CHECK_CUDA(cudaEventRecord(grid_desc->events[dst_rank], stream));
      }
    }

    if (o1 == output) {
      // o1 is output. Return.
      return;
    }
  } else {
    // For special cases that skip packing and are pipelined, need to record events to
    // enforce input data dependency
    if (pipelined) {
      for (int j = 0; j < splits_a.size(); ++j) { CHECK_CUDA(cudaEventRecord(grid_desc->events[j], stream)); }
    }
  }

  // Communicate
  if (splits_a.size() > 1) {
    cudecompAlltoall(handle, grid_desc, o1, send_counts, send_offsets, o2, recv_counts, recv_offsets,
                     recv_offsets_nvshmem, comm_axis, stream);
  } else {
    o2 = o1;
  }

  // Unpack into output buffer
  if (!fwd && grid_desc->config.transpose_axis_contiguous[ax_a]) {
    // Transpose out
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
      if (i > 0) {
        strides_in[i] = strides_in[i - 1] * extents[i - 1];
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

        if (j != 0 && comm_info.homogeneous && comm_info.nnodes != 1) {
          // Perform pipelining in pairs to intra-node comms behind inter-node transfers
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
                                    recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced);
        }

        if (o2 != o3) {
          size_t shift = offsets_b[src_rank];
          for (int i = 0; i < 3; ++i) {
            if (pinfo_b_h.order[i] == ax_b) break;
            shift *= shape_g_b_h[i];
          }

          T* src = o2 + recv_offsets[src_rank];
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
        T* src = o2;
        T* dst = o3 + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
        localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
      }
    }
  } else if ((!fwd && grid_desc->config.transpose_axis_contiguous[ax_b]) ||
             (fwd && grid_desc->config.transpose_axis_contiguous[ax_a] &&
              !grid_desc->config.transpose_axis_contiguous[ax_b])) {
    // Split transpose
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

      std::vector<int> dst_ranks{dst_rank};
      std::vector<int> src_ranks{src_rank};

      if (j != 0 && comm_info.homogeneous && comm_info.nnodes != 1) {
        // Perform pipelining in pairs to intra-node comms behind inter-node transfers
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
                                  recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced);
      }

      if (o2 != o3) {
        for (int i = 0; i < 3; ++i) {
          if (ax_b == pinfo_a.order[i]) extents[i] = splits_b[src_rank];
          if (i > 0) { strides_in[i] = strides_in[i - 1] * extents[i - 1]; }
        }

        size_t shift = offsets_b[src_rank];
        for (int i = 0; i < 3; ++i) {
          if (pinfo_b_h.order[i] == ax_b) break;
          shift *= shape_g_b_h[i];
        }

        T* src = o2 + recv_offsets[src_rank];
        T* dst = o3 + shift + getPencilPtrOffset(pinfo_b_h, output_halo_extents);
        localPermute(handle, extents, order, strides_in, strides_out, src, dst, stream);
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

      std::vector<int> dst_ranks{dst_rank};
      std::vector<int> src_ranks{src_rank};

      if (j != 0 && comm_info.homogeneous && comm_info.nnodes != 1) {
        // Perform pipelining in pairs to intra-node comms behind inter-node transfers
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
                                  recv_offsets_nvshmem, comm_axis, src_ranks, dst_ranks, stream, nvshmem_synced);
      }

      if (o2 != o3) {
        size_t shift = offsets_b[src_rank];
        for (int i = 0; i < 3; ++i) {
          if (pinfo_b_h.order[i] == ax_b) break;
          shift *= shape_g_b_h[i];
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
}

template <typename T>
void cudecompTransposeXToY(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeXToY");
  cudecompTranspose_(0, 1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeYToZ(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeYToZ");
  cudecompTranspose_(1, 1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeZToY(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeZToY");
  cudecompTranspose_(2, -1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompTransposeYToX(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* output,
                           T* work, const int32_t input_halo_extents_ptr[] = nullptr,
                           const int32_t output_halo_extents_ptr[] = nullptr, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompTransposeYToX");
  cudecompTranspose_(1, -1, handle, grid_desc, input, output, work, input_halo_extents_ptr, output_halo_extents_ptr,
                     stream);
  nvtx::rangePop();
}

} // namespace cudecomp

#endif // TRANSPOSE_H
