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

#ifndef HALO_H
#define HALO_H

#include <array>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "internal/checks.h"
#include "internal/comm_routines.h"
#include "internal/cudecomp_kernels.h"
#include "internal/nvtx.h"

namespace cudecomp {

template <typename T>
void cudecompUpdateHalos_(int ax, const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* work,
                          const int32_t halo_extents_ptr[], const bool halo_periods_ptr[], int32_t dim,
                          cudaStream_t stream) {
  std::array<int32_t, 6> halo_extents{};
  if (halo_extents_ptr) std::copy(halo_extents_ptr, halo_extents_ptr + 6, halo_extents.begin());
  std::array<bool, 3> halo_periods{};
  if (halo_periods_ptr) std::copy(halo_periods_ptr, halo_periods_ptr + 3, halo_periods.begin());

  // Get pencil info
  cudecompPencilInfo_t pinfo_h;
  CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo_h, ax, halo_extents.data()));

  // Get global ordered shapes
  auto shape_g_h = getShapeG(pinfo_h);

  // Get neighbors
  std::array<int32_t, 2> neighbors;
  CHECK_CUDECOMP(cudecompGetShiftedRank(handle, grid_desc, ax, dim, -1, halo_periods[dim], &neighbors[0]));
  CHECK_CUDECOMP(cudecompGetShiftedRank(handle, grid_desc, ax, dim, 1, halo_periods[dim], &neighbors[1]));

  // Quick return if no halos
  if (halo_extents[dim] == 0 && halo_extents[dim + 3] == 0) { return; }

  // Select correct case based on pencil memory order and transfer dim
  int c;
  if (dim != pinfo_h.order[0] && dim != pinfo_h.order[1]) {
    c = 2;
  } else {
    c = 1;
  }

  if (neighbors[0] == handle->rank && neighbors[1] == handle->rank) {
    // Single rank in this dimension and periodic. Use periodic self-copy case.
    c = 0;
  } else if (neighbors[0] == -1 && neighbors[1] == -1) {
    // Single rank in this dimension and not periodic. Return.
    return;
  }

  bool managed = isManagedPointer(input);
  if (c == 2 && managed &&
      (haloBackendRequiresNvshmem(grid_desc->config.halo_comm_backend) ||
       haloBackendRequiresMpi(grid_desc->config.halo_comm_backend))) {
    // For managed memory, always stage to work space if using MPI/NVSHMEM
    // Can revisit for NVSHMEM if input is nvshmem allocated
    c = 1;
  }

  switch (c) {
  case 0: {
    // Periodic self-copy case
    cudecompBatchedD2DMemcpy3DParams<T> memcpy_params;
    std::array<int32_t, 3> lx{};

    // Left
    lx[dim] = shape_g_h[dim] - halo_extents[dim] - halo_extents[dim + 3];
    memcpy_params.src[0] = input + getPencilPtrOffset(pinfo_h, lx);
    memcpy_params.dest[0] = input + getPencilPtrOffset(pinfo_h, {0, 0, 0});

    // Right
    lx[dim] = halo_extents[dim];
    memcpy_params.src[1] = input + getPencilPtrOffset(pinfo_h, lx);
    lx[dim] = shape_g_h[dim] - halo_extents[dim + 3];
    memcpy_params.dest[1] = input + getPencilPtrOffset(pinfo_h, lx);

    for (int i = 0; i < 2; ++i) {
      memcpy_params.src_strides[0][i] = pinfo_h.shape[0] * pinfo_h.shape[1];
      memcpy_params.src_strides[1][i] = pinfo_h.shape[0];
      memcpy_params.dest_strides[0][i] = pinfo_h.shape[0] * pinfo_h.shape[1];
      memcpy_params.dest_strides[1][i] = pinfo_h.shape[0];
      memcpy_params.extents[0][i] = (dim == pinfo_h.order[2]) ? halo_extents[dim + 3*i] : pinfo_h.shape[2];
      memcpy_params.extents[1][i] = (dim == pinfo_h.order[1]) ? halo_extents[dim + 3*i] : pinfo_h.shape[1];
      memcpy_params.extents[2][i] = (dim == pinfo_h.order[0]) ? halo_extents[dim + 3*i] : pinfo_h.shape[0];
    }

    memcpy_params.ncopies = 2;
    cudecomp_batched_d2d_memcpy_3d(memcpy_params, stream);
  } break;

  case 1: {
    // Strided halo (pack, send/recv, unpack)
    cudecompBatchedD2DMemcpy3DParams<T> memcpy_params;
    std::array<int32_t, 3> lx{};

    size_t halo_size_l = shape_g_h[(dim + 1) % 3] * shape_g_h[(dim + 2) % 3] * halo_extents[dim];
    size_t halo_size_r = shape_g_h[(dim + 1) % 3] * shape_g_h[(dim + 2) % 3] * halo_extents[dim + 3];
    T* send_buff = work;
    T* recv_buff = work + halo_size_l + halo_size_r;

    // Pack
    // Left
    lx[dim] = halo_extents[dim];
    memcpy_params.src[0] = input + getPencilPtrOffset(pinfo_h, lx);
    memcpy_params.dest[0] = send_buff;

    // Right
    lx[dim] = shape_g_h[dim] - halo_extents[dim] - halo_extents[dim + 3];
    memcpy_params.src[1] = input + getPencilPtrOffset(pinfo_h, lx);
    memcpy_params.dest[1] = send_buff + halo_size_r;

    for (int i = 0; i < 2; ++i) {
      memcpy_params.src_strides[0][i] = pinfo_h.shape[0] * pinfo_h.shape[1];
      memcpy_params.src_strides[1][i] = pinfo_h.shape[0];
      memcpy_params.dest_strides[1][i] = (dim == pinfo_h.order[0]) ? halo_extents[dim + 3*((i+1)%2)] : pinfo_h.shape[0];
      memcpy_params.dest_strides[0][i] =
          memcpy_params.dest_strides[1][i] * ((dim == pinfo_h.order[1]) ? halo_extents[dim + 3*((i+1)%2)] : pinfo_h.shape[1]);
      memcpy_params.extents[0][i] = (dim == pinfo_h.order[2]) ? halo_extents[dim + 3*((i+1)%2)] : pinfo_h.shape[2];
      memcpy_params.extents[1][i] = (dim == pinfo_h.order[1]) ? halo_extents[dim + 3*((i+1)%2)] : pinfo_h.shape[1];
      memcpy_params.extents[2][i] = (dim == pinfo_h.order[0]) ? halo_extents[dim + 3*((i+1)%2)] : pinfo_h.shape[0];
    }

    memcpy_params.ncopies = 2;
    cudecomp_batched_d2d_memcpy_3d(memcpy_params, stream);

    std::array<comm_count_t, 2> send_counts{static_cast<comm_count_t>(halo_size_r), static_cast<comm_count_t>(halo_size_l)};
    std::array<comm_count_t, 2> recv_counts{static_cast<comm_count_t>(halo_size_l), static_cast<comm_count_t>(halo_size_r)};
    std::array<size_t, 2> send_offsets{};
    std::array<size_t, 2> recv_offsets{};
    send_offsets[1] = halo_size_r;
    recv_offsets[1] = halo_size_l;

    cudecompSendRecvPair(handle, grid_desc, neighbors, send_buff, send_counts, send_offsets, recv_buff, recv_counts, recv_offsets, stream);

    // Unpack
    // Left
    memcpy_params.src[0] = recv_buff;
    memcpy_params.dest[0] = input + getPencilPtrOffset(pinfo_h, {0, 0, 0});

    // Right
    memcpy_params.src[1] = recv_buff + halo_size_l;
    lx[dim] = shape_g_h[dim] - halo_extents[dim + 3];
    memcpy_params.dest[1] = input + getPencilPtrOffset(pinfo_h, lx);

    for (int i = 0; i < 2; ++i) {
      memcpy_params.dest_strides[0][i] = pinfo_h.shape[0] * pinfo_h.shape[1];
      memcpy_params.dest_strides[1][i] = pinfo_h.shape[0];
      memcpy_params.src_strides[1][i] = (dim == pinfo_h.order[0]) ? halo_extents[dim + 3*i] : pinfo_h.shape[0];
      memcpy_params.src_strides[0][i] =
          memcpy_params.src_strides[1][i] * ((dim == pinfo_h.order[1]) ? halo_extents[dim + 3*i] : pinfo_h.shape[1]);
      memcpy_params.extents[0][i] = (dim == pinfo_h.order[2]) ? halo_extents[dim + 3*i] : pinfo_h.shape[2];
      memcpy_params.extents[1][i] = (dim == pinfo_h.order[1]) ? halo_extents[dim + 3*i] : pinfo_h.shape[1];
      memcpy_params.extents[2][i] = (dim == pinfo_h.order[0]) ? halo_extents[dim + 3*i] : pinfo_h.shape[0];
    }

    if (neighbors[0] == -1) {
      // Left is non-periodic, unpack right only
      memcpy_params.src[0] = memcpy_params.src[1];
      memcpy_params.dest[0] = memcpy_params.dest[1];
      memcpy_params.src_strides[0][0] = memcpy_params.src_strides[0][1];
      memcpy_params.src_strides[1][0] = memcpy_params.src_strides[1][1];
      memcpy_params.dest_strides[0][0] = memcpy_params.dest_strides[0][1];
      memcpy_params.dest_strides[1][0] = memcpy_params.dest_strides[1][1];
      memcpy_params.extents[0][0] = memcpy_params.extents[0][1];
      memcpy_params.extents[1][0] = memcpy_params.extents[1][1];
      memcpy_params.extents[2][0] = memcpy_params.extents[2][1];
      memcpy_params.ncopies = 1;
    } else if (neighbors[1] == -1) {
      // Right is non-periodic, unpack left only
      memcpy_params.ncopies = 1;
    } else {
      memcpy_params.ncopies = 2;
    }

    cudecomp_batched_d2d_memcpy_3d(memcpy_params, stream);
  } break;

  case 2: {
    // Contiguous (direct send/recv)
    std::array<int32_t, 3> lx{};

    size_t halo_size_l = shape_g_h[(dim + 1) % 3] * shape_g_h[(dim + 2) % 3] * halo_extents[dim];
    size_t halo_size_r = shape_g_h[(dim + 1) % 3] * shape_g_h[(dim + 2) % 3] * halo_extents[dim + 3];
    std::array<comm_count_t, 2> send_counts{static_cast<comm_count_t>(halo_size_r), static_cast<comm_count_t>(halo_size_l)};
    std::array<comm_count_t, 2> recv_counts{static_cast<comm_count_t>(halo_size_l), static_cast<comm_count_t>(halo_size_r)};
    std::array<size_t, 2> send_offsets;
    std::array<size_t, 2> recv_offsets;

    // Left
    lx[dim] = halo_extents[dim];
    send_offsets[0] = getPencilPtrOffset(pinfo_h, lx);
    recv_offsets[0] = getPencilPtrOffset(pinfo_h, {0, 0, 0});
    // Right
    lx[dim] = shape_g_h[dim] - halo_extents[dim] - halo_extents[dim + 3];
    send_offsets[1] = getPencilPtrOffset(pinfo_h, lx);
    lx[dim] = shape_g_h[dim] - halo_extents[dim + 3];
    recv_offsets[1] = getPencilPtrOffset(pinfo_h, lx);

    cudecompSendRecvPair(handle, grid_desc, neighbors, input, send_counts, send_offsets, input, recv_counts, recv_offsets,
                         stream);
  } break;
  }
}

template <typename T>
void cudecompUpdateHalosX(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* work,
                          const int32_t halo_extents_ptr[], const bool halo_periods_ptr[], int32_t dim,
                          cudaStream_t stream) {
  std::stringstream os;
  os << "cudecompUpdateHalosX_" << dim;
  nvtx::rangePush(os.str());
  cudecompUpdateHalos_(0, handle, grid_desc, input, work, halo_extents_ptr, halo_periods_ptr, dim, stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompUpdateHalosY(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* work,
                          const int32_t halo_extents_ptr[], const bool halo_periods_ptr[], int32_t dim,
                          cudaStream_t stream) {
  std::stringstream os;
  os << "cudecompUpdateHalosY_" << dim;
  nvtx::rangePush(os.str());
  cudecompUpdateHalos_(1, handle, grid_desc, input, work, halo_extents_ptr, halo_periods_ptr, dim, stream);
  nvtx::rangePop();
}

template <typename T>
void cudecompUpdateHalosZ(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc, T* input, T* work,
                          const int32_t halo_extents_ptr[], const bool halo_periods_ptr[], int32_t dim,
                          cudaStream_t stream) {
  std::stringstream os;
  os << "cudecompUpdateHalosZ_" << dim;
  nvtx::rangePush(os.str());
  cudecompUpdateHalos_(2, handle, grid_desc, input, work, halo_extents_ptr, halo_periods_ptr, dim, stream);
  nvtx::rangePop();
}

} // namespace cudecomp

#endif // HALO_H
