/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMM_ROUTINES_H
#define COMM_ROUTINES_H

#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "internal/checks.h"
#include "internal/cudecomp_kernels.h"
#include "internal/nvtx.h"

namespace cudecomp {

static inline MPI_Datatype getMpiDataType(float) { return MPI_FLOAT; }
static inline MPI_Datatype getMpiDataType(double) { return MPI_DOUBLE; }
static inline MPI_Datatype getMpiDataType(cuda::std::complex<float>) { return MPI_C_FLOAT_COMPLEX; }
static inline MPI_Datatype getMpiDataType(cuda::std::complex<double>) { return MPI_C_DOUBLE_COMPLEX; }
template <typename T> static inline MPI_Datatype getMpiDataType() { return getMpiDataType(T(0)); }

static inline bool canUseMpiAlltoall(const std::vector<comm_count_t>& send_counts,
                                     const std::vector<comm_count_t>& send_offsets,
                                     const std::vector<comm_count_t>& recv_counts,
                                     const std::vector<comm_count_t>& recv_offsets) {
  auto scount = send_counts[0];
  auto rcount = recv_counts[0];
  // Check that send and recv counts are constants
  for (int i = 1; i < send_counts.size(); ++i) {
    if (send_counts[i] != scount) { return false; }
  }
  for (int i = 1; i < recv_counts.size(); ++i) {
    if (recv_counts[i] != rcount) { return false; }
  }

  // Check that offsets are contiguous and equal to counts
  for (int i = 0; i < send_offsets.size(); ++i) {
    if (send_offsets[i] != i * scount) { return false; }
  }
  for (int i = 0; i < recv_offsets.size(); ++i) {
    if (recv_offsets[i] != i * rcount) { return false; }
  }

  return true;
}

static inline void checkMpiInt32Limit(int64_t val, cudecompTransposeCommBackend_t backend) {
  if (val > std::numeric_limits<std::int32_t>::max()) {
    std::ostringstream os;
    os << "MPI count and/or offset argument exeeding int32_t limit in ";
    os << cudecompTransposeCommBackendToString(backend);
    os << " transpose backend.";
    std::string str = os.str();
    THROW_NOT_SUPPORTED(str.c_str());
  }
}

static inline void checkMpiInt32Limit(int64_t val, cudecompHaloCommBackend_t backend) {
  if (val > std::numeric_limits<std::int32_t>::max()) {
    std::ostringstream os;
    os << "MPI count and/or offset argument exeeding int32_t limit in ";
    os << cudecompHaloCommBackendToString(backend);
    os << " halo backend.";
    std::string str = os.str();
    THROW_NOT_SUPPORTED(str.c_str());
  }
}

#ifdef ENABLE_NVSHMEM
#define CUDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ 8 // max number of intra-group transfers to schedule between team syncs
template <typename T>
static void
nvshmemAlltoallV(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                 const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                 T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                 const std::vector<comm_count_t>& recv_offsets, cudecompCommAxis comm_axis, cudaStream_t stream) {
  auto& comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
  auto team = comm_info.nvshmem_team;
  int self_rank = comm_info.rank;

  nvshmemx_team_sync_on_stream(team, stream);

  // Event dependency on external stream for intra-group transfers
  CHECK_CUDA(cudaEventRecord(grid_desc->events[0], stream));
  for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
    CHECK_CUDA(cudaStreamWaitEvent(handle->streams[i], grid_desc->events[0], 0));
  }

  cudecompNvshmemA2AParams<T> params;

  // Inter-group transfers (non-blocking)
  bool need_quiet = false;
  params.send_buff = send_buff;
  params.recv_buff = recv_buff;
  int count = 0;
  for (int i = 1; i < send_counts.size(); ++i) {
    int src_rank, dst_rank;
    getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
    int dst_rank_global = getGlobalRank(handle, grid_desc, comm_axis, dst_rank);
    if (nvshmem_ptr(recv_buff, dst_rank_global)) { continue; }

    params.send_offsets[count] = send_offsets[dst_rank];
    params.recv_offsets[count] = recv_offsets[dst_rank];
    params.send_counts[count] = send_counts[dst_rank];
    params.peer_ranks[count] = dst_rank_global;
    count++;

    if (count == CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY) {
      params.ntransfers = count;
      cudecomp_nvshmem_alltoallv(params, stream);
      count = 0;
      need_quiet = true;
    }
  }
  if (count != 0) {
    params.ntransfers = count;
    cudecomp_nvshmem_alltoallv(params, stream);
    need_quiet = true;
  }

  // Intra-group transfers (blocking, scheduled after non-blocking inter-group transfers for concurrency)
  count = 0;
  for (int i = 1; i < send_counts.size(); ++i) {
    int src_rank, dst_rank;
    getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
    int dst_rank_global = getGlobalRank(handle, grid_desc, comm_axis, dst_rank);
    if (nvshmem_ptr(recv_buff, dst_rank_global)) {

      if (comm_info.ngroups == 1 && handle->device_p2p_ce_count == 1 &&
          count % CUDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ == 0) {
        // For single group, single P2P CE (e.g. NVSwitch), synchronize NVSHMEM team every
        // CUDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ transfers This helps reduce CE contention due to accumulation of
        // jitter.
        for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
          CHECK_CUDA(cudaEventRecord(grid_desc->events[0], handle->streams[i]));
          CHECK_CUDA(cudaStreamWaitEvent(handle->streams[handle->device_p2p_ce_count], grid_desc->events[0], 0));
        }

        nvshmemx_team_sync_on_stream(team, handle->streams[handle->device_p2p_ce_count]);

        CHECK_CUDA(cudaEventRecord(grid_desc->events[0], handle->streams[handle->device_p2p_ce_count]));
        for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
          CHECK_CUDA(cudaStreamWaitEvent(handle->streams[i], grid_desc->events[0], 0));
        }
      }

      nvshmemx_putmem_on_stream(recv_buff + recv_offsets[dst_rank], send_buff + send_offsets[dst_rank],
                                send_counts[dst_rank] * sizeof(T), dst_rank_global,
                                handle->streams[count % handle->device_p2p_ce_count]);
      count++;
    }
  }

  // Self-copy with cudaMemcpy
  CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                             send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));

  // Event dependency on internal streams for completion of intra-group transfers
  for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
    CHECK_CUDA(cudaEventRecord(grid_desc->events[0], handle->streams[i]));
    CHECK_CUDA(cudaStreamWaitEvent(stream, grid_desc->events[0], 0));
  }

  if (need_quiet) { nvshmemx_quiet_on_stream(stream); }
  nvshmemx_team_sync_on_stream(team, stream);
}
#endif

template <typename T>
static void cudecompAlltoall(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                             const std::vector<comm_count_t>& send_counts,
                             const std::vector<comm_count_t>& send_offsets, T* recv_buff,
                             const std::vector<comm_count_t>& recv_counts,
                             const std::vector<comm_count_t>& recv_offsets,
                             const std::vector<comm_count_t>& recv_offsets_nvshmem, cudecompCommAxis comm_axis,
                             cudaStream_t stream, cudecompTransposePerformanceSample* current_sample = nullptr) {
  nvtx::rangePush("cudecompAlltoall");

  if (handle->performance_report_enable) {
    CHECK_CUDA(cudaEventRecord(current_sample->alltoall_start_events[current_sample->alltoall_timing_count], stream));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      transposeBackendRequiresMpi(grid_desc->config.transpose_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("CUDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via cudecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  std::vector<MPI_Request> reqs;
  switch (grid_desc->config.transpose_comm_backend) {
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      nvshmemAlltoallV(handle, grid_desc, send_buff, send_counts, send_offsets, recv_buff, recv_counts,
                       recv_offsets_nvshmem, comm_axis, stream);
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via cudecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case CUDECOMP_TRANSPOSE_COMM_NCCL: {
    auto& comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
    // For fully intra-group alltoall, use distinct NCCL local comm instead of global comm as it is faster.
    auto comm = (comm_info.ngroups == 1) ? *grid_desc->nccl_local_comm : *grid_desc->nccl_comm;

    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < send_counts.size(); ++i) {
      int peer_rank_global = getGlobalRank(handle, grid_desc, comm_axis, i);
      if (comm_info.ngroups == 1) { peer_rank_global = handle->rank_to_clique_rank[peer_rank_global]; }
      if (send_counts[i] != 0) {
        CHECK_NCCL(ncclSend(send_buff + send_offsets[i], send_counts[i] * sizeof(T), ncclChar, peer_rank_global, comm,
                            stream));
      }
      if (recv_counts[i] != 0) {
        CHECK_NCCL(ncclRecv(recv_buff + recv_offsets[i], recv_counts[i] * sizeof(T), ncclChar, peer_rank_global, comm,
                            stream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());
    break;
  }
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P: {
    std::vector<MPI_Request> reqs(2 * send_counts.size(), MPI_REQUEST_NULL);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto comm =
        (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    // Self-copy with cudaMemcpy
    CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                               send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));

    for (int i = 1; i < recv_counts.size(); ++i) {
      int src_rank, dst_rank;
      getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
      if (recv_counts[src_rank] != 0) {
        checkMpiInt32Limit(recv_counts[src_rank], grid_desc->config.transpose_comm_backend);
        int32_t rc = static_cast<int32_t>(recv_counts[src_rank]);
        CHECK_MPI(
            MPI_Irecv(recv_buff + recv_offsets[src_rank], rc, getMpiDataType<T>(), src_rank, 0, comm, &reqs[src_rank]));
      }
    }

    for (int i = 1; i < send_counts.size(); ++i) {
      int src_rank, dst_rank;
      getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
      if (send_counts[dst_rank] != 0) {
        checkMpiInt32Limit(send_counts[dst_rank], grid_desc->config.transpose_comm_backend);
        int32_t sc = static_cast<int32_t>(send_counts[dst_rank]);
        CHECK_MPI(MPI_Isend(send_buff + send_offsets[dst_rank], sc, getMpiDataType<T>(), dst_rank, 0, comm,
                            &reqs[dst_rank + send_counts.size()]));
      }
    }

    CHECK_MPI(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));

    break;
  }
  case CUDECOMP_TRANSPOSE_COMM_MPI_A2A: {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto comm =
        (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    bool use_alltoall = canUseMpiAlltoall(send_counts, send_offsets, recv_counts, recv_offsets);

    if (use_alltoall) {
      checkMpiInt32Limit(send_counts[0], grid_desc->config.transpose_comm_backend);
      checkMpiInt32Limit(recv_counts[0], grid_desc->config.transpose_comm_backend);
      int32_t sc = static_cast<int32_t>(send_counts[0]);
      int32_t rc = static_cast<int32_t>(recv_counts[0]);
      CHECK_MPI(MPI_Alltoall(send_buff, sc, getMpiDataType<T>(), recv_buff, rc, getMpiDataType<T>(), comm));
    } else {
      // Self-copy with cudaMemcpy
      CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                                 send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));

      // Convert count/offset args to int32
      std::vector<int32_t> send_counts_i32(send_counts.size());
      std::vector<int32_t> send_offsets_i32(send_offsets.size());
      std::vector<int32_t> recv_counts_i32(recv_counts.size());
      std::vector<int32_t> recv_offsets_i32(recv_offsets.size());

      for (int i = 0; i < send_counts.size(); ++i) {
        int64_t sc = send_counts[i];
        int64_t so = send_offsets[i];
        int64_t rc = recv_counts[i];
        int64_t ro = recv_offsets[i];

        checkMpiInt32Limit(sc, grid_desc->config.transpose_comm_backend);
        checkMpiInt32Limit(so, grid_desc->config.transpose_comm_backend);
        checkMpiInt32Limit(rc, grid_desc->config.transpose_comm_backend);
        checkMpiInt32Limit(ro, grid_desc->config.transpose_comm_backend);

        send_counts_i32[i] = static_cast<int32_t>(sc);
        send_offsets_i32[i] = static_cast<int32_t>(so);
        recv_counts_i32[i] = static_cast<int32_t>(rc);
        recv_offsets_i32[i] = static_cast<int32_t>(ro);
      }

      // Exclude self copy portion
      send_counts_i32[self_rank] = 0;
      recv_counts_i32[self_rank] = 0;

      CHECK_MPI(MPI_Alltoallv(send_buff, send_counts_i32.data(), send_offsets_i32.data(), getMpiDataType<T>(),
                              recv_buff, recv_counts_i32.data(), recv_offsets_i32.data(), getMpiDataType<T>(), comm));
    }
    break;
  }
  default: {
    break;
  }
  }

  if (handle->performance_report_enable) {
    CHECK_CUDA(cudaEventRecord(current_sample->alltoall_end_events[current_sample->alltoall_timing_count], stream));
    current_sample->alltoall_timing_count++;
  }

  nvtx::rangePop();
}

template <typename T>
static void
cudecompAlltoallPipelined(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                          const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                          T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                          const std::vector<comm_count_t>& recv_offsets,
                          const std::vector<comm_count_t>& recv_offsets_nvshmem, cudecompCommAxis comm_axis,
                          const std::vector<int>& src_ranks, const std::vector<int>& dst_ranks, cudaStream_t stream,
                          bool& synced, cudecompTransposePerformanceSample* current_sample = nullptr) {

  // If there are no transfers to complete, quick return
  if (send_counts.size() == 0 && recv_counts.size() == 0) { return; }

  std::ostringstream os;
  os << "cudecompAlltoallPipelined_";
  for (int i = 0; i < src_ranks.size(); ++i) {
    os << src_ranks[i] << "," << dst_ranks[i];
    if (i != src_ranks.size() - 1) os << "_";
  }
  nvtx::rangePush(os.str());

  int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;
  if (handle->performance_report_enable && src_ranks[0] != self_rank) {
    // Note: skipping self-copy for timing as it should be overlapped
    CHECK_CUDA(cudaStreamWaitEvent(handle->streams[0], grid_desc->events[dst_ranks[0]], 0));
    CHECK_CUDA(cudaEventRecord(current_sample->alltoall_start_events[current_sample->alltoall_timing_count],
                               handle->streams[0]));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      transposeBackendRequiresMpi(grid_desc->config.transpose_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("CUDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via cudecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  switch (grid_desc->config.transpose_comm_backend) {
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      auto team = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.nvshmem_team
                                                   : grid_desc->col_comm_info.nvshmem_team;
      auto& comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info
                                                         : grid_desc->col_comm_info;
      auto pl_stream = handle->streams[0];
      int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

      bool barrier = false;
      for (int i = 0; i < src_ranks.size(); ++i) {
        int src_rank = src_ranks[i];
        int dst_rank = dst_ranks[i];

        if (src_rank == self_rank) {
          // Self-copy with cudaMemcpy
          CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets_nvshmem[self_rank], send_buff + send_offsets[self_rank],
                                     send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        } else {
          CHECK_CUDA(cudaStreamWaitEvent(pl_stream, grid_desc->events[dst_rank], 0));
          if (!synced) {
            nvshmemx_team_sync_on_stream(team, pl_stream);
            // Only need to sync on the first remote operation of an alltoall sequence to ensure reads on other ranks
            // from previous communication have completed.
            synced = true;
          }

          int dst_rank_global = getGlobalRank(handle, grid_desc, comm_axis, dst_rank);
          comm_info.nvshmem_signal_counts[src_rank]++;
          nvshmemx_putmem_signal_nbi_on_stream(
              recv_buff + recv_offsets_nvshmem[dst_rank], send_buff + send_offsets[dst_rank],
              send_counts[dst_rank] * sizeof(T),
              &comm_info.nvshmem_signals[comm_info.rank], comm_info.nvshmem_signal_counts[src_rank], NVSHMEM_SIGNAL_SET,
              dst_rank_global, pl_stream);

          barrier = true;
        }
      }

      if (barrier) {
        nvshmemx_quiet_on_stream(pl_stream);
        for (int i = 0; i < src_ranks.size(); ++i) {
          int src_rank = src_ranks[i];
          int dst_rank = dst_ranks[i];
          if (src_rank != self_rank) {
            nvshmemx_signal_wait_until_on_stream(&comm_info.nvshmem_signals[src_rank], NVSHMEM_CMP_EQ,
                                                 comm_info.nvshmem_signal_counts[src_rank], pl_stream);
            CHECK_CUDA(cudaEventRecord(grid_desc->events[dst_rank], pl_stream));
            CHECK_CUDA(cudaStreamWaitEvent(stream, grid_desc->events[dst_rank], 0));
          }
        }
      }
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via cudecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case CUDECOMP_TRANSPOSE_COMM_NCCL_PL: {
    auto comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
    // For fully intra-group alltoall, use distinct NCCL local comm instead of global comm as it is faster.
    auto comm = (comm_info.ngroups == 1) ? *grid_desc->nccl_local_comm : *grid_desc->nccl_comm;
    auto pl_stream = handle->streams[0];
    int self_rank = comm_info.rank;

    bool group_started = false;
    for (int i = 0; i < src_ranks.size(); ++i) {
      int src_rank = src_ranks[i];
      int dst_rank = dst_ranks[i];

      if (src_rank == self_rank) {
        // Self-copy with cudaMemcpy
        CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                                   send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        CHECK_CUDA(cudaStreamWaitEvent(pl_stream, grid_desc->events[dst_rank], 0));

        if (!group_started) {
          CHECK_NCCL(ncclGroupStart());
          group_started = true;
        }
        int src_rank_global = getGlobalRank(handle, grid_desc, comm_axis, src_rank);
        int dst_rank_global = getGlobalRank(handle, grid_desc, comm_axis, dst_rank);
        if (comm_info.ngroups == 1) {
          src_rank_global = handle->rank_to_clique_rank[src_rank_global];
          dst_rank_global = handle->rank_to_clique_rank[dst_rank_global];
        }
        if (send_counts[dst_rank] != 0) {
          CHECK_NCCL(ncclSend(send_buff + send_offsets[dst_rank], send_counts[dst_rank] * sizeof(T), ncclChar,
                              dst_rank_global, comm, pl_stream));
        }
        if (recv_counts[src_rank] != 0) {
          CHECK_NCCL(ncclRecv(recv_buff + recv_offsets[src_rank], recv_counts[src_rank] * sizeof(T), ncclChar,
                              src_rank_global, comm, pl_stream));
        }
      }
    }

    if (group_started) { CHECK_NCCL(ncclGroupEnd()); }

    for (int i = 0; i < src_ranks.size(); ++i) {
      int src_rank = src_ranks[i];
      int dst_rank = dst_ranks[i];
      if (src_rank != self_rank) {
        CHECK_CUDA(cudaEventRecord(grid_desc->events[dst_rank], pl_stream));
        CHECK_CUDA(cudaStreamWaitEvent(stream, grid_desc->events[dst_rank], 0));
      }
    }
    break;
  }
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL: {
    auto comm =
        (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    std::vector<MPI_Request> reqs(2 * src_ranks.size(), MPI_REQUEST_NULL);
    for (int i = 0; i < src_ranks.size(); ++i) {
      int src_rank = src_ranks[i];
      int dst_rank = dst_ranks[i];
      if (src_rank == self_rank) {
        // Self-copy with cudaMemcpy
        CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                                   send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        CHECK_CUDA(cudaEventSynchronize(grid_desc->events[dst_rank]));

        if (send_counts[dst_rank] != 0) {
          checkMpiInt32Limit(send_counts[dst_rank], grid_desc->config.transpose_comm_backend);
          int32_t sc = static_cast<int32_t>(send_counts[dst_rank]);
          CHECK_MPI(
              MPI_Isend(send_buff + send_offsets[dst_rank], sc, getMpiDataType<T>(), dst_rank, 0, comm, &reqs[i]));
        }
        if (recv_counts[src_rank] != 0) {
          checkMpiInt32Limit(recv_counts[src_rank], grid_desc->config.transpose_comm_backend);
          int32_t rc = static_cast<int32_t>(recv_counts[src_rank]);
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[src_rank], rc, getMpiDataType<T>(), src_rank, 0, comm,
                              &reqs[i + src_ranks.size()]));
        }
      }
    }

    CHECK_MPI(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));

    break;
  }
  default: {
    break;
  }
  }

  if (handle->performance_report_enable && src_ranks[0] != self_rank) {
    CHECK_CUDA(cudaEventRecord(current_sample->alltoall_end_events[current_sample->alltoall_timing_count],
                               handle->streams[0]));
    current_sample->alltoall_timing_count++;
  }
  nvtx::rangePop();
}

template <typename T>
static void cudecompSendRecvPair(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc,
                                 const std::array<int32_t, 2>& peer_ranks, T* send_buff,
                                 const std::array<comm_count_t, 2>& send_counts,
                                 const std::array<size_t, 2>& send_offsets, T* recv_buff,
                                 const std::array<comm_count_t, 2>& recv_counts,
                                 const std::array<size_t, 2>& recv_offsets, cudaStream_t stream = 0,
                                 cudecompHaloPerformanceSample* current_sample = nullptr) {
  nvtx::rangePush("cudecompSendRecvPair");

  if (handle->performance_report_enable && current_sample) {
    CHECK_CUDA(cudaEventRecord(current_sample->sendrecv_start_event, stream));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      haloBackendRequiresMpi(grid_desc->config.halo_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("CUDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via cudecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  switch (grid_desc->config.halo_comm_backend) {
  case CUDECOMP_HALO_COMM_NVSHMEM:
  case CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      nvshmemx_barrier_all_on_stream(stream);
      for (int i = 0; i < send_counts.size(); ++i) {
        if (peer_ranks[i] == handle->rank) {
          // Self-copy with cudaMemcpy
          CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i],
                                     send_counts[i] * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        } else {
          if (peer_ranks[(i + 1) % 2] != -1) {
            nvshmemx_putmem_nbi_on_stream(recv_buff + recv_offsets[i], send_buff + send_offsets[(i + 1) % 2],
                                          send_counts[(i + 1) % 2] * sizeof(T), peer_ranks[(i + 1) % 2], stream);
          }
        }
        if (grid_desc->config.halo_comm_backend == CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING) {
          nvshmemx_barrier_all_on_stream(stream);
        }
      }

      if (grid_desc->config.halo_comm_backend == CUDECOMP_HALO_COMM_NVSHMEM) {
        nvshmemx_barrier_all_on_stream(stream);
      };
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via cudecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case CUDECOMP_HALO_COMM_NCCL: {
    auto comm = *grid_desc->nccl_comm;
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with cudaMemcpy
        CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
      } else {
        if (send_counts[(i + 1) % 2] != 0 && peer_ranks[(i + 1) % 2] != -1) {
          CHECK_NCCL(ncclSend(send_buff + send_offsets[(i + 1) % 2], send_counts[(i + 1) % 2] * sizeof(T), ncclChar,
                              peer_ranks[(i + 1) % 2], comm, stream));
        }
        if (recv_counts[i] != 0 && peer_ranks[i] != -1) {
          CHECK_NCCL(
              ncclRecv(recv_buff + recv_offsets[i], recv_counts[i] * sizeof(T), ncclChar, peer_ranks[i], comm, stream));
        }
      }
    }
    CHECK_NCCL(ncclGroupEnd());
    break;
  }
  case CUDECOMP_HALO_COMM_MPI: {
    auto comm = handle->mpi_comm;
    std::vector<MPI_Request> reqs(2 * send_counts.size(), MPI_REQUEST_NULL);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with cudaMemcpy
        CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
      } else {
        if (recv_counts[(i + 1) % 2] != 0 && peer_ranks[(i + 1) % 2] != -1) {
          checkMpiInt32Limit(recv_counts[(i + 1) % 2], grid_desc->config.halo_comm_backend);
          int32_t rc = static_cast<int32_t>(recv_counts[(i + 1) % 2]);
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[(i + 1) % 2], rc, getMpiDataType<T>(), peer_ranks[(i + 1) % 2],
                              0, comm, &reqs[(i + 1) % 2]));
        }
        if (send_counts[i] != 0 && peer_ranks[i] != -1) {
          checkMpiInt32Limit(send_counts[i], grid_desc->config.halo_comm_backend);
          int32_t sc = static_cast<int32_t>(send_counts[i]);
          CHECK_MPI(MPI_Isend(send_buff + send_offsets[i], sc, getMpiDataType<T>(), peer_ranks[i], 0, comm,
                              &reqs[i + send_counts.size()]));
        }
      }
    }

    CHECK_MPI(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));
    break;
  }
  case CUDECOMP_HALO_COMM_MPI_BLOCKING: {
    auto comm = handle->mpi_comm;
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with cudaMemcpy
        CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
      } else {
        MPI_Request r = MPI_REQUEST_NULL;
        if (recv_counts[(i + 1) % 2] != 0 && peer_ranks[(i + 1) % 2] != -1) {
          checkMpiInt32Limit(recv_counts[(i + 1) % 2], grid_desc->config.halo_comm_backend);
          int32_t rc = static_cast<int32_t>(recv_counts[(i + 1) % 2]);
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[(i + 1) % 2], rc, getMpiDataType<T>(), peer_ranks[(i + 1) % 2],
                              0, comm, &r));
        }
        if (send_counts[i] != 0 && peer_ranks[i] != -1) {
          checkMpiInt32Limit(send_counts[i], grid_desc->config.halo_comm_backend);
          int32_t sc = static_cast<int32_t>(send_counts[i]);
          CHECK_MPI(MPI_Send(send_buff + send_offsets[i], sc, getMpiDataType<T>(), peer_ranks[i], 0, comm));
        }
        CHECK_MPI(MPI_Wait(&r, MPI_STATUS_IGNORE));
      }
    }

    break;
  }
  default: {
    break;
  }
  }

  if (handle->performance_report_enable && current_sample) {
    CHECK_CUDA(cudaEventRecord(current_sample->sendrecv_end_event, stream));
  }

  nvtx::rangePop();
}

} // namespace cudecomp

#endif // COMM_ROUTINES_H
