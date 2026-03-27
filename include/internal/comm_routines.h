/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 The Authors.
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

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "internal/checks.h"
#include "internal/hipdecomp_kernels.h"
#include "internal/nvtx.h"

namespace hipdecomp {

static inline MPI_Datatype getMpiDataType(float) { return MPI_FLOAT; }
static inline MPI_Datatype getMpiDataType(double) { return MPI_DOUBLE; }
static inline MPI_Datatype getMpiDataType(std::complex<float>) { return MPI_C_FLOAT_COMPLEX; }
static inline MPI_Datatype getMpiDataType(std::complex<double>) { return MPI_C_DOUBLE_COMPLEX; }
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

#ifdef ENABLE_NVSHMEM
#define HIPDECOMP_NVSHMEM_CHUNK_SZ (static_cast<size_t>(1024 * 1024 * 1024))
#define HIPDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ 8 // max number of intra-group transfers to schedule between team syncs
template <typename T>
static void
nvshmemAlltoallV(const hipdecompHandle_t& handle, const hipdecompGridDesc_t& grid_desc, T* send_buff,
                 const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                 T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                 const std::vector<comm_count_t>& recv_offsets, hipdecompCommAxis comm_axis, hipStream_t stream) {
  auto comm_info = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
  auto comm = comm_info.mpi_comm;
  auto team = comm_info.nvshmem_team;
  int self_rank = comm_info.rank;

  // Event dependency on external stream for intra-group transfers
  CHECK_HIP(hipEventRecord(grid_desc->events[0], stream));
  for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
    CHECK_HIP(hipStreamWaitEvent(handle->streams[i], grid_desc->events[0], 0));
  }

  // Using hipEventSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
  CHECK_HIP(hipEventSynchronize(grid_desc->nvshmem_sync_event));
  CHECK_MPI(MPI_Barrier(comm));
  // nvshmemx_team_sync_on_stream(team, stream);

  hipdecompNvshmemA2AParams<T> params;

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

    if (count == HIPDECOMP_NVSHMEM_A2A_PARAM_CAPACITY) {
      params.ntransfers = count;
      hipdecomp_nvshmem_alltoallv(params, stream);
      count = 0;
      need_quiet = true;
    }
  }
  if (count != 0) {
    params.ntransfers = count;
    hipdecomp_nvshmem_alltoallv(params, stream);
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
          count % HIPDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ == 0) {
        // For single group, single P2P CE (e.g. NVSwitch), synchronize NVSHMEM team every
        // HIPDECOMP_NVSHMEM_INTRAGROUP_SYNC_FREQ transfers This helps reduce CE contention due to accumulation of
        // jitter.
        for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
          CHECK_HIP(hipEventRecord(grid_desc->events[0], handle->streams[i]));
          CHECK_HIP(hipStreamWaitEvent(handle->streams[handle->device_p2p_ce_count], grid_desc->events[0], 0));
        }

        nvshmemx_team_sync_on_stream(team, handle->streams[handle->device_p2p_ce_count]);

        CHECK_HIP(hipEventRecord(grid_desc->events[0], handle->streams[handle->device_p2p_ce_count]));
        for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
          CHECK_HIP(hipStreamWaitEvent(handle->streams[i], grid_desc->events[0], 0));
        }
      }

      // Use host call for direct P2P accessible entries
      // Need to chunk host API calls due to 2 GiB limitation in API
      size_t send_bytes = send_counts[dst_rank] * sizeof(T);
      size_t nchunks = (send_bytes + HIPDECOMP_NVSHMEM_CHUNK_SZ - 1) / HIPDECOMP_NVSHMEM_CHUNK_SZ;
      for (size_t j = 0; j < nchunks; ++j) {
        nvshmemx_putmem_on_stream(recv_buff + recv_offsets[dst_rank] + j * (HIPDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                                  send_buff + send_offsets[dst_rank] + j * (HIPDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                                  std::min(HIPDECOMP_NVSHMEM_CHUNK_SZ, send_bytes - j * HIPDECOMP_NVSHMEM_CHUNK_SZ),
                                  dst_rank_global, handle->streams[count % handle->device_p2p_ce_count]);
      }
      count++;
    }
  }

  // Self-copy with hipMemcpy
  CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                           send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));

  // Event dependency on internal streams for completion of intra-group transfers
  for (int i = 0; i < handle->device_p2p_ce_count; ++i) {
    CHECK_HIP(hipEventRecord(grid_desc->events[0], handle->streams[i]));
    CHECK_HIP(hipStreamWaitEvent(stream, grid_desc->events[0], 0));
  }

  if (need_quiet) { nvshmemx_quiet_on_stream(stream); }

  // Using hipStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_MPI(MPI_Barrier(comm));
  // nvshmemx_team_sync_on_stream(team, stream);
}
#endif

template <typename T>
static void hipdecompAlltoall(const hipdecompHandle_t& handle, const hipdecompGridDesc_t& grid_desc, T* send_buff,
                              const std::vector<comm_count_t>& send_counts,
                              const std::vector<comm_count_t>& send_offsets, T* recv_buff,
                              const std::vector<comm_count_t>& recv_counts,
                              const std::vector<comm_count_t>& recv_offsets,
                              const std::vector<comm_count_t>& recv_offsets_nvshmem, hipdecompCommAxis comm_axis,
                              hipStream_t stream, hipdecompTransposePerformanceSample* current_sample = nullptr) {
  nvtx::rangePush("hipdecompAlltoall");

  if (handle->performance_report_enable) {
    CHECK_HIP(hipEventRecord(current_sample->alltoall_start_events[current_sample->alltoall_timing_count], stream));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      transposeBackendRequiresMpi(grid_desc->config.transpose_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("HIPDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via hipdecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  std::vector<MPI_Request> reqs;
  switch (grid_desc->config.transpose_comm_backend) {
  case HIPDECOMP_TRANSPOSE_COMM_NVSHMEM: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      nvshmemAlltoallV(handle, grid_desc, send_buff, send_counts, send_offsets, recv_buff, recv_counts,
                       recv_offsets_nvshmem, comm_axis, stream);
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via hipdecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case HIPDECOMP_TRANSPOSE_COMM_NCCL: {
    auto comm_info = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
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
  case HIPDECOMP_TRANSPOSE_COMM_MPI_P2P: {
    std::vector<MPI_Request> reqs(2 * send_counts.size(), MPI_REQUEST_NULL);
    CHECK_HIP(hipStreamSynchronize(stream));

    auto comm =
        (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    // Self-copy with hipMemcpy
    CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                             send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));

    for (int i = 1; i < recv_counts.size(); ++i) {
      int src_rank, dst_rank;
      getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
      if (recv_counts[src_rank] != 0) {
        CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[src_rank], recv_counts[src_rank], getMpiDataType<T>(), src_rank, 0,
                            comm, &reqs[src_rank]));
      }
    }

    for (int i = 1; i < send_counts.size(); ++i) {
      int src_rank, dst_rank;
      getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
      if (send_counts[dst_rank] != 0) {
        CHECK_MPI(MPI_Isend(send_buff + send_offsets[dst_rank], send_counts[dst_rank], getMpiDataType<T>(), dst_rank, 0,
                            comm, &reqs[dst_rank + send_counts.size()]));
      }
    }

    CHECK_MPI(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));

    break;
  }
  case HIPDECOMP_TRANSPOSE_COMM_MPI_A2A: {
    CHECK_HIP(hipStreamSynchronize(stream));
    auto comm =
        (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    bool use_alltoall = canUseMpiAlltoall(send_counts, send_offsets, recv_counts, recv_offsets);

    if (use_alltoall) {
      CHECK_MPI(MPI_Alltoall(send_buff, send_counts[0], getMpiDataType<T>(), recv_buff, recv_counts[0],
                             getMpiDataType<T>(), comm));
    } else {
      // Self-copy with hipMemcpy
      CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                               send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));

      auto send_counts_mod = send_counts;
      auto recv_counts_mod = recv_counts;
      send_counts_mod[self_rank] = 0;
      recv_counts_mod[self_rank] = 0;
      CHECK_MPI(MPI_Alltoallv(send_buff, send_counts_mod.data(), send_offsets.data(), getMpiDataType<T>(), recv_buff,
                              recv_counts_mod.data(), recv_offsets.data(), getMpiDataType<T>(), comm));
    }
    break;
  }
  default: {
    break;
  }
  }

  if (handle->performance_report_enable) {
    CHECK_HIP(hipEventRecord(current_sample->alltoall_end_events[current_sample->alltoall_timing_count], stream));
    current_sample->alltoall_timing_count++;
  }

  nvtx::rangePop();
}

template <typename T>
static void
hipdecompAlltoallPipelined(const hipdecompHandle_t& handle, const hipdecompGridDesc_t& grid_desc, T* send_buff,
                           const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                           T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                           const std::vector<comm_count_t>& recv_offsets,
                           const std::vector<comm_count_t>& recv_offsets_nvshmem, hipdecompCommAxis comm_axis,
                           const std::vector<int>& src_ranks, const std::vector<int>& dst_ranks, hipStream_t stream,
                           bool& synced, hipdecompTransposePerformanceSample* current_sample = nullptr) {

  // If there are no transfers to complete, quick return
  if (send_counts.size() == 0 && recv_counts.size() == 0) { return; }

  std::ostringstream os;
  os << "hipdecompAlltoallPipelined_";
  for (int i = 0; i < src_ranks.size(); ++i) {
    os << src_ranks[i] << "," << dst_ranks[i];
    if (i != src_ranks.size() - 1) os << "_";
  }
  nvtx::rangePush(os.str());

  int self_rank = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;
  if (handle->performance_report_enable && src_ranks[0] != self_rank) {
    // Note: skipping self-copy for timing as it should be overlapped
    CHECK_HIP(hipStreamWaitEvent(handle->streams[0], grid_desc->events[dst_ranks[0]], 0));
    CHECK_HIP(hipEventRecord(current_sample->alltoall_start_events[current_sample->alltoall_timing_count],
                             handle->streams[0]));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      transposeBackendRequiresMpi(grid_desc->config.transpose_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("HIPDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via hipdecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  switch (grid_desc->config.transpose_comm_backend) {
  case HIPDECOMP_TRANSPOSE_COMM_NVSHMEM_PL: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      auto comm =
          (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
      // auto team = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.nvshmem_team
      //                                             : grid_desc->col_comm_info.nvshmem_team;
      auto pl_stream = handle->streams[0];
      int self_rank = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

      bool barrier = false;
      for (int i = 0; i < src_ranks.size(); ++i) {
        int src_rank = src_ranks[i];
        int dst_rank = dst_ranks[i];

        if (src_rank == self_rank) {
          // Self-copy with hipMemcpy
          CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets_nvshmem[self_rank], send_buff + send_offsets[self_rank],
                                   send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));
        } else {
          CHECK_HIP(hipStreamWaitEvent(pl_stream, grid_desc->events[dst_rank], 0));
          if (!synced) {
            // Using hipEventSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
            CHECK_HIP(hipEventSynchronize(grid_desc->nvshmem_sync_event));
            CHECK_MPI(MPI_Barrier(comm));
            // Only need to sync on the first remote operation of an alltoall sequence to ensure reads on other ranks
            // from previous communication have completed.
            synced = true;
          }

          int dst_rank_global = getGlobalRank(handle, grid_desc, comm_axis, dst_rank);
          // Need to chunk host API calls due to 2 GiB limitation in API
          size_t send_bytes = send_counts[dst_rank] * sizeof(T);
          int nchunks = (send_bytes + HIPDECOMP_NVSHMEM_CHUNK_SZ - 1) / HIPDECOMP_NVSHMEM_CHUNK_SZ;
          for (int j = 0; j < nchunks; ++j) {
            nvshmemx_putmem_nbi_on_stream(
                recv_buff + recv_offsets_nvshmem[dst_rank] + j * (HIPDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                send_buff + send_offsets[dst_rank] + j * (HIPDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                std::min(static_cast<size_t>(HIPDECOMP_NVSHMEM_CHUNK_SZ), send_bytes - j * HIPDECOMP_NVSHMEM_CHUNK_SZ),
                dst_rank_global, pl_stream);
          }

          barrier = true;
        }
      }

      if (barrier) {
        nvshmemx_quiet_on_stream(pl_stream);
        // Using hipStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
        CHECK_HIP(hipStreamSynchronize(pl_stream));
        CHECK_MPI(MPI_Barrier(comm));

        // nvshmemx_team_sync_on_stream(team, pl_stream);
        // for (int i = 0; i < src_ranks.size(); ++i) {
        //  int src_rank = src_ranks[i];
        //  int dst_rank = dst_ranks[i];
        //  if (src_rank != self_rank) {
        //    CHECK_HIP(hipEventRecord(grid_desc->events[dst_rank], pl_stream));
        //    CHECK_HIP(hipStreamWaitEvent(stream, grid_desc->events[dst_rank], 0));
        //  }
        //}
      }
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via hipdecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case HIPDECOMP_TRANSPOSE_COMM_NCCL_PL: {
    auto comm_info = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
    // For fully intra-group alltoall, use distinct NCCL local comm instead of global comm as it is faster.
    auto comm = (comm_info.ngroups == 1) ? *grid_desc->nccl_local_comm : *grid_desc->nccl_comm;
    auto pl_stream = handle->streams[0];
    int self_rank = comm_info.rank;

    bool group_started = false;
    for (int i = 0; i < src_ranks.size(); ++i) {
      int src_rank = src_ranks[i];
      int dst_rank = dst_ranks[i];

      if (src_rank == self_rank) {
        // Self-copy with hipMemcpy
        CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                                 send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));
      } else {
        CHECK_HIP(hipStreamWaitEvent(pl_stream, grid_desc->events[dst_rank], 0));

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
        CHECK_HIP(hipEventRecord(grid_desc->events[dst_rank], pl_stream));
        CHECK_HIP(hipStreamWaitEvent(stream, grid_desc->events[dst_rank], 0));
      }
    }
    break;
  }
  case HIPDECOMP_TRANSPOSE_COMM_MPI_P2P_PL: {
    auto comm =
        (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == HIPDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    std::vector<MPI_Request> reqs(2 * src_ranks.size(), MPI_REQUEST_NULL);
    for (int i = 0; i < src_ranks.size(); ++i) {
      int src_rank = src_ranks[i];
      int dst_rank = dst_ranks[i];
      if (src_rank == self_rank) {
        // Self-copy with hipMemcpy
        CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                                 send_counts[self_rank] * sizeof(T), hipMemcpyDeviceToDevice, stream));
      } else {
        CHECK_HIP(hipEventSynchronize(grid_desc->events[dst_rank]));

        if (send_counts[dst_rank] != 0) {
          CHECK_MPI(MPI_Isend(send_buff + send_offsets[dst_rank], send_counts[dst_rank], getMpiDataType<T>(), dst_rank,
                              0, comm, &reqs[i]));
        }
        if (recv_counts[src_rank] != 0) {
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[src_rank], recv_counts[src_rank], getMpiDataType<T>(), src_rank,
                              0, comm, &reqs[i + src_ranks.size()]));
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
    CHECK_HIP(
        hipEventRecord(current_sample->alltoall_end_events[current_sample->alltoall_timing_count], handle->streams[0]));
    current_sample->alltoall_timing_count++;
  }
  nvtx::rangePop();
}

template <typename T>
static void hipdecompSendRecvPair(const hipdecompHandle_t& handle, const hipdecompGridDesc_t& grid_desc,
                                  const std::array<int32_t, 2>& peer_ranks, T* send_buff,
                                  const std::array<comm_count_t, 2>& send_counts,
                                  const std::array<size_t, 2>& send_offsets, T* recv_buff,
                                  const std::array<comm_count_t, 2>& recv_counts,
                                  const std::array<size_t, 2>& recv_offsets, hipStream_t stream = 0,
                                  hipdecompHaloPerformanceSample* current_sample = nullptr) {
  nvtx::rangePush("hipdecompSendRecvPair");

  if (handle->performance_report_enable && current_sample) {
    CHECK_HIP(hipEventRecord(current_sample->sendrecv_start_event, stream));
  }

#ifdef ENABLE_NVSHMEM
  if (handle->rank == 0 && handle->nvshmem_initialized && !handle->nvshmem_mixed_buffer_warning_issued &&
      haloBackendRequiresMpi(grid_desc->config.halo_comm_backend) &&
      (nvshmem_ptr(send_buff, handle->rank) || nvshmem_ptr(recv_buff, handle->rank))) {
    printf("HIPDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via hipdecompMalloc) is "
           "being used with an MPI communication backend. This may cause issues with some MPI "
           "implementations. See the documentation for additional details and possible workarounds "
           "if you encounter issues.\n");
    handle->nvshmem_mixed_buffer_warning_issued = true;
  }
#endif

  switch (grid_desc->config.halo_comm_backend) {
  case HIPDECOMP_HALO_COMM_NVSHMEM:
  case HIPDECOMP_HALO_COMM_NVSHMEM_BLOCKING: {
#ifdef ENABLE_NVSHMEM
    if (nvshmem_ptr(send_buff, handle->rank) && nvshmem_ptr(recv_buff, handle->rank)) {
      nvshmemx_quiet_on_stream(stream);
      nvshmemx_sync_all_on_stream(stream);
      for (int i = 0; i < send_counts.size(); ++i) {
        if (peer_ranks[i] == handle->rank) {
          // Self-copy with hipMemcpy
          CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                   hipMemcpyDeviceToDevice, stream));
        } else {
          if (peer_ranks[(i + 1) % 2] != -1) {
            nvshmemx_putmem_nbi_on_stream(recv_buff + recv_offsets[i], send_buff + send_offsets[(i + 1) % 2],
                                          send_counts[(i + 1) % 2] * sizeof(T), peer_ranks[(i + 1) % 2], stream);
          }
        }
        if (grid_desc->config.halo_comm_backend == HIPDECOMP_HALO_COMM_NVSHMEM_BLOCKING) {
          nvshmemx_quiet_on_stream(stream);
          nvshmemx_sync_all_on_stream(stream);
        }
      }

      if (grid_desc->config.halo_comm_backend == HIPDECOMP_HALO_COMM_NVSHMEM) {
        nvshmemx_quiet_on_stream(stream);
        nvshmemx_sync_all_on_stream(stream);
      };
      break;
    } else {
      THROW_INVALID_USAGE("NVSHMEM communication backends require workspace allocated via hipdecompMalloc.");
    }
#else
    THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
  }
  case HIPDECOMP_HALO_COMM_NCCL: {
    auto comm = *grid_desc->nccl_comm;
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with hipMemcpy
        CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                 hipMemcpyDeviceToDevice, stream));
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
  case HIPDECOMP_HALO_COMM_MPI: {
    auto comm = handle->mpi_comm;
    std::vector<MPI_Request> reqs(2 * send_counts.size(), MPI_REQUEST_NULL);
    CHECK_HIP(hipStreamSynchronize(stream));
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with hipMemcpy
        CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                 hipMemcpyDeviceToDevice, stream));
      } else {
        if (recv_counts[(i + 1) % 2] != 0 && peer_ranks[(i + 1) % 2] != -1) {
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[(i + 1) % 2], recv_counts[(i + 1) % 2], getMpiDataType<T>(),
                              peer_ranks[(i + 1) % 2], 0, comm, &reqs[(i + 1) % 2]));
        }
        if (send_counts[i] != 0 && peer_ranks[i] != -1) {
          CHECK_MPI(MPI_Isend(send_buff + send_offsets[i], send_counts[i], getMpiDataType<T>(), peer_ranks[i], 0, comm,
                              &reqs[i + send_counts.size()]));
        }
      }
    }

    CHECK_MPI(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));
    break;
  }
  case HIPDECOMP_HALO_COMM_MPI_BLOCKING: {
    auto comm = handle->mpi_comm;
    CHECK_HIP(hipStreamSynchronize(stream));
    for (int i = 0; i < send_counts.size(); ++i) {
      if (peer_ranks[i] == handle->rank) {
        // Self-copy with hipMemcpy
        CHECK_HIP(hipMemcpyAsync(recv_buff + recv_offsets[i], send_buff + send_offsets[i], send_counts[i] * sizeof(T),
                                 hipMemcpyDeviceToDevice, stream));
      } else {
        MPI_Request r = MPI_REQUEST_NULL;
        if (recv_counts[(i + 1) % 2] != 0 && peer_ranks[(i + 1) % 2] != -1) {
          CHECK_MPI(MPI_Irecv(recv_buff + recv_offsets[(i + 1) % 2], recv_counts[(i + 1) % 2], getMpiDataType<T>(),
                              peer_ranks[(i + 1) % 2], 0, comm, &r));
        }
        if (send_counts[i] != 0 && peer_ranks[i] != -1) {
          CHECK_MPI(MPI_Send(send_buff + send_offsets[i], send_counts[i], getMpiDataType<T>(), peer_ranks[i], 0, comm));
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
    CHECK_HIP(hipEventRecord(current_sample->sendrecv_end_event, stream));
  }

  nvtx::rangePop();
}

} // namespace hipdecomp

#endif // COMM_ROUTINES_H
