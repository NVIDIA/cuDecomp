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

#ifdef ENABLE_NVSHMEM
#define CUDECOMP_NVSHMEM_CHUNK_SZ (static_cast<size_t>(1024 * 1024 * 1024))
template <typename T>
static void
nvshmemAlltoallV(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                 const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                 T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                 const std::vector<comm_count_t>& recv_offsets, cudecompCommAxis comm_axis, cudaStream_t stream = 0) {
  auto comm = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
  // auto team =
  //    (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.nvshmem_team :
  //    grid_desc->col_comm_info.nvshmem_team;
  int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

  // Using cudaStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_MPI(MPI_Barrier(comm));
  // nvshmemx_team_sync_on_stream(team, stream);

  cudecompNvshmemA2AParams<T> params;
  params.send_buff = send_buff;
  params.recv_buff = recv_buff;
  int count = 0;
  for (int i = 1; i < send_counts.size(); ++i) {
    int src_rank, dst_rank;
    getAlltoallPeerRanks(grid_desc, comm_axis, i, src_rank, dst_rank);
    int dst_rank_global = getGlobalRank(grid_desc, comm_axis, dst_rank);
    if (nvshmem_ptr(recv_buff, dst_rank_global)) {
      // Use host call for direct P2P accessible entries
      // Need to chunk host API calls due to 2 GiB limitation in API
      size_t send_bytes = send_counts[dst_rank] * sizeof(T);
      size_t nchunks = (send_bytes + CUDECOMP_NVSHMEM_CHUNK_SZ - 1) / CUDECOMP_NVSHMEM_CHUNK_SZ;
      for (size_t j = 0; j < nchunks; ++j) {
        nvshmemx_putmem_nbi_on_stream(
            recv_buff + recv_offsets[dst_rank] + j * (CUDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
            send_buff + send_offsets[dst_rank] + j * (CUDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
            std::min(CUDECOMP_NVSHMEM_CHUNK_SZ, send_bytes - j * CUDECOMP_NVSHMEM_CHUNK_SZ),
            dst_rank_global, stream);
      }
      continue;
    }

    params.send_offsets[count] = send_offsets[dst_rank];
    params.recv_offsets[count] = recv_offsets[dst_rank];
    params.send_counts[count] = send_counts[dst_rank];
    params.peer_ranks[count] = dst_rank_global;
    count++;

    if (count == CUDECOMP_NVSHMEM_A2A_PARAM_CAPACITY) {
      params.ntransfers = count;
      cudecomp_nvshmem_alltoallv(params, stream);
      count = 0;
    }
  }
  if (count != 0) {
    params.ntransfers = count;
    cudecomp_nvshmem_alltoallv(params, stream);
  }

  // Self-copy with cudaMemcpy
  CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                             send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));

  nvshmemx_quiet_on_stream(stream);

  // Using cudaStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_MPI(MPI_Barrier(comm));
  // nvshmemx_team_sync_on_stream(team, stream);
}
#endif

template <typename T>
static void
cudecompAlltoall(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                 const std::vector<comm_count_t>& send_counts, const std::vector<comm_count_t>& send_offsets,
                 T* recv_buff, const std::vector<comm_count_t>& recv_counts,
                 const std::vector<comm_count_t>& recv_offsets, const std::vector<comm_count_t>& recv_offsets_nvshmem,
                 cudecompCommAxis comm_axis, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompAlltoall");

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
    auto comm_info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;
    // For fully intranode alltoall, use distinct NCCL local comm instead of global comm as it is faster.
    auto comm = (comm_info.nnodes == 1) ? handle->nccl_local_comm : handle->nccl_comm;

    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < send_counts.size(); ++i) {
      int peer_rank_global = getGlobalRank(grid_desc, comm_axis, i);
      if (comm_info.nnodes == 1) { peer_rank_global = handle->rank_to_local_rank[peer_rank_global]; }
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
  case CUDECOMP_TRANSPOSE_COMM_MPI_A2A: {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto send_counts_mod = send_counts;
    auto recv_counts_mod = recv_counts;
    auto comm =
        (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
    int self_rank = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.rank : grid_desc->col_comm_info.rank;

    // Self-copy with cudaMemcpy
    CHECK_CUDA(cudaMemcpyAsync(recv_buff + recv_offsets[self_rank], send_buff + send_offsets[self_rank],
                               send_counts[self_rank] * sizeof(T), cudaMemcpyDeviceToDevice, stream));

    send_counts_mod[self_rank] = 0;
    recv_counts_mod[self_rank] = 0;
    CHECK_MPI(MPI_Alltoallv(send_buff, send_counts_mod.data(), send_offsets.data(), getMpiDataType<T>(), recv_buff,
                            recv_counts_mod.data(), recv_offsets.data(), getMpiDataType<T>(), comm));
    break;
  }
  default: {
    break;
  }
  }
  nvtx::rangePop();
}

template <typename T>
static void cudecompAlltoallPipelined(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc, T* send_buff,
                                      const std::vector<comm_count_t>& send_counts,
                                      const std::vector<comm_count_t>& send_offsets, T* recv_buff,
                                      const std::vector<comm_count_t>& recv_counts,
                                      const std::vector<comm_count_t>& recv_offsets,
                                      const std::vector<comm_count_t>& recv_offsets_nvshmem, cudecompCommAxis comm_axis,
                                      const std::vector<int>& src_ranks, const std::vector<int>& dst_ranks,
                                      cudaStream_t stream, bool& synced) {
  std::ostringstream os;
  os << "cudecompAlltoallPipelined_";
  for (int i = 0; i < src_ranks.size(); ++i) {
    os << src_ranks[i] << "," << dst_ranks[i];
    if (i != src_ranks.size() - 1) os << "_";
  }
  nvtx::rangePush(os.str());

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
      auto comm =
          (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.mpi_comm : grid_desc->col_comm_info.mpi_comm;
      // auto team = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info.nvshmem_team
      //                                             : grid_desc->col_comm_info.nvshmem_team;
      auto pl_stream = handle->pl_stream;
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
            // Using cudaStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
            CHECK_CUDA(cudaStreamSynchronize(pl_stream));
            CHECK_MPI(MPI_Barrier(comm));
            // Only need to sync on the first remote operation of an alltoall sequence to ensure reads on other ranks
            // from previous communication have completed.
            synced = true;
          }

          int dst_rank_global = getGlobalRank(grid_desc, comm_axis, dst_rank);
          // Need to chunk host API calls due to 2 GiB limitation in API
          size_t send_bytes = send_counts[dst_rank] * sizeof(T);
          int nchunks = (send_bytes + CUDECOMP_NVSHMEM_CHUNK_SZ - 1) / CUDECOMP_NVSHMEM_CHUNK_SZ;
          for (int j = 0; j < nchunks; ++j) {
            nvshmemx_putmem_nbi_on_stream(
                recv_buff + recv_offsets_nvshmem[dst_rank] + j * (CUDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                send_buff + send_offsets[dst_rank] + j * (CUDECOMP_NVSHMEM_CHUNK_SZ / sizeof(T)),
                std::min(static_cast<size_t>(CUDECOMP_NVSHMEM_CHUNK_SZ), send_bytes - j * CUDECOMP_NVSHMEM_CHUNK_SZ),
                dst_rank_global, pl_stream);
          }

          barrier = true;
        }
      }

      if (barrier) {
        nvshmemx_quiet_on_stream(pl_stream);
        // Using cudaStreamSynchronize + barrier instead of nvshmemx_team_sync_on_stream for lower latency
        CHECK_CUDA(cudaStreamSynchronize(pl_stream));
        CHECK_MPI(MPI_Barrier(comm));

        // nvshmemx_team_sync_on_stream(team, pl_stream);
        // for (int i = 0; i < src_ranks.size(); ++i) {
        //  int src_rank = src_ranks[i];
        //  int dst_rank = dst_ranks[i];
        //  if (src_rank != self_rank) {
        //    CHECK_CUDA(cudaEventRecord(grid_desc->events[dst_rank], pl_stream));
        //    CHECK_CUDA(cudaStreamWaitEvent(stream, grid_desc->events[dst_rank], 0));
        //  }
        //}
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
    // For fully intranode alltoall, use distinct NCCL local comm instead of global comm as it is faster.
    auto comm = (comm_info.nnodes == 1) ? handle->nccl_local_comm : handle->nccl_comm;
    auto pl_stream = handle->pl_stream;
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
        int src_rank_global = getGlobalRank(grid_desc, comm_axis, src_rank);
        int dst_rank_global = getGlobalRank(grid_desc, comm_axis, dst_rank);
        if (comm_info.nnodes == 1) {
          src_rank_global = handle->rank_to_local_rank[src_rank_global];
          dst_rank_global = handle->rank_to_local_rank[dst_rank_global];
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
  nvtx::rangePop();
}

template <typename T>
static void cudecompSendRecvPair(const cudecompHandle_t& handle, const cudecompGridDesc_t& grid_desc,
                                 const std::array<int32_t, 2>& peer_ranks, T* send_buff,
                                 const std::array<comm_count_t, 2>& send_counts,
                                 const std::array<size_t, 2>& send_offsets, T* recv_buff,
                                 const std::array<comm_count_t, 2>& recv_counts,
                                 const std::array<size_t, 2>& recv_offsets, cudaStream_t stream = 0) {
  nvtx::rangePush("cudecompSendRecvPair");

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
      nvshmemx_quiet_on_stream(stream);
      nvshmemx_sync_all_on_stream(stream);
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
          nvshmemx_quiet_on_stream(stream);
          nvshmemx_sync_all_on_stream(stream);
        }
      }

      if (grid_desc->config.halo_comm_backend == CUDECOMP_HALO_COMM_NVSHMEM) {
        nvshmemx_quiet_on_stream(stream);
        nvshmemx_sync_all_on_stream(stream);
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
    auto comm = handle->nccl_comm;
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
  nvtx::rangePop();
}

} // namespace cudecomp

#endif // COMM_ROUTINES_H
