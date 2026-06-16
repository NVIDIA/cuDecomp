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

#ifndef CUDECOMP_COMMON_H
#define CUDECOMP_COMMON_H

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <mpi.h>
#include <nccl.h>
#include <nvml.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/graph.h"
#include "internal/raii_wrappers.h"

namespace cudecomp {
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
typedef std::pair<std::array<unsigned char, NVML_GPU_FABRIC_UUID_LEN>, unsigned int> mnnvl_info;
#else
typedef std::pair<std::array<unsigned char, 1>, unsigned int> mnnvl_info;
#endif
typedef std::shared_ptr<ncclComm_t> ncclComm;
struct nvshmemRuntimeState {
#ifdef ENABLE_NVSHMEM
  ~nvshmemRuntimeState() noexcept { finalize(); }

  void finalize() noexcept {
    if (initialized) {
      nvshmem_finalize();
      initialized = false;
    }
    nvshmem_allocations.clear();
    nvshmem_allocation_size = 0;
  }

  bool initialized = false;                              // Flag to track NVSHMEM initialization
  size_t nvshmem_symmetric_size = 0;                     // NVSHMEM symmetric size
  bool nvshmem_vmm = true;                               // Flag to track if NVSHMEM is using VMM allocations
  std::unordered_map<void*, size_t> nvshmem_allocations; // Table to record NVSHMEM allocations
  size_t nvshmem_allocation_size = 0;                    // Total of NVSHMEM allocations
#endif
};
typedef std::shared_ptr<nvshmemRuntimeState> nvshmemRuntime;
#ifdef ENABLE_NVSHMEM
struct nvshmemProcessState {
  // NVSHMEM fixes the PE mapping at initialization and does not support reinitializing
  // on a different set or order of ranks, even after nvshmem_finalize().
  std::vector<int> init_world_ranks;
  std::weak_ptr<nvshmemRuntimeState> active_runtime;
  bool mixed_buffer_warning_issued = false;
};

void warnIfNvshmemBufferUsedWithMpi(cudecompHandle_t handle, const void* send_buff, const void* recv_buff);
#endif
} // namespace cudecomp

// cuDecomp handle containing general information
struct cudecompHandle {
  cudecompHandle() = default;
  ~cudecompHandle() noexcept;

  cudecompHandle(const cudecompHandle&) = delete;
  cudecompHandle& operator=(const cudecompHandle&) = delete;
  cudecompHandle(cudecompHandle&&) = delete;
  cudecompHandle& operator=(cudecompHandle&&) = delete;

  MPI_Comm mpi_comm = MPI_COMM_NULL; // MPI communicator
  int32_t rank;                      // MPI rank
  int32_t nranks;                    // MPI size

  MPI_Comm mpi_local_comm = MPI_COMM_NULL; // MPI local communicator
  int32_t local_rank;                      // MPI rank
  int32_t local_nranks;                    // MPI size

  MPI_Comm mpi_clique_comm = MPI_COMM_NULL; // MPI MNNVL clique local communicator
  int32_t clique_rank;                      // MPI rank
  int32_t clique_nranks;                    // MPI size

  // Entries for NCCL management
  cudecomp::ncclComm nccl_comm;       // NCCL communicator (global)
  cudecomp::ncclComm nccl_local_comm; // NCCL communicator (intra-node, or intra-clique on MNNVL systems)
  bool nccl_enable_ubr = false;       // Flag to control NCCL user buffer registration usage
  std::unordered_map<void*, std::vector<std::pair<cudecomp::ncclComm, void*>>>
      nccl_ubr_handles; // map of allocated buffer address to NCCL registration handle(s)

  std::vector<cudecomp::cudaStream> streams; // internal streams for concurrent scheduling

#if CUTENSOR_MAJOR >= 2
  cutensorHandle_t cutensor_handle = nullptr;            // cuTENSOR handle;
  cutensorPlanPreference_t cutensor_plan_pref = nullptr; // cuTENSOR plan preference;
  bool cutensor_needs_permute_chunking = false;          // Flag to enable large tensor workaround
#else
  cutensorHandle_t cutensor_handle; // cuTENSOR handle;
#endif

  std::vector<std::array<char, MPI_MAX_PROCESSOR_NAME>> hostnames; // list of hostnames by rank
  std::vector<int32_t> rank_to_local_rank;                         // list of local rank mappings

  bool initialized = false;

  // Multi-node NVLINK (MNNVL)
  bool cuda_cumem_enable = false; // Flag to control whether cuMem* APIs are used for cudecompMalloc/Free
  std::vector<cudecomp::mnnvl_info> rank_to_mnnvl_info; // list of mnnvl information (clusterUuid, cliqueId) by rank
  std::vector<unsigned int> rank_to_clique;             // list of rank to MNNVL clique mappings
  std::vector<int> rank_to_clique_rank;                 // list of rank to MNNVL clique rank mappings

  // CUDA graphs
  bool cuda_graphs_enable = false; // Flag to control whether CUDA graphs are used

  // Performance reporting related entries
  bool performance_report_enable = false; // flag to track if performance reporting is enabled
  int32_t performance_report_detail =
      0; // performance report detail level: 0=aggregated, 1=per-sample rank 0, 2=per-sample all ranks
  int32_t performance_report_samples = 20;       // number of performance samples to keep for final report
  int32_t performance_report_warmup_samples = 3; // number of initial warmup samples to ignore for each configuration
  std::string performance_report_write_dir =
      ""; // directory to write CSV performance reports, empty means no file writing

  // Miscellaneous
  bool nvml_initialized = false;                        // Flag to track NVML initialization
  int32_t device_p2p_ce_count = 0;                      // number of P2P CEs available
  int32_t device_num_sms = 0;                           // number of SMs on the device
  int32_t device_max_threads_per_sm = 0;                // maximum threads per SM
  bool col_major_rank_order_env_warning_issued = false; // Warn once for deprecated rank order env var
};

// Structure with information about row/column communicator
struct cudecompCommInfo {
  cudecompCommInfo() = default;
  ~cudecompCommInfo() noexcept { reset(); }

  cudecompCommInfo(const cudecompCommInfo&) = delete;
  cudecompCommInfo& operator=(const cudecompCommInfo&) = delete;
  cudecompCommInfo(cudecompCommInfo&&) = delete;
  cudecompCommInfo& operator=(cudecompCommInfo&&) = delete;

  void reset() noexcept {
    if (mpi_comm != MPI_COMM_NULL) {
      MPI_Comm comm = mpi_comm;
      mpi_comm = MPI_COMM_NULL;
      MPI_Comm_free(&comm);
    }
#ifdef ENABLE_NVSHMEM
    if (nvshmem_team != NVSHMEM_TEAM_INVALID) {
      nvshmem_team_destroy(nvshmem_team);
      nvshmem_team = NVSHMEM_TEAM_INVALID;
    }
    if (nvshmem_signals) {
      nvshmem_free(nvshmem_signals);
      nvshmem_signals = nullptr;
    }
#endif
    rank = 0;
    nranks = 0;
    ngroups = 0;
    npergroup = 0;
    mnnvl_active = false;
  }

  MPI_Comm mpi_comm = MPI_COMM_NULL;
  int32_t rank = 0;
  int32_t nranks = 0;

  int32_t ngroups = 0;   // number of p2p groups (i.e. grouping of ranks with fast interconnect) in communicator
  int32_t npergroup = 0; // number of ranks per p2p group

#ifdef ENABLE_NVSHMEM
  nvshmem_team_t nvshmem_team = NVSHMEM_TEAM_INVALID;
  uint64_t* nvshmem_signals = nullptr;
#endif

  bool mnnvl_active = false; // flag to indicate whether communicator has MNNVL connections
};

// Structure to contain data for transpose performance sample
struct cudecompTransposePerformanceSample {
  cudecomp::cudaEventTimed transpose_start_event;
  cudecomp::cudaEventTimed transpose_end_event;
  std::vector<cudecomp::cudaEventTimed> alltoall_start_events;
  std::vector<cudecomp::cudaEventTimed> alltoall_end_events;
  int32_t alltoall_timing_count = 0;
  size_t alltoall_bytes = 0;
  bool valid = false;
};

// Collection of transpose performance samples for a specific configuration
struct cudecompTransposePerformanceSampleCollection {
  std::vector<cudecompTransposePerformanceSample> samples;
  int32_t sample_idx = 0;
  int32_t warmup_count = 0;
};

// Structure to contain data for halo performance sample
struct cudecompHaloPerformanceSample {
  cudecomp::cudaEventTimed halo_start_event;
  cudecomp::cudaEventTimed halo_end_event;
  cudecomp::cudaEventTimed sendrecv_start_event;
  cudecomp::cudaEventTimed sendrecv_end_event;
  size_t sendrecv_bytes = 0;
  bool valid = false;
};

// Collection of halo performance samples for a specific configuration
struct cudecompHaloPerformanceSampleCollection {
  std::vector<cudecompHaloPerformanceSample> samples;
  int32_t sample_idx = 0;
  int32_t warmup_count = 0;
};

// cuDecomp grid descriptor containing grid-specific information
struct cudecompGridDesc {
  ~cudecompGridDesc() noexcept {
    row_comm_info.reset();
    col_comm_info.reset();
#ifdef ENABLE_NVSHMEM
    if (nvshmem_block_counters) { cudaFree(nvshmem_block_counters); }
#endif
  }

  cudecompHandle_t handle = nullptr;    // owning cuDecomp handle
  cudecompGridDescConfig_t config;      // configuration struct
  bool gdims_dist_set = false;          // flag to record if gdims_dist was set to non-default values
  bool transpose_mem_order_set = false; // flag to record if transpose_mem_order was set to non-default values

  int32_t pidx[2]; // processor grid index;

  cudecompCommInfo row_comm_info; // row communicator information
  cudecompCommInfo col_comm_info; // column communicator information

  std::vector<cudecomp::cudaEvent> events; // CUDA events used for scheduling
  cudecomp::cudaEvent nvshmem_sync_event;  // NVSHMEM event used for synchronization

#ifdef ENABLE_NVSHMEM
  int* nvshmem_block_counters = nullptr;    // device memory counters for SM alltoallv last-block detection
  cudecomp::nvshmemRuntime nvshmem_runtime; // Shared reference to initialized NVSHMEM runtime
#endif

  cudecomp::graphCache graph_cache; // CUDA graph cache

  cudecomp::ncclComm nccl_comm; // NCCL communicator (global), shared from handle
  cudecomp::ncclComm
      nccl_local_comm; // NCCL communicator (intra-node, or intra-clique on MNNVL systems), shared from handle

  // Performance reporting related entries
  std::vector<cudecomp::cudaEventTimed> alltoall_start_events; // events for alltoall timing
  std::vector<cudecomp::cudaEventTimed> alltoall_end_events;   // events for alltoall timing
  int32_t alltoall_timing_count = 0;              // count of alltoall timing events pairs (for pipelined alltoall)
  cudecomp::cudaEventTimed transpose_start_event; // event for transpose timing
  cudecomp::cudaEventTimed transpose_end_event;   // event for transpose timing

  std::unordered_map<std::tuple<int32_t, int32_t, std::array<int32_t, 3>, std::array<int32_t, 3>,
                                std::array<int32_t, 3>, std::array<int32_t, 3>, bool, bool, cudecompDataType_t>,
                     cudecompTransposePerformanceSampleCollection>
      transpose_perf_samples_map;

  std::unordered_map<std::tuple<int32_t, int32_t, std::array<int32_t, 3>, std::array<bool, 3>, std::array<int32_t, 3>,
                                bool, cudecompDataType_t>,
                     cudecompHaloPerformanceSampleCollection>
      halo_perf_samples_map;

  bool initialized = false;
};

namespace cudecomp {

using comm_count_t = int64_t;

enum cudecompCommAxis { CUDECOMP_COMM_COL = 0, CUDECOMP_COMM_ROW = 1 };

static inline void setProcessGridIndex(const cudecompHandle_t handle, cudecompGridDesc_t grid_desc) {
  switch (grid_desc->config.rank_order) {
  case CUDECOMP_RANK_ORDER_COL_MAJOR:
    grid_desc->pidx[0] = handle->rank % grid_desc->config.pdims[0];
    grid_desc->pidx[1] = handle->rank / grid_desc->config.pdims[0];
    break;
  case CUDECOMP_RANK_ORDER_DEFAULT:
  case CUDECOMP_RANK_ORDER_ROW_MAJOR:
  default:
    grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
    grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];
    break;
  }
}

// Helper function to convert row or column rank to global rank
static inline int getGlobalRank(const cudecompHandle_t, const cudecompGridDesc_t grid_desc, cudecompCommAxis axis,
                                int axis_rank) {
  switch (grid_desc->config.rank_order) {
  case CUDECOMP_RANK_ORDER_COL_MAJOR:
    return (axis == CUDECOMP_COMM_ROW) ? grid_desc->pidx[0] + axis_rank * grid_desc->config.pdims[0]
                                       : grid_desc->config.pdims[0] * grid_desc->pidx[1] + axis_rank;
  case CUDECOMP_RANK_ORDER_DEFAULT:
  case CUDECOMP_RANK_ORDER_ROW_MAJOR:
  default:
    return (axis == CUDECOMP_COMM_ROW) ? grid_desc->config.pdims[1] * grid_desc->pidx[0] + axis_rank
                                       : grid_desc->pidx[1] + axis_rank * grid_desc->config.pdims[1];
  }
}

// Helper function to return maximum pencil size across all processes for a given axis
static inline int64_t getGlobalMaxPencilSize(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                                             int32_t axis) {
  int64_t size = 1;
  int j = 0;
  for (int i = 0; i < 3; ++i) {
    if (i != axis) {
      int64_t dim = (grid_desc->config.gdims_dist[i] + grid_desc->config.pdims[j] - 1) / grid_desc->config.pdims[j];
      // Tack any difference in gdim and gdim_dist to last pencil in gdim_dist decomposition
      dim += (grid_desc->config.gdims[i] - grid_desc->config.gdims_dist[i]);
      size *= dim;
      j++;
    } else {
      size *= grid_desc->config.gdims[i];
    }
  }

  return size;
}

// Helper function to return pointer offset at local index lx (in global order)
static size_t getPencilPtrOffset(const cudecompPencilInfo_t& pinfo, const std::array<int32_t, 3>& lx) {
  return lx[pinfo.order[0]] + lx[pinfo.order[1]] * pinfo.shape[0] +
         lx[pinfo.order[2]] * pinfo.shape[0] * pinfo.shape[1];
}

// Helper function to return shape in global order
static inline std::array<int32_t, 3> getShapeG(const cudecompPencilInfo_t& pinfo) {
  std::array<int32_t, 3> shape_g;
  for (int i = 0; i < 3; ++i) {
    shape_g[pinfo.order[i]] = pinfo.shape[i];
  }
  return shape_g;
}

// Function to compute the GCD using the Euclidean algorithm
static inline int gcd(int a, int b) {
  while (b != 0) {
    int temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

static inline bool transposeBackendRequiresMpi(cudecompTransposeCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_TRANSPOSE_COMM_MPI_P2P || comm_backend == CUDECOMP_TRANSPOSE_COMM_MPI_A2A ||
          comm_backend == CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL);
}

static inline bool haloBackendRequiresMpi(cudecompHaloCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_HALO_COMM_MPI || comm_backend == CUDECOMP_HALO_COMM_MPI_BLOCKING);
}

static inline bool transposeBackendRequiresNccl(cudecompTransposeCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_TRANSPOSE_COMM_NCCL || comm_backend == CUDECOMP_TRANSPOSE_COMM_NCCL_PL);
}

static inline bool haloBackendRequiresNccl(cudecompHaloCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_HALO_COMM_NCCL);
}

static inline bool transposeBackendRequiresNvshmem(cudecompTransposeCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_TRANSPOSE_COMM_NVSHMEM || comm_backend == CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL ||
          comm_backend == CUDECOMP_TRANSPOSE_COMM_NVSHMEM_SM);
}

static inline bool haloBackendRequiresNvshmem(cudecompHaloCommBackend_t comm_backend) {
  return (comm_backend == CUDECOMP_HALO_COMM_NVSHMEM || comm_backend == CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING);
}

static inline bool isManagedPointer(void* ptr) {
  // Check if input pointer is managed
  cudaPointerAttributes attr;
  CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));
  return attr.type == cudaMemoryTypeManaged;
}

static void setCommInfo(cudecompHandle_t& handle, cudecompGridDesc_t& grid_desc, MPI_Comm mpi_comm,
                        cudecompCommAxis comm_axis) {
  auto& info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;

  info.reset();
  info.mpi_comm = mpi_comm;
  CHECK_MPI(MPI_Comm_rank(info.mpi_comm, &info.rank));
  CHECK_MPI(MPI_Comm_size(info.mpi_comm, &info.nranks));

  int count = 0;

  if (handle->rank_to_clique.size() == 0) {
    // Count occurences of hostname in row/col communicator
    std::map<std::string, int> host_counts;
    for (int i = 0; i < info.nranks; ++i) {
      int peer_rank_global = getGlobalRank(handle, grid_desc, comm_axis, i);
      std::string hostname = std::string(handle->hostnames[peer_rank_global].data());
      host_counts[hostname]++;
    }

    // Find largest homogeneous peer-to-peer group size. This can be smaller than
    // a full node when running across nodes with different GPU counts.
    for (const auto& e : host_counts) {
      if (count == 0) {
        count = e.second;
      } else {
        if (count != e.second) { count = gcd(count, e.second); }
      }
    }
  } else {
    // For MNNVL configurations, count occurences of clique in row/col communicator
    std::map<unsigned int, int> clique_counts;
    for (int i = 0; i < info.nranks; ++i) {
      int peer_rank_global = getGlobalRank(handle, grid_desc, comm_axis, i);
      unsigned int clique = handle->rank_to_clique[peer_rank_global];
      clique_counts[clique]++;
    }

    // Find largest homogeneous peer-to-peer group size. This can be smaller than
    // a full clique when running across multiple cliques of differing sizes.
    for (const auto& e : clique_counts) {
      if (count == 0) {
        count = e.second;
      } else {
        if (count != e.second) { count = gcd(count, e.second); }
      }
    }

    // Check if any cliques contain multiple nodes (i.e. there are MNNVL connections in this communicator)
    std::map<unsigned int, std::string> clique_to_hostname;
    for (int i = 0; i < info.nranks; ++i) {
      int peer_rank_global = getGlobalRank(handle, grid_desc, comm_axis, i);
      unsigned int clique = handle->rank_to_clique[peer_rank_global];
      std::string hostname = std::string(handle->hostnames[peer_rank_global].data());
      if (clique_to_hostname.count(clique)) {
        if (clique_to_hostname[clique] != hostname) {
          // Multiple hostnames in clique detected, MNNVL connections are present
          info.mnnvl_active = true;
          break;
        }
      } else {
        clique_to_hostname[clique] = hostname;
      }
    }
  }

  info.npergroup = count;
  info.ngroups = info.nranks / info.npergroup;
}

static void createCommInfo(cudecompHandle_t& handle, cudecompGridDesc_t& grid_desc, bool need_nvshmem = false) {
  grid_desc->row_comm_info.reset();
  grid_desc->col_comm_info.reset();

  setProcessGridIndex(handle, grid_desc);

  MPI_Comm row_comm;
  CHECK_MPI(MPI_Comm_split(handle->mpi_comm, grid_desc->pidx[0], handle->rank, &row_comm));
  setCommInfo(handle, grid_desc, row_comm, CUDECOMP_COMM_ROW);

  MPI_Comm col_comm;
  CHECK_MPI(MPI_Comm_split(handle->mpi_comm, grid_desc->pidx[1], handle->rank, &col_comm));
  setCommInfo(handle, grid_desc, col_comm, CUDECOMP_COMM_COL);

#ifdef ENABLE_NVSHMEM
  if (need_nvshmem) {
    nvshmem_team_config_t tmp;
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, grid_desc->config.pdims[1], &tmp, 0,
                          &grid_desc->row_comm_info.nvshmem_team, &tmp, 0, &grid_desc->col_comm_info.nvshmem_team);

    grid_desc->row_comm_info.nvshmem_signals =
        (uint64_t*)nvshmem_malloc(grid_desc->row_comm_info.nranks * sizeof(uint64_t));
    if (!grid_desc->row_comm_info.nvshmem_signals) { THROW_NVSHMEM_ERROR("nvshmem_malloc failed"); }
    CHECK_CUDA(
        cudaMemset(grid_desc->row_comm_info.nvshmem_signals, 0, grid_desc->row_comm_info.nranks * sizeof(uint64_t)));

    grid_desc->col_comm_info.nvshmem_signals =
        (uint64_t*)nvshmem_malloc(grid_desc->col_comm_info.nranks * sizeof(uint64_t));
    if (!grid_desc->col_comm_info.nvshmem_signals) { THROW_NVSHMEM_ERROR("nvshmem_malloc failed"); }
    CHECK_CUDA(
        cudaMemset(grid_desc->col_comm_info.nvshmem_signals, 0, grid_desc->col_comm_info.nranks * sizeof(uint64_t)));
  }
#else
  if (need_nvshmem) { THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends."); }
#endif
}

static inline void getAlltoallPeerRanks(cudecompGridDesc_t grid_desc, cudecompCommAxis comm_axis, int iter,
                                        int& src_rank, int& dst_rank) {

  const auto& info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;

  // Return self for single rank case or when iter is zero
  if (info.nranks == 1 || iter == 0) {
    src_rank = info.rank;
    dst_rank = info.rank;
    return;
  }

  bool power_of_2 = !(info.nranks & info.nranks - 1);

  // Mixing up transfer ordering to distribute intra-group transfers between inter-group ones.
  if (iter % 2 == 1) {
    iter = iter / 2 + 1;
  } else {
    iter = info.nranks / 2 + iter / 2;
  }
  if (power_of_2) {
    // Case 1: power of two size, use XOR communication pattern
    dst_rank = info.rank ^ iter;
    src_rank = info.rank ^ iter;
  } else {
    // Case 2: not a power of two size, use multi-level ring
    // communication pattern (intra-group rings followed by inter-group rings)
    int groupid = info.rank / info.npergroup;
    if (iter < info.npergroup) {
      dst_rank = (info.rank + iter) % info.npergroup + groupid * info.npergroup;
      src_rank = (info.rank + info.npergroup - iter) % info.npergroup + groupid * info.npergroup;
    } else {
      dst_rank = (info.rank + iter) % info.nranks;
      if (dst_rank >= groupid * info.npergroup and dst_rank < (groupid + 1) * info.npergroup) {
        dst_rank += info.npergroup;
        dst_rank %= info.nranks;
      }
      src_rank = (info.rank + info.nranks - iter) % info.nranks;
      if (src_rank >= groupid * info.npergroup and src_rank < (groupid + 1) * info.npergroup) {
        src_rank += (info.nranks - info.npergroup);
        src_rank %= info.nranks;
      }
    }
  }
}

static inline std::vector<int64_t> getSplits(int64_t N, int nchunks, int pad) {
  std::vector<int64_t> splits(nchunks, N / nchunks);
  for (int i = 0; i < N % nchunks; ++i) {
    splits[i] += 1;
  }

  // Add padding to last populated pencil
  splits[std::min(N, (int64_t)nchunks) - 1] += pad;

  return splits;
}

template <typename T> static inline bool anyNonzeros(const std::array<T, 3>& arr) {
  return (arr[0] != T(0) || arr[1] != T(0) || arr[2] != T(0));
}

// Assigns an integer ID to every unique value in a vector
template <typename T> std::unordered_map<T, unsigned int> getUniqueIds(const std::vector<T>& v) {
  std::unordered_map<T, unsigned int> ids;
  for (const auto& e : v) {
    if (ids.count(e) == 0) { ids[e] = static_cast<unsigned int>(ids.size()); }
  }
  return ids;
}

// Custom deleter for NCCL communicators
struct ncclCommDeleter {
  void operator()(ncclComm_t* comm) {
    if (comm && *comm) {
      CHECK_NCCL(ncclCommDestroy(*comm));
      delete comm;
    }
  }
};

// Helper function to create a shared_ptr NCCL communicator
static inline ncclComm createNcclComm(ncclComm_t comm) {
  return std::shared_ptr<ncclComm_t>(new ncclComm_t(comm), ncclCommDeleter());
}

// Helper to check if pencil axis has empty pencils
static inline bool checkForEmptyPencils(const cudecompGridDesc_t grid_desc, int axis) {
  int j = 0;
  for (int i = 0; i < 3; ++i) {
    if (i != axis) {
      int64_t d = grid_desc->config.gdims_dist[i] / grid_desc->config.pdims[j];
      if (d == 0) { return true; }
      j++;
    }
  }
  return false;
}

// Workspace buffer alignment in bytes
static constexpr int CUDECOMP_WORKSPACE_ALIGN_BYTES = 256;

// Helper to round element count to nearest multiple of nbytes (assuming smallest supported type float)
static inline int64_t alignCountToBytes(int64_t count, int nbytes) {
  int64_t count_bytes = count * sizeof(float);
  int64_t count_bytes_rounded = ((count_bytes + nbytes - 1) / nbytes) * nbytes;
  return count_bytes_rounded / sizeof(float);
}

} // namespace cudecomp

#endif // CUDECOMP_COMMON_H
