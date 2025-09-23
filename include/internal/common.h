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

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "cudecomp.h"
#include "internal/checks.h"
#include "internal/graph.h"

namespace cudecomp {
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
typedef std::pair<std::array<unsigned char, NVML_GPU_FABRIC_UUID_LEN>, unsigned int> mnnvl_info;
#else
typedef std::pair<std::array<unsigned char, 1>, unsigned int> mnnvl_info;
#endif
typedef std::shared_ptr<ncclComm_t> ncclComm;
} // namespace cudecomp

// cuDecomp handle containing general information
struct cudecompHandle {

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
  std::unordered_map<void*, std::vector<std::pair<ncclComm_t, void*>>>
      nccl_ubr_handles; // map of allocated buffer address to NCCL registration handle(s)

  std::vector<cudaStream_t> streams; // internal streams for concurrent scheduling

  cutensorHandle_t cutensor_handle; // cuTENSOR handle;
#if CUTENSOR_MAJOR >= 2
  cutensorPlanPreference_t cutensor_plan_pref; // cuTENSOR plan preference;
#endif

  std::vector<std::array<char, MPI_MAX_PROCESSOR_NAME>> hostnames; // list of hostnames by rank
  std::vector<int32_t> rank_to_local_rank;                         // list of local rank mappings

  bool initialized = false;

  // Entries for NVSHMEM management and warning generation
  bool nvshmem_initialized = false;                      // Flag to track NVSHMEM initialization
  int n_grid_descs_using_nvshmem = 0;                    // Count of grid descriptors using NVSHMEM
  bool nvshmem_mixed_buffer_warning_issued = false;      // Warn once if NVSHMEM buffer is used with MPI
  size_t nvshmem_symmetric_size;                         // NVSHMEM symmetric size
  bool nvshmem_vmm;                                      // Flag to track if NVSHMEM is using VMM allocations
  std::unordered_map<void*, size_t> nvshmem_allocations; // Table to record NVSHMEM allocations
  size_t nvshmem_allocation_size = 0;                    // Total of NVSHMEM allocations

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
  int32_t device_p2p_ce_count = 0;       // number of P2P CEs available
  bool use_col_major_rank_order = false; // Flag to control whether to use column-major rank order
};

// Structure with information about row/column communicator
struct cudecompCommInfo {
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  int32_t rank;
  int32_t nranks;

  int32_t ngroups = 0;   // number of p2p groups (i.e. grouping of ranks with fast interconnect) in communicator
  int32_t npergroup = 0; // number of ranks per p2p group

#ifdef ENABLE_NVSHMEM
  nvshmem_team_t nvshmem_team = NVSHMEM_TEAM_INVALID;
#endif
};

// Structure to contain data for transpose performance sample
struct cudecompTransposePerformanceSample {
  cudaEvent_t transpose_start_event;
  cudaEvent_t transpose_end_event;
  std::vector<cudaEvent_t> alltoall_start_events;
  std::vector<cudaEvent_t> alltoall_end_events;
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
  cudaEvent_t halo_start_event;
  cudaEvent_t halo_end_event;
  cudaEvent_t sendrecv_start_event;
  cudaEvent_t sendrecv_end_event;
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
  cudecompGridDescConfig_t config;      // configuration struct
  bool gdims_dist_set = false;          // flag to record if gdims_dist was set to non-default values
  bool transpose_mem_order_set = false; // flag to record if transpose_mem_order was set to non-default values

  int32_t pidx[2]; // processor grid index;

  cudecompCommInfo row_comm_info; // row communicator information
  cudecompCommInfo col_comm_info; // column communicator information

  std::vector<cudaEvent_t> events{nullptr}; // CUDA events used for scheduling
  cudaEvent_t nvshmem_sync_event = nullptr; // NVSHMEM event used for synchronization

  cudecomp::graphCache graph_cache; // CUDA graph cache

  cudecomp::ncclComm nccl_comm; // NCCL communicator (global), shared from handle
  cudecomp::ncclComm
      nccl_local_comm; // NCCL communicator (intra-node, or intra-clique on MNNVL systems), shared from handle

  // Performance reporting related entries
  std::vector<cudaEvent_t> alltoall_start_events; // events for alltoall timing
  std::vector<cudaEvent_t> alltoall_end_events;   // events for alltoall timing
  int32_t alltoall_timing_count = 0;              // count of alltoall timing events pairs (for pipelined alltoall)
  cudaEvent_t transpose_start_event;              // event for transpose timing
  cudaEvent_t transpose_end_event;                // event for transpose timing

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

using comm_count_t = int32_t;

enum cudecompCommAxis { CUDECOMP_COMM_COL = 0, CUDECOMP_COMM_ROW = 1 };

// Helper function to convert row or column rank to global rank
static inline int getGlobalRank(const cudecompHandle_t handle, const cudecompGridDesc_t grid_desc,
                                cudecompCommAxis axis, int axis_rank) {
  if (handle->use_col_major_rank_order) {
    // Column-major rank order
    return (axis == CUDECOMP_COMM_ROW) ? grid_desc->pidx[0] + axis_rank * grid_desc->config.pdims[0]
                                       : grid_desc->config.pdims[0] * grid_desc->pidx[1] + axis_rank;
  } else {
    // Row-major rank order (default)
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
  return (comm_backend == CUDECOMP_TRANSPOSE_COMM_NVSHMEM || comm_backend == CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL);
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
  }

  info.npergroup = count;
  info.ngroups = info.nranks / info.npergroup;
}

static inline void getAlltoallPeerRanks(cudecompGridDesc_t grid_desc, cudecompCommAxis comm_axis, int iter,
                                        int& src_rank, int& dst_rank) {

  const auto& info = (comm_axis == CUDECOMP_COMM_ROW) ? grid_desc->row_comm_info : grid_desc->col_comm_info;

  // Quick return for single rank case
  if (info.nranks == 1) {
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

} // namespace cudecomp

#endif // CUDECOMP_COMMON_H
