/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backend_utils.h"

#include "backend_config.h"

namespace cudecomp_test {

std::vector<TransposeBackend> transposeBackends() {
  std::vector<TransposeBackend> backends = {
      {CUDECOMP_TRANSPOSE_COMM_MPI_P2P, "mpi-p2p", "mpi"},  {CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL, "mpi-p2p-pl", "mpi"},
      {CUDECOMP_TRANSPOSE_COMM_MPI_A2A, "mpi-a2a", "mpi"},  {CUDECOMP_TRANSPOSE_COMM_NCCL, "nccl", "nccl"},
      {CUDECOMP_TRANSPOSE_COMM_NCCL_PL, "nccl-pl", "nccl"},
  };

#if CUDECOMP_TEST_ENABLE_NVSHMEM
  backends.push_back({CUDECOMP_TRANSPOSE_COMM_NVSHMEM, "nvshmem", "nvshmem"});
  backends.push_back({CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL, "nvshmem-pl", "nvshmem"});
  backends.push_back({CUDECOMP_TRANSPOSE_COMM_NVSHMEM_SM, "nvshmem-sm", "nvshmem"});
#endif

  return backends;
}

std::vector<HaloBackend> haloBackends() {
  std::vector<HaloBackend> backends = {
      {CUDECOMP_HALO_COMM_MPI, "mpi", "mpi"},
      {CUDECOMP_HALO_COMM_MPI_BLOCKING, "mpi-blocking", "mpi"},
      {CUDECOMP_HALO_COMM_NCCL, "nccl", "nccl"},
  };

#if CUDECOMP_TEST_ENABLE_NVSHMEM
  backends.push_back({CUDECOMP_HALO_COMM_NVSHMEM, "nvshmem", "nvshmem"});
  backends.push_back({CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING, "nvshmem-blocking", "nvshmem"});
#endif

  return backends;
}

} // namespace cudecomp_test
