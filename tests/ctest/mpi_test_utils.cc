/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mpi_test_utils.h"

namespace cudecomp_test {

MpiTestComm::MpiTestComm(MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) return;

  MPI_Comm_dup(comm, &comm_);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm_);
  MPI_Comm_rank(local_comm_, &local_rank_);
  MPI_Comm_size(local_comm_, &local_size_);
}

MpiTestComm::MpiTestComm(MpiTestComm&& other) noexcept
    : comm_(other.comm_), local_comm_(other.local_comm_), rank_(other.rank_), size_(other.size_),
      local_rank_(other.local_rank_), local_size_(other.local_size_) {
  other.comm_ = MPI_COMM_NULL;
  other.local_comm_ = MPI_COMM_NULL;
  other.rank_ = -1;
  other.size_ = 0;
  other.local_rank_ = -1;
  other.local_size_ = 0;
}

MpiTestComm& MpiTestComm::operator=(MpiTestComm&& other) noexcept {
  if (this != &other) {
    reset();
    comm_ = other.comm_;
    local_comm_ = other.local_comm_;
    rank_ = other.rank_;
    size_ = other.size_;
    local_rank_ = other.local_rank_;
    local_size_ = other.local_size_;

    other.comm_ = MPI_COMM_NULL;
    other.local_comm_ = MPI_COMM_NULL;
    other.rank_ = -1;
    other.size_ = 0;
    other.local_rank_ = -1;
    other.local_size_ = 0;
  }
  return *this;
}

MpiTestComm::~MpiTestComm() { reset(); }

MpiTestComm MpiTestComm::world() { return fromComm(MPI_COMM_WORLD); }

MpiTestComm MpiTestComm::split(const MpiTestComm& parent_comm, int requested_ranks) {
  const bool valid_request = requested_ranks > 0 && requested_ranks <= parent_comm.size();
  const bool active = valid_request && parent_comm.rank() < requested_ranks;

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(parent_comm.mpiComm(), active ? 0 : MPI_UNDEFINED, parent_comm.rank(), &comm);

  MpiTestComm result = fromComm(comm);
  if (comm != MPI_COMM_NULL) { MPI_Comm_free(&comm); }
  return result;
}

MpiTestComm MpiTestComm::fromComm(MPI_Comm comm) { return MpiTestComm(comm); }

MPI_Comm MpiTestComm::mpiComm() const { return comm_; }

MPI_Comm MpiTestComm::localComm() const { return local_comm_; }

bool MpiTestComm::valid() const { return comm_ != MPI_COMM_NULL; }

int MpiTestComm::rank() const { return rank_; }

int MpiTestComm::size() const { return size_; }

int MpiTestComm::localRank() const { return local_rank_; }

int MpiTestComm::localSize() const { return local_size_; }

void MpiTestComm::reset() {
  int finalized = 0;
  MPI_Finalized(&finalized);

  if (!finalized) {
    if (local_comm_ != MPI_COMM_NULL) { MPI_Comm_free(&local_comm_); }
    if (comm_ != MPI_COMM_NULL) { MPI_Comm_free(&comm_); }
  }

  local_comm_ = MPI_COMM_NULL;
  comm_ = MPI_COMM_NULL;
  rank_ = -1;
  size_ = 0;
  local_rank_ = -1;
  local_size_ = 0;
}

} // namespace cudecomp_test
