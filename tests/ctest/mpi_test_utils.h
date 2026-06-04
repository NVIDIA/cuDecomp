/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDECOMP_TEST_MPI_TEST_UTILS_H
#define CUDECOMP_TEST_MPI_TEST_UTILS_H

#include <mpi.h>

namespace cudecomp_test {

class MpiTestComm {
public:
  MpiTestComm() = default;
  MpiTestComm(const MpiTestComm&) = delete;
  MpiTestComm& operator=(const MpiTestComm&) = delete;
  MpiTestComm(MpiTestComm&& other) noexcept;
  MpiTestComm& operator=(MpiTestComm&& other) noexcept;
  ~MpiTestComm();

  static MpiTestComm world();
  static MpiTestComm split(const MpiTestComm& parent_comm, int requested_ranks);
  static MpiTestComm fromComm(MPI_Comm comm);

  MPI_Comm mpiComm() const;
  MPI_Comm localComm() const;
  bool valid() const;
  int rank() const;
  int size() const;
  int localRank() const;
  int localSize() const;
  void reset();

private:
  explicit MpiTestComm(MPI_Comm comm);

  MPI_Comm comm_ = MPI_COMM_NULL;
  MPI_Comm local_comm_ = MPI_COMM_NULL;
  int rank_ = -1;
  int size_ = 0;
  int local_rank_ = -1;
  int local_size_ = 0;
};

} // namespace cudecomp_test

#endif
