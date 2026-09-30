! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program fortran_cpu_api_test
  use cudecomp
  use mpi_f08
  implicit none
  type(cudecompHandle) :: handle
  integer :: ierr

  call MPI_Init(ierr)
  call check(cudecompInit(handle, MPI_COMM_WORLD))
  call check(cudecompFinalize(handle))
  call check(cudecompInit(handle, MPI_COMM_WORLD%MPI_VAL))
  call check(cudecompFinalize(handle))
  call MPI_Finalize(ierr)
contains
  subroutine check(result)
    integer, intent(in) :: result
    if (result /= CUDECOMP_RESULT_SUCCESS) call MPI_Abort(MPI_COMM_WORLD, result, ierr)
  end subroutine
end program
