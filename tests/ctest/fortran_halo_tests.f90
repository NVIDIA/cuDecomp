! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

module cudecomp_fortran_halo_tests
  use, intrinsic :: iso_fortran_env, only: int64, real32, real64
  use cudafor
  use cudecomp
  use mpi

  implicit none

  ! Standalone CTest fixture for the Fortran halo API. The included fixture body
  ! is instantiated once per dtype so the axis and optional-padding matrix stays
  ! identical across real and complex coverage.
  integer, parameter :: halo_test_ranks = 4
  integer, parameter :: k_gdims(3) = [9, 10, 11]
  integer, parameter :: k_gdims_dist(3) = [8, 8, 8]
  integer, parameter :: k_pdims(2) = [2, 2]
  integer, parameter :: k_halo_extents(3) = [1, 1, 1]
  integer, parameter :: k_zero_extents(3) = [0, 0, 0]
  integer, parameter :: k_explicit_padding(3) = [1, 0, 2]
  logical, parameter :: k_halo_periods(3) = [.true., .false., .true.]
  logical, parameter :: k_default_axis_contiguous(3) = [.false., .false., .false.]
  integer, parameter :: k_default_mem_order(3, 3) = reshape([-1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 3])
  integer, parameter :: k_mixed_mem_order(3, 3) = reshape([3, 2, 1, 3, 2, 1, 3, 2, 1], [3, 3])

  integer :: rank = -1
  integer :: nranks = 0
  integer :: ierr = 0
  integer :: local_comm = MPI_COMM_NULL
  integer :: local_rank = 0
  integer :: failures = 0
  integer :: global_failures = 0
  logical :: handle_initialized = .false.
  type(cudecompHandle) :: handle

contains

  subroutine run_all_tests()
    implicit none

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

    if (nranks /= halo_test_ranks) then
      call record_failure("Fortran halo test requires exactly 4 MPI ranks")
    else
      call initialize_gpu()
    endif

    if (failures == 0) then
      call expect_success(cudecompInit(handle, MPI_COMM_WORLD), "cudecompInit")
      handle_initialized = failures == 0
    endif

    if (handle_initialized) then
      call run_halo_r32()
      call run_halo_r64()
      call run_halo_c32()
      call run_halo_c64()
      call expect_success(cudecompFinalize(handle), "cudecompFinalize")
    endif

    call MPI_Allreduce(failures, global_failures, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (rank == 0 .and. global_failures == 0) then
      write(*, '("Fortran halo test passed")')
    elseif (rank == 0) then
      write(*, '("Fortran halo test failed with ", i0, " rank-local failure(s)")') global_failures
    endif

    if (local_comm /= MPI_COMM_NULL) call MPI_Comm_free(local_comm, ierr)
    call MPI_Finalize(ierr)
    if (global_failures /= 0) call exit(1)
  end subroutine run_all_tests

  subroutine initialize_gpu()
    implicit none

    integer :: status
    integer :: num_devices

    call MPI_Comm_split_Type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
    call MPI_Comm_rank(local_comm, local_rank, ierr)

    status = cudaGetDeviceCount(num_devices)
    if (status /= cudaSuccess) then
      call record_failure("cudaGetDeviceCount failed")
      return
    endif
    if (num_devices <= 0) then
      call record_failure("Fortran halo test requires at least one visible CUDA device")
      return
    endif

    status = cudaSetDevice(mod(local_rank, num_devices))
    if (status /= cudaSuccess) call record_failure("cudaSetDevice failed")
  end subroutine initialize_gpu

  subroutine setup_halo_config(config, axis_contiguous, mem_order, rank_order)
    implicit none

    type(cudecompGridDescConfig), intent(out) :: config
    logical, intent(in) :: axis_contiguous(3)
    integer, intent(in) :: mem_order(3, 3)
    integer, intent(in) :: rank_order

    call expect_success(cudecompGridDescConfigSetDefaults(config), "cudecompGridDescConfigSetDefaults")
    config%gdims = k_gdims
    config%gdims_dist = k_gdims_dist
    config%pdims = k_pdims
    config%rank_order = rank_order
    config%halo_comm_backend = CUDECOMP_HALO_COMM_MPI
    config%transpose_axis_contiguous = axis_contiguous
    config%transpose_mem_order = mem_order
  end subroutine setup_halo_config

  function axis_name(axis) result(name)
    implicit none

    integer, intent(in) :: axis
    character(len=1) :: name

    select case (axis)
    case (1)
      name = "X"
    case (2)
      name = "Y"
    case (3)
      name = "Z"
    case default
      name = "?"
    end select
  end function axis_name

  function pencil_lower(pinfo, dim) result(lower)
    implicit none

    type(cudecompPencilInfo), intent(in) :: pinfo
    integer, intent(in) :: dim
    integer(int64) :: lower

    lower = int(pinfo%lo(dim), int64) - int(pinfo%halo_extents(pinfo%order(dim)), int64)
  end function pencil_lower

  subroutine local_coordinate(linear_index, pinfo, local)
    implicit none

    integer(int64), intent(in) :: linear_index
    type(cudecompPencilInfo), intent(in) :: pinfo
    integer(int64), intent(out) :: local(3)
    integer(int64) :: offset
    integer(int64) :: shape_1
    integer(int64) :: shape_2

    offset = linear_index - 1_int64
    shape_1 = int(pinfo%shape(1), int64)
    shape_2 = int(pinfo%shape(2), int64)
    local(1) = pencil_lower(pinfo, 1) + modulo(offset, shape_1)
    local(2) = pencil_lower(pinfo, 2) + modulo(offset / shape_1, shape_2)
    local(3) = pencil_lower(pinfo, 3) + offset / (shape_1 * shape_2)
  end subroutine local_coordinate

  function is_internal_coordinate(pinfo, local) result(is_internal)
    implicit none

    type(cudecompPencilInfo), intent(in) :: pinfo
    integer(int64), intent(in) :: local(3)
    logical :: is_internal

    is_internal = local(1) >= int(pinfo%lo(1), int64) .and. local(1) <= int(pinfo%hi(1), int64) .and. &
                  local(2) >= int(pinfo%lo(2), int64) .and. local(2) <= int(pinfo%hi(2), int64) .and. &
                  local(3) >= int(pinfo%lo(3), int64) .and. local(3) <= int(pinfo%hi(3), int64)
  end function is_internal_coordinate

  function is_padding_coordinate(pinfo, local) result(is_padding)
    implicit none

    type(cudecompPencilInfo), intent(in) :: pinfo
    integer(int64), intent(in) :: local(3)
    logical :: is_padding

    is_padding = local(1) > int(pinfo%hi(1), int64) + int(pinfo%halo_extents(pinfo%order(1)), int64) .or. &
                 local(2) > int(pinfo%hi(2), int64) + int(pinfo%halo_extents(pinfo%order(2)), int64) .or. &
                 local(3) > int(pinfo%hi(3), int64) + int(pinfo%halo_extents(pinfo%order(3)), int64)
  end function is_padding_coordinate

  subroutine global_coordinate(pinfo, local, global)
    implicit none

    type(cudecompPencilInfo), intent(in) :: pinfo
    integer(int64), intent(in) :: local(3)
    integer(int64), intent(out) :: global(3)
    integer :: dim

    do dim = 1, 3
      global(pinfo%order(dim)) = local(dim)
    enddo
  end subroutine global_coordinate

  function wrap_index(index, size) result(wrapped)
    implicit none

    integer(int64), intent(in) :: index
    integer(int64), intent(in) :: size
    integer(int64) :: wrapped

    wrapped = modulo(index - 1_int64, size) + 1_int64
  end function wrap_index

  function global_linear_index(global) result(linear_index)
    implicit none

    integer(int64), intent(in) :: global(3)
    integer(int64) :: linear_index

    linear_index = global(1) + int(k_gdims(1), int64) * &
                              ((global(2) - 1_int64) + (global(3) - 1_int64) * int(k_gdims(2), int64))
  end function global_linear_index

  subroutine expect_success(result, context)
    implicit none

    integer, intent(in) :: result
    character(len=*), intent(in) :: context

    if (result /= CUDECOMP_RESULT_SUCCESS) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "returned", result
      failures = failures + 1
    endif
  end subroutine expect_success

  subroutine record_failure(context)
    implicit none

    character(len=*), intent(in) :: context

    write(*, *) "rank", rank, ": FAIL", trim(context)
    failures = failures + 1
  end subroutine record_failure

#define ARRTYPE real(real32)
#define DTYPE CUDECOMP_FLOAT
#define DTYPE_NAME "R32"
#define RUN_HALO_CASE run_halo_r32
#define RUN_HALO_SCENARIO run_halo_scenario_r32
#define RUN_HALO_AXIS run_halo_axis_r32
#define INITIALIZE_HALO_PENCIL initialize_halo_pencil_r32
#define INITIALIZE_HALO_REFERENCE initialize_halo_reference_r32
#define HALO_PENCIL_VALUE halo_pencil_value_r32
#define UNSET_HALO_VALUE unset_halo_value_r32
#define EXPECT_HALO_PENCIL_MATCH expect_halo_pencil_match_r32
#include "fortran_halo_case.inc"
#undef EXPECT_HALO_PENCIL_MATCH
#undef UNSET_HALO_VALUE
#undef HALO_PENCIL_VALUE
#undef INITIALIZE_HALO_REFERENCE
#undef INITIALIZE_HALO_PENCIL
#undef RUN_HALO_AXIS
#undef RUN_HALO_SCENARIO
#undef RUN_HALO_CASE
#undef DTYPE_NAME
#undef DTYPE
#undef ARRTYPE

#define ARRTYPE real(real64)
#define DTYPE CUDECOMP_DOUBLE
#define DTYPE_NAME "R64"
#define RUN_HALO_CASE run_halo_r64
#define RUN_HALO_SCENARIO run_halo_scenario_r64
#define RUN_HALO_AXIS run_halo_axis_r64
#define INITIALIZE_HALO_PENCIL initialize_halo_pencil_r64
#define INITIALIZE_HALO_REFERENCE initialize_halo_reference_r64
#define HALO_PENCIL_VALUE halo_pencil_value_r64
#define UNSET_HALO_VALUE unset_halo_value_r64
#define EXPECT_HALO_PENCIL_MATCH expect_halo_pencil_match_r64
#include "fortran_halo_case.inc"
#undef EXPECT_HALO_PENCIL_MATCH
#undef UNSET_HALO_VALUE
#undef HALO_PENCIL_VALUE
#undef INITIALIZE_HALO_REFERENCE
#undef INITIALIZE_HALO_PENCIL
#undef RUN_HALO_AXIS
#undef RUN_HALO_SCENARIO
#undef RUN_HALO_CASE
#undef DTYPE_NAME
#undef DTYPE
#undef ARRTYPE

#define ARRTYPE complex(real32)
#define DTYPE CUDECOMP_FLOAT_COMPLEX
#define DTYPE_NAME "C32"
#define RUN_HALO_CASE run_halo_c32
#define RUN_HALO_SCENARIO run_halo_scenario_c32
#define RUN_HALO_AXIS run_halo_axis_c32
#define INITIALIZE_HALO_PENCIL initialize_halo_pencil_c32
#define INITIALIZE_HALO_REFERENCE initialize_halo_reference_c32
#define HALO_PENCIL_VALUE halo_pencil_value_c32
#define UNSET_HALO_VALUE unset_halo_value_c32
#define EXPECT_HALO_PENCIL_MATCH expect_halo_pencil_match_c32
#include "fortran_halo_case.inc"
#undef EXPECT_HALO_PENCIL_MATCH
#undef UNSET_HALO_VALUE
#undef HALO_PENCIL_VALUE
#undef INITIALIZE_HALO_REFERENCE
#undef INITIALIZE_HALO_PENCIL
#undef RUN_HALO_AXIS
#undef RUN_HALO_SCENARIO
#undef RUN_HALO_CASE
#undef DTYPE_NAME
#undef DTYPE
#undef ARRTYPE

#define ARRTYPE complex(real64)
#define DTYPE CUDECOMP_DOUBLE_COMPLEX
#define DTYPE_NAME "C64"
#define RUN_HALO_CASE run_halo_c64
#define RUN_HALO_SCENARIO run_halo_scenario_c64
#define RUN_HALO_AXIS run_halo_axis_c64
#define INITIALIZE_HALO_PENCIL initialize_halo_pencil_c64
#define INITIALIZE_HALO_REFERENCE initialize_halo_reference_c64
#define HALO_PENCIL_VALUE halo_pencil_value_c64
#define UNSET_HALO_VALUE unset_halo_value_c64
#define EXPECT_HALO_PENCIL_MATCH expect_halo_pencil_match_c64
#include "fortran_halo_case.inc"
#undef EXPECT_HALO_PENCIL_MATCH
#undef UNSET_HALO_VALUE
#undef HALO_PENCIL_VALUE
#undef INITIALIZE_HALO_REFERENCE
#undef INITIALIZE_HALO_PENCIL
#undef RUN_HALO_AXIS
#undef RUN_HALO_SCENARIO
#undef RUN_HALO_CASE
#undef DTYPE_NAME
#undef DTYPE
#undef ARRTYPE

end module cudecomp_fortran_halo_tests

program cudecomp_fortran_halo_test
  use cudecomp_fortran_halo_tests

  implicit none

  call run_all_tests()
end program cudecomp_fortran_halo_test
