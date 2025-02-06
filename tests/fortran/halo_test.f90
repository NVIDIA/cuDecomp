! SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) call exit(1)
#define CHECK_CUDA_EXIT(f) if (f /= cudaSuccess) call exit(1)

#ifdef R32
#define ARRTYPE real(real32)
#define DTYPE CUDECOMP_FLOAT
module halo_CUDECOMP_FLOAT_mod
#endif

#ifdef R64
#define ARRTYPE real(real64)
#define DTYPE CUDECOMP_DOUBLE
module halo_CUDECOMP_DOUBLE_mod
#endif

#ifdef C32
#define ARRTYPE complex(real32)
#define DTYPE CUDECOMP_FLOAT_COMPLEX
module halo_CUDECOMP_FLOAT_COMPLEX_mod
#endif

#ifdef C64
#define ARRTYPE complex(real64)
#define DTYPE CUDECOMP_DOUBLE_COMPLEX
module halo_CUDECOMP_DOUBLE_COMPLEX_mod
#endif

  use, intrinsic :: iso_fortran_env, only: real32, real64
  contains
  function compare_pencils(ref, res, pinfo) result(mismatch)
    use cudecomp
    implicit none
    type(cudecompPencilInfo) :: pinfo
    ARRTYPE :: ref(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)), &
                   pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)), &
                   pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3)))
    ARRTYPE :: res(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)), &
                   pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)), &
                   pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3)))

    logical :: mismatch
    mismatch = any(ref /= res)
  end function compare_pencils

  subroutine initialize_pencil(ref, pinfo, gdims)
    use cudecomp
    implicit none
    ARRTYPE, allocatable :: ref(:,:,:)

    type(cudecompPencilInfo) :: pinfo
    integer :: i, j, k
    integer :: gdims(3)
    integer :: gx(3)

    ! Allocate reference pencil with halo regions
    allocate(ref(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)), &
                 pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)), &
                 pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3))))

    ref = -1

    ! Iterate over internal region only
    do k = pinfo%lo(3), pinfo%hi(3)
      do j = pinfo%lo(2), pinfo%hi(2)
        do i = pinfo%lo(1), pinfo%hi(1)
          ! Compute ordered global coordinate
          gx(pinfo%order(1)) = i
          gx(pinfo%order(2)) = j
          gx(pinfo%order(3)) = k

          ! Set reference to global linear index
          ref(i,j,k) = gx(1) + gdims(1) * ((gx(2) - 1) + (gx(3) - 1) * gdims(2))

        end do
      end do
    end do
  end subroutine initialize_pencil

  subroutine initialize_reference(ref, pinfo, gdims, halo_periods)
    use cudecomp
    implicit none
    ARRTYPE, allocatable :: ref(:,:,:)

    type(cudecompPencilInfo) :: pinfo
    integer :: i, j, k, l
    integer :: gdims(3)
    integer :: gx(3)
    logical :: halo_periods(3)
    logical :: unset

    ! Allocate reference pencil with halo regions
    allocate(ref(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)), &
                 pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)), &
                 pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3))))

    ref = -1

    ! Iterate over entire pencil, set values including halo regions
    do k = pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)), pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3))
      do j = pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)), pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2))
        do i = pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)), pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1))
          ! Compute ordered global coordinate
          gx(pinfo%order(1)) = i
          gx(pinfo%order(2)) = j
          gx(pinfo%order(3)) = k

          ! Handle halo entries on global boundary and periodicity
          unset = .false.
          do l = 1, 3
            if (gx(pinfo%order(l)) < 1 .or. gx(pinfo%order(l)) > gdims(pinfo%order(l))) then
              if (halo_periods(pinfo%order(l))) then
                ! If halo entry is on boundary and periodic, wrap around index
                gx(pinfo%order(l)) = mod(gx(pinfo%order(l)) - 1 + gdims(pinfo%order(l)), gdims(pinfo%order(l))) + 1
              else
                ! If halo entry is on boundary but not periodic, mark entry for unset value (-1)
                unset = .true.
              end if
            end if
          end do

          ! Set reference to global linear index
          if (.not. unset) then
            ref(i,j,k) = gx(1) + gdims(1) * ((gx(2) - 1) + (gx(3) - 1) * gdims(2))
          end if

        end do
      end do
    end do
  end subroutine initialize_reference

  subroutine flat_copy(src, dst, count)
    implicit none
    ARRTYPE :: src(*)
    ARRTYPE, device :: dst(*)
    integer(8) :: count

    dst(1:count) = src(1:count)

  end subroutine flat_copy
end module

program main
  use cudafor
  use mpi
  use cudecomp
  use, intrinsic :: iso_fortran_env, only: real32, real64

#ifdef R32
  use halo_CUDECOMP_FLOAT_mod
#endif

#ifdef R64
  use halo_CUDECOMP_DOUBLE_mod
#endif

#ifdef C32
  use halo_CUDECOMP_FLOAT_COMPLEX_mod
#endif

#ifdef C64
  use halo_CUDECOMP_DOUBLE_COMPLEX_mod
#endif

  implicit none

  ! Command line arguments
  integer :: gx, gy, gz
  integer :: comm_backend
  logical :: axis_contiguous(3)
  integer :: gdims_dist(3)
  integer :: halo_extents(3)
  logical :: halo_periods(3)
  integer :: mem_order(3, 3)
  logical :: use_managed_memory
  integer :: pr, pc
  integer :: axis

  ! MPI
  integer :: rank, local_rank, nranks, ierr
  integer :: local_comm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options
  type(cudecompGridDesc) :: grid_desc
  type(cudecompPencilInfo) :: pinfo

  integer :: pdims(2)
  integer :: gdims(3)
  integer(8) :: data_num_elements, workspace_num_elements

  ! data
  ARRTYPE, allocatable :: ref(:, :, :), init(:, :, :), data(:)
  ARRTYPE, allocatable, device, target:: data_d(:)
  ARRTYPE, allocatable, managed, target:: data_m(:)
  ARRTYPE, pointer, device, contiguous :: work_d(:)
  ARRTYPE, pointer, device:: input(:)
  integer :: dtype = DTYPE

  integer :: i, j, k, l, idt, iarg
  integer :: skip_count
  character(len=16) :: arg

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  call MPI_Comm_split_Type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
  call MPI_Comm_rank(local_comm, local_rank, ierr)
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank))

  ! Parse command-line arguments
  gx = 256
  gy = 256
  gz = 256
  pr = 0
  pc = 0
  comm_backend = 0
  axis_contiguous(:) = .false.
  gdims_dist(:) = 0
  halo_extents(:) = 1
  halo_periods(:) = .true.
  axis = 1
  use_managed_memory = .false.

  skip_count = 0
  do i = 1, command_argument_count()
    if (skip_count > 0) then
      skip_count = skip_count - 1
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--gx')
        call get_command_argument(i+1, arg)
        read(arg, *) gx
        skip_count = 1
      case('--gy')
        call get_command_argument(i+1, arg)
        read(arg, *) gy
        skip_count = 1
      case('--gz')
        call get_command_argument(i+1, arg)
        read(arg, *) gz
        skip_count = 1
      case('--backend')
        call get_command_argument(i+1, arg)
        read(arg, *) comm_backend
        skip_count = 1
      case('--pr')
        call get_command_argument(i+1, arg)
        read(arg, *) pr
        skip_count = 1
      case('--pc')
        call get_command_argument(i+1, arg)
        read(arg, *) pc
        skip_count = 1
      case('--acx')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        axis_contiguous(1) = iarg
        skip_count = 1
      case('--acy')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        axis_contiguous(2) = iarg
        skip_count = 1
      case('--acz')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        axis_contiguous(3) = iarg
        skip_count = 1
      case('--gdx')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(1)
        skip_count = 1
      case('--gdy')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(2)
        skip_count = 1
      case('--gdz')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(3)
        skip_count = 1
      case('--hex')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(1)
        skip_count = 1
      case('--hey')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(2)
        skip_count = 1
      case('--hez')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(3)
        skip_count = 1
      case('--hpx')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        halo_periods(1) = iarg
        skip_count = 1
      case('--hpy')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        halo_periods(2) = iarg
        skip_count = 1
      case('--hpz')
        call get_command_argument(i+1, arg)
        read(arg, *) iarg
        halo_periods(3) = iarg
        skip_count = 1
      case('--ax')
        call get_command_argument(i+1, arg)
        read(arg, *) axis
        skip_count = 1
      case('--mem_order')
        l = 1
        do j = 1, 3
          do k = 1, 3
            call get_command_argument(i+l, arg)
            read(arg, *) mem_order(k, j)
            l = l + 1
          enddo
        enddo
        skip_count = 9
      case('-m')
        use_managed_memory = .true.
      case(' ')
        skip_count = 1
      case default
        print*, "Unknown argument."
        call exit(1)
    end select
  end do

  ! Finish setting up gdim_dist
  gdims_dist(1) = gx - gdims_dist(1)
  gdims_dist(2) = gy - gdims_dist(2)
  gdims_dist(3) = gz - gdims_dist(3)

  pdims(1) = pr
  pdims(2) = pc

  if (rank == 0) then
     write(*,"('Running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid ...')") gx, gy, gz
  end if

  ! Initialize cuDecomp
  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  ! Setup grid descriptor
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config))
  config%pdims = pdims
  gdims = [gx, gy, gz]
  config%gdims = gdims
  config%gdims_dist = gdims_dist
  config%transpose_axis_contiguous = axis_contiguous
  config%transpose_mem_order = mem_order

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
  options%halo_extents = halo_extents
  options%halo_periods = halo_periods
  options%halo_axis = axis
  options%dtype = dtype
  options%grid_mode = CUDECOMP_AUTOTUNE_GRID_HALO

  if (comm_backend /= 0) then
    config%halo_comm_backend = comm_backend
  else
    options%autotune_halo_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

  if (rank == 0) then
     write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
     write(*,"('running using ', a, ' halo backend ...')") &
               cudecompHaloCommBackendToString(config%halo_comm_backend)
  endif

  ! Get pencil information
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, pinfo, axis, halo_extents))

  ! Get workspace size
  CHECK_CUDECOMP_EXIT(cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents, workspace_num_elements))

  ! Allocate data arrays
  data_num_elements = pinfo%size
  allocate(data(data_num_elements))

  ! Create reference data
  call initialize_pencil(init, pinfo, gdims)
  call initialize_reference(ref, pinfo, gdims, halo_periods)

  if (use_managed_memory) then
    allocate(data_m(data_num_elements))
  else
    allocate(data_d(data_num_elements))
  endif
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements))

  ! Running correctness tests
  if (rank == 0) write(*,"('Running correctness tests using ', a, ' backend ...')") &
                         cudecompHaloCommBackendToString(config%halo_comm_backend)

  ! Initialize data to initial pencil data
  if (use_managed_memory) then
    call flat_copy(init, data_m, pinfo%size)
  else
    call flat_copy(init, data_d, pinfo%size)
  endif

  if (use_managed_memory) then
    input => data_m
  else
    input => data_d
  endif

  do i = 1, 3
    select case(axis)
      case(1)
        CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i))
      case(2)
        CHECK_CUDECOMP_EXIT(cudecompUpdateHalosY(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i))
      case(3)
        CHECK_CUDECOMP_EXIT(cudecompUpdateHalosZ(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i))
    end select
  end do

  data = input
  if (compare_pencils(ref, data, pinfo)) then
    print*, "FAILED cudecompUpdateHalos"
    call exit(1)
  endif

  if (use_managed_memory) then
    deallocate(data_m)
  else
    deallocate(data_d)
  endif
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))

  call MPI_Finalize(ierr)

end program main
