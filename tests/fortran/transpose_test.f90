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

#if defined(R32)
#define ARRTYPE real(real32)
#define DTYPE CUDECOMP_FLOAT
#define MODNAME transpose_CUDECOMP_FLOAT_mod
#elif defined(R64)
#define ARRTYPE real(real64)
#define DTYPE CUDECOMP_DOUBLE
#define MODNAME transpose_CUDECOMP_DOUBLE_mod
#elif defined(C32)
#define ARRTYPE complex(real32)
#define DTYPE CUDECOMP_FLOAT_COMPLEX
#define MODNAME transpose_CUDECOMP_FLOAT_COMPLEX_mod
#elif defined(C64)
#define ARRTYPE complex(real64)
#define DTYPE CUDECOMP_DOUBLE_COMPLEX
#define MODNAME transpose_CUDECOMP_DOUBLE_COMPLEX_mod
#endif

#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) call exit(1)
#define CHECK_CUDA_EXIT(f) if (f /= cudaSuccess) call exit(1)

module MODNAME
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
    mismatch = any(ref(pinfo%lo(1): pinfo%hi(1), pinfo%lo(2): pinfo%hi(2), pinfo%lo(3): pinfo%hi(3)) /= &
                   res(pinfo%lo(1): pinfo%hi(1), pinfo%lo(2): pinfo%hi(2), pinfo%lo(3): pinfo%hi(3)))
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

  subroutine flat_copy(src, dst, count)
    implicit none
    ARRTYPE :: src(*)
    ARRTYPE, device :: dst(*)
    integer(8) :: count

    dst(1:count) = src(1:count)

  end subroutine flat_copy
end module MODNAME

program main
  use cudafor
  use mpi
  use cudecomp
  use, intrinsic :: iso_fortran_env, only: real32, real64

  use MODNAME

  implicit none

  ! Command line arguments
  integer :: gx, gy, gz
  integer :: comm_backend
  logical :: axis_contiguous(3)
  integer :: gdims_dist(3)
  integer :: halo_extents(3)
  logical :: out_of_place, use_managed_memory
  integer :: pr, pc

  ! MPI
  integer :: rank, local_rank, nranks, ierr
  integer :: local_comm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options
  type(cudecompGridDesc) :: grid_desc
  type(cudecompPencilInfo) :: pinfo_x, pinfo_y, pinfo_z

  integer :: pdims(2)
  integer :: gdims(3)
  integer(8) :: data_num_elements, workspace_num_elements

  ! data
  ARRTYPE, allocatable :: xref(:, :, :), yref(:, :, :), zref(:, :, :), data(:)
  ARRTYPE, allocatable, device, target:: data_d(:), data_2_d(:)
  ARRTYPE, allocatable, managed, target:: data_m(:), data_2_m(:)
  ARRTYPE, pointer, device, contiguous :: work_d(:)
  ARRTYPE, pointer, device:: input(:), output(:)
  integer :: dtype = DTYPE

  integer :: i, j, k, idt
  logical :: skip_next
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
  halo_extents(:) = 0
  out_of_place = .false.
  use_managed_memory = .false.

  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--gx')
        call get_command_argument(i+1, arg)
        read(arg, *) gx
        skip_next = .true.
      case('--gy')
        call get_command_argument(i+1, arg)
        read(arg, *) gy
        skip_next = .true.
      case('--gz')
        call get_command_argument(i+1, arg)
        read(arg, *) gz
        skip_next = .true.
      case('--backend')
        call get_command_argument(i+1, arg)
        read(arg, *) comm_backend
        skip_next = .true.
      case('--pr')
        call get_command_argument(i+1, arg)
        read(arg, *) pr
        skip_next = .true.
      case('--pc')
        call get_command_argument(i+1, arg)
        read(arg, *) pc
        skip_next = .true.
      case('--acx')
        call get_command_argument(i+1, arg)
        read(arg, *) axis_contiguous(1)
        skip_next = .true.
      case('--acy')
        call get_command_argument(i+1, arg)
        read(arg, *) axis_contiguous(2)
        skip_next = .true.
      case('--acz')
        call get_command_argument(i+1, arg)
        read(arg, *) axis_contiguous(3)
        skip_next = .true.
      case('--gdx')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(1)
        skip_next = .true.
      case('--gdy')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(2)
        skip_next = .true.
      case('--gdz')
        call get_command_argument(i+1, arg)
        read(arg, *) gdims_dist(3)
        skip_next = .true.
      case('--hex')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(1)
        skip_next = .true.
      case('--hey')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(2)
        skip_next = .true.
      case('--hez')
        call get_command_argument(i+1, arg)
        read(arg, *) halo_extents(3)
        skip_next = .true.
      case('-o')
        out_of_place = .true.
      case('-m')
        use_managed_memory = .true.
      case(' ')
        skip_next = .true.
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
     write(*,"('running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid ...')") gx, gy, gz
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

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
  options%dtype = dtype
  options%transpose_use_inplace_buffers = .not. out_of_place

  if (comm_backend /= 0) then
    config%transpose_comm_backend = comm_backend
  else
    options%autotune_transpose_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

  if (rank == 0) then
     write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
     write(*,"('running using ', a, ' transpose backend ...')") &
               cudecompTransposeCommBackendToString(config%transpose_comm_backend)
  endif

  ! Get x-pencil information
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, pinfo_x, 1, halo_extents))

  ! Get y-pencil information
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, pinfo_y, 2, halo_extents))

  ! Get z-pencil information
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, pinfo_z, 3, halo_extents))

  ! Get workspace size
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, workspace_num_elements))

  ! Allocate data arrays
  data_num_elements = max(pinfo_x%size, pinfo_y%size, pinfo_z%size)
  allocate(data(data_num_elements))

  ! Create reference data
  call initialize_pencil(xref, pinfo_x, gdims)
  call initialize_pencil(yref, pinfo_y, gdims)
  call initialize_pencil(zref, pinfo_z, gdims)

  if (use_managed_memory) then
    allocate(data_m(data_num_elements))
  else
    allocate(data_d(data_num_elements))
  endif
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements))
  if (out_of_place) then
    if (use_managed_memory) then
      allocate(data_2_m(data_num_elements))
    else
      allocate(data_2_d(data_num_elements))
    endif
  endif

  ! Running correctness tests
  if (rank == 0) write(*,"('running correctness tests...')")

  ! Initialize data to reference x-pencil data
  if (use_managed_memory) then
    call flat_copy(xref, data_m, pinfo_x%size)
  else
    call flat_copy(xref, data_d, pinfo_x%size)
  endif

  if (use_managed_memory) then
    input => data_m
    output => data_m
    if (out_of_place) output => data_2_m
  else
    input => data_d
    output => data_d
    if (out_of_place) output => data_2_d
  endif

  work_d = 0
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, input, output, work_d, dtype, pinfo_x%halo_extents, pinfo_y%halo_extents))
  data = output
  if (compare_pencils(yref, data, pinfo_y)) then
    print*, "FAILED cudecompTranposeXToY"
    call exit(1)
  endif

  if (out_of_place) then
    if (use_managed_memory) then
      output => data_m
      input => data_2_m
    else
      output => data_d
      input => data_2_d
    endif
  endif

  work_d = 0
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, input, output, work_d, dtype, pinfo_y%halo_extents, pinfo_z%halo_extents))
  data = output
  if (compare_pencils(zref, data, pinfo_z)) then
    print*, "FAILED cudecompTranposeYToZ"
    call exit(1)
  endif

  if (out_of_place) then
    if (use_managed_memory) then
      output => data_2_m
      input => data_m
    else
      output => data_2_d
      input => data_d
    endif
  endif

  work_d = 0
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, input, output, work_d, dtype, pinfo_z%halo_extents, pinfo_y%halo_extents))
  data = output
  if (compare_pencils(yref, data, pinfo_y)) then
    print*, "FAILED cudecompTranposeZToY"
    call exit(1)
  endif

  if (out_of_place) then
    if (use_managed_memory) then
      output => data_m
      input => data_2_m
    else
      output => data_d
      input => data_2_d
    endif
  endif

  work_d = 0
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, input, output, work_d, dtype, pinfo_y%halo_extents, pinfo_x%halo_extents))
  data = output
  if (compare_pencils(xref, data, pinfo_x)) then
    print*, "FAILED cudecompTranposeXToY"
    call exit(1)
  endif

  if (use_managed_memory) then
    deallocate(data_m)
    if (out_of_place) deallocate(data_2_m)
  else
    deallocate(data_d)
    if (out_of_place) deallocate(data_2_d)
  endif
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))

  call MPI_Finalize(ierr)

end program main
