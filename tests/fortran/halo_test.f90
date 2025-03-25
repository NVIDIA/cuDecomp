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
#define CHECK_CUDECOMP(f) if (f /= CUDECOMP_RESULT_SUCCESS) then; res = 1; return; endif
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
  use cudafor
  use cudecomp
  use mpi

  type(cudecompHandle) :: handle
  integer :: rank, nranks
  type(cudecompGridDesc) :: grid_desc_cache(5)
  logical :: grid_desc_cache_set(5) = .false.
  ARRTYPE, pointer, device, contiguous :: work_d(:)
  integer :: work_backend = -1

  contains
  function compare_pencils(ref, res, pinfo) result(mismatch)
    use cudecomp
    implicit none
    type(cudecompPencilInfo) :: pinfo
    ARRTYPE :: ref(pinfo%shape(1), pinfo%shape(2), pinfo%shape(3))
    ARRTYPE :: res(pinfo%shape(1), pinfo%shape(2), pinfo%shape(3))

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

    ! Allocate reference pencil with halo and padding regions
    allocate(ref(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)) + pinfo%padding(pinfo%order(1)), &
                 pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)) + pinfo%padding(pinfo%order(2)), &
                 pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3)) + pinfo%padding(pinfo%order(3))))

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

    ! Allocate reference pencil with halo and padding regions
    allocate(ref(pinfo%lo(1) - pinfo%halo_extents(pinfo%order(1)): pinfo%hi(1) + pinfo%halo_extents(pinfo%order(1)) + pinfo%padding(pinfo%order(1)), &
                 pinfo%lo(2) - pinfo%halo_extents(pinfo%order(2)): pinfo%hi(2) + pinfo%halo_extents(pinfo%order(2)) + pinfo%padding(pinfo%order(2)), &
                 pinfo%lo(3) - pinfo%halo_extents(pinfo%order(3)): pinfo%hi(3) + pinfo%halo_extents(pinfo%order(3)) + pinfo%padding(pinfo%order(3))))

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

  subroutine read_testfile(filename, testcases)
    implicit none
    integer :: i, stat, ncases
    character(len=*) :: filename
    character(len=256), allocatable:: testcases(:)

    ! Count number of cases in file
    ncases = 0
    open(42, file=trim(filename), status='old')
    do while (1)
      read(42, '(A)', iostat=stat)
      if (stat < 0) exit
      ncases = ncases + 1
    enddo
    close(42)

    allocate(testcases(ncases))
    open(42, file=trim(filename), status='old')
    do i = 1, ncases
      read(42, '(A)') testcases(i)(:)
    enddo
    close(42)

  end subroutine read_testfile

  subroutine cache_grid_desc(grid_desc, backend)
    implicit none
    type(cudecompGridDesc) :: grid_desc
    integer :: backend

    if (grid_desc_cache_set(backend)) then
      CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_cache(backend)))
    endif

    grid_desc_cache(backend) = grid_desc
    grid_desc_cache_set(backend) = .true.
  end subroutine cache_grid_desc

  function run_test(arguments, silent) result(res)
    implicit none

    character(len=*) :: arguments
    logical :: silent
    integer :: res

    ! Command line arguments
    integer :: gx, gy, gz
    integer :: comm_backend
    logical :: axis_contiguous
    integer :: gdims_dist(3)
    integer :: halo_extents(3)
    logical :: halo_periods(3)
    integer :: padding(3)
    integer :: mem_order(3)
    logical :: use_managed_memory
    integer :: pr, pc
    integer :: axis

    ! MPI
    integer :: local_rank, ierr
    integer :: local_comm

    ! cudecomp
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
    ARRTYPE, pointer, device:: input(:)
    integer :: dtype = DTYPE

    integer :: i, j, k, l, idt, iarg
    integer :: skip_count, nspaces
    character(len=16) :: arg
    character(len=16), allocatable :: args(:)

    res = 0

    ! Split arguments string into list
    nspaces = 0
    do i = 1, len_trim(arguments)
      if (arguments(i:i) == ' ') nspaces = nspaces + 1
    enddo
    allocate(args(nspaces + 1))
    read(arguments, *) args(:)

    ! Parse command-line arguments
    gx = 256
    gy = 256
    gz = 256
    pr = 0
    pc = 0
    comm_backend = 0
    axis_contiguous = .false.
    gdims_dist(:) = 0
    halo_extents(:) = 1
    halo_periods(:) = .true.
    padding(:) = 0
    mem_order(:) = -1
    axis = 1
    use_managed_memory = .false.

    skip_count = 0
    do i = 1, size(args)
      if (skip_count > 0) then
        skip_count = skip_count - 1
        cycle
      end if
      read(args(i), *) arg
      select case(arg)
        case('--gx')
          read(args(i+1), *) arg
          read(arg, *) gx
          skip_count = 1
        case('--gy')
          read(args(i+1), *) arg
          read(arg, *) gy
          skip_count = 1
        case('--gz')
          read(args(i+1), *) arg
          read(arg, *) gz
          skip_count = 1
        case('--backend')
          read(args(i+1), *) arg
          read(arg, *) comm_backend
          skip_count = 1
        case('--pr')
          read(args(i+1), *) arg
          read(arg, *) pr
          skip_count = 1
        case('--pc')
          read(args(i+1), *) arg
          read(arg, *) pc
          skip_count = 1
        case('--ac')
          read(args(i+1), *) arg
          read(arg, *) iarg
          axis_contiguous = iarg
          skip_count = 1
        case('--gd')
          read(args(i+1), *) arg
          read(arg, *) gdims_dist(1)
          skip_count = 1
          do j = 1, 3
            read(args(i+j), *) arg
            read(arg, *) gdims_dist(j)
          enddo
          skip_count = 3
        case('--hex')
          read(args(i+1), *) arg
          read(arg, *) halo_extents(1)
          skip_count = 1
        case('--hey')
          read(args(i+1), *) arg
          read(arg, *) halo_extents(2)
          skip_count = 1
        case('--hez')
          read(args(i+1), *) arg
          read(arg, *) halo_extents(3)
          skip_count = 1
        case('--hpx')
          read(args(i+1), *) arg
          read(arg, *) iarg
          halo_periods(1) = iarg
          skip_count = 1
        case('--hpy')
          read(args(i+1), *) arg
          read(arg, *) iarg
          halo_periods(2) = iarg
          skip_count = 1
        case('--hpz')
          read(args(i+1), *) arg
          read(arg, *) iarg
          halo_periods(3) = iarg
          skip_count = 1
        case('--pdx')
          read(args(i+1), *) arg
          read(arg, *) padding(1)
          skip_count = 1
        case('--pdy')
          read(args(i+1), *) arg
          read(arg, *) padding(2)
          skip_count = 1
        case('--pdz')
          read(args(i+1), *) arg
          read(arg, *) padding(3)
          skip_count = 1
        case('--ax')
          read(args(i+1), *) arg
          read(arg, *) axis
          skip_count = 1
        case('--mem_order')
          do j = 1, 3
            read(args(i+j), *) arg
            read(arg, *) mem_order(j)
          enddo
          skip_count = 3
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

    if (.not. silent .and. rank == 0) then
       write(*,"('Running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid ...')") gx, gy, gz
    end if

    ! Setup grid descriptor
    CHECK_CUDECOMP(cudecompGridDescConfigSetDefaults(config))
    config%pdims = pdims
    gdims = [gx, gy, gz]
    config%gdims = gdims
    config%gdims_dist = gdims_dist
    config%transpose_axis_contiguous(:) = axis_contiguous
    config%transpose_mem_order(:, 1) = mem_order
    config%transpose_mem_order(:, 2) = mem_order
    config%transpose_mem_order(:, 3) = mem_order

    CHECK_CUDECOMP(cudecompGridDescAutotuneOptionsSetDefaults(options))
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

    CHECK_CUDECOMP(cudecompGridDescCreate(handle, grid_desc, config, options))
    call cache_grid_desc(grid_desc, config%halo_comm_backend)

    if (.not. silent .and. rank == 0) then
       write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
       write(*,"('running using ', a, ' halo backend ...')") &
                 cudecompHaloCommBackendToString(config%halo_comm_backend)
    endif

    ! Get pencil information
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, pinfo, axis, halo_extents, padding))

    ! Get workspace size
    CHECK_CUDECOMP(cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents, workspace_num_elements))

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

    ! Allocate workspace (reuse exising workspace if able)
    if (work_backend == config%halo_comm_backend) then
      if (size(work_d) < workspace_num_elements) then
        CHECK_CUDECOMP(cudecompFree(handle, grid_desc, work_d))
        CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements))
      endif
    elseif (work_backend > 0 .and. work_backend /= config%halo_comm_backend) then
      CHECK_CUDECOMP(cudecompFree(handle, grid_desc_cache(work_backend), work_d))
      CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements))
      work_backend = config%halo_comm_backend;
    else
      CHECK_CUDECOMP(cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements))
      work_backend = config%halo_comm_backend;
    endif

    ! Running correctness tests
    if (.not. silent .and. rank == 0) write(*,"('Running correctness tests using ', a, ' backend ...')") &
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
          CHECK_CUDECOMP(cudecompUpdateHalosX(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i, padding))
        case(2)
          CHECK_CUDECOMP(cudecompUpdateHalosY(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i, padding))
        case(3)
          CHECK_CUDECOMP(cudecompUpdateHalosZ(handle, grid_desc, input, work_d, dtype, pinfo%halo_extents, halo_periods, i, padding))
      end select
    end do

    data = input
    if (compare_pencils(ref, data, pinfo)) then
      print*, "FAILED cudecompUpdateHalos"
      res = 1
      return
    endif

    if (use_managed_memory) then
      deallocate(data_m)
    else
      deallocate(data_d)
    endif

  end function
end module

program main
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

  integer :: i, idx
  real(real64) :: t0
  integer :: local_rank, ierr
  integer :: local_comm
  integer :: res, retcode
  logical :: using_testfile
  character(len=16) :: arg
  character(len=256) :: binname
  character(len=256) :: testfile
  character(len=256), allocatable:: testcases(:)
  integer :: nfailed
  character(len=256), allocatable:: failed_cases(:)

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  call MPI_Comm_split_Type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
  call MPI_Comm_rank(local_comm, local_rank, ierr)
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank))

  using_testfile = .false.
  do i = 1, command_argument_count()
    call get_command_argument(i, arg)
    if (arg == "--testfile") then
      call get_command_argument(i+1, testfile)
      using_testfile = .true.
      exit
    endif
  enddo

  ! Initialize cuDecomp
  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  if (.not. using_testfile) then
    allocate(testcases(1))
    testcases(1)(:) = ' '
    idx = 1
    do i = 1, command_argument_count()
      call get_command_argument(i, arg)
      testcases(1)(idx:idx+len_trim(arg)) = trim(arg)
      idx = idx + len_trim(arg) + 1
    enddo
  else
    call read_testfile(testfile, testcases)
  endif


  nfailed = 0
  allocate(failed_cases(size(testcases)))

  t0 = MPI_Wtime()
  if (using_testfile .and. rank == 0) write(*,"('Running ', i0, ' tests...')"), size(testcases)
  call get_command_argument(0, binname)
  do i = 1, size(testcases)
    if (using_testfile .and. rank == 0) write(*, "('command: ', A, ' ', A)"), trim(binname), trim(testcases(i))

    res = run_test(testcases(i), using_testfile)
    if (rank == 0) then
      call MPI_Reduce(MPI_IN_PLACE, res, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    else
      call MPI_Reduce(res, res, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    endif
    if (using_testfile .and. rank == 0) then
      if (res /= 0) then
        print*, "FAILED"
        nfailed = nfailed + 1
        failed_cases(nfailed)(:) = testcases(i)(:)
      else
        print*, "PASSED"
      endif
    endif
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (using_testfile .and. mod(i, 10) == 0) then
      if (rank == 0) then
        write(*, "('Completed ', i0, '/', i0, ' tests, running time ', f0.8, 's')"), i, size(testcases), MPI_Wtime() - t0
      endif
    endif
  enddo

  retcode = 0
  if (using_testfile) then
    if (rank == 0) then
      write(*, "('Completed all tests, running time ', f0.8, ' s')"), MPI_Wtime() - t0
      if (nfailed == 0) then
        write(*, "(A)"), "Passed all tests."
      else
        write(*, "('Failed ', i0, '/', i0, ' tests. Failing cases:')"), nfailed, size(testcases)
        do i = 1, nfailed
          write(*, "(A, ' ', A)"), trim(binname), trim(failed_cases(i))
        enddo
      endif
    endif
    if (nfailed /= 0) retcode = 1;
  endif

  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))
  call MPI_Finalize(ierr)

  call exit(retcode)
end program main
