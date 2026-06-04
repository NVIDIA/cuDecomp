! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program cudecomp_fortran_api_test
  use, intrinsic :: iso_fortran_env, only: real32, real64
  use cudafor
  use cudecomp
  use mpi

  implicit none

  integer, parameter :: api_test_ranks = 4
  integer, parameter :: k_gdims(3) = [9, 10, 11]
  integer, parameter :: k_gdims_dist(3) = [8, 9, 10]
  integer, parameter :: k_pdims(2) = [2, 2]
  integer, parameter :: k_halo_extents(3) = [1, 2, 1]
  integer, parameter :: k_padding(3) = [1, 0, 2]

  integer :: rank = -1
  integer :: nranks = 0
  integer :: ierr = 0
  integer :: local_comm = MPI_COMM_NULL
  integer :: local_rank = 0
  integer :: failures = 0
  integer :: global_failures = 0
  logical :: handle_initialized = .false.
  type(cudecompHandle) :: handle

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  if (nranks /= api_test_ranks) then
    call record_failure("Fortran API test requires exactly 4 MPI ranks")
  else
    call initialize_gpu()
  endif

  if (failures == 0) then
    call expect_success(cudecompInit(handle, MPI_COMM_WORLD), "cudecompInit")
    handle_initialized = failures == 0
  endif

  if (handle_initialized) then
    call test_default_values()
    call test_grid_descriptor_contracts(handle)
    call test_descriptor_queries(handle)
    call test_workspace_and_shifted_rank(handle)
    call test_dtype_sizes_and_strings()
    call test_typed_malloc_free(handle)
    call expect_success(cudecompFinalize(handle), "cudecompFinalize")
  endif

  call MPI_Allreduce(failures, global_failures, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0 .and. global_failures == 0) then
    write(*, '("Fortran API test passed")')
  elseif (rank == 0) then
    write(*, '("Fortran API test failed with ", i0, " rank-local failure(s)")') global_failures
  endif

  if (local_comm /= MPI_COMM_NULL) call MPI_Comm_free(local_comm, ierr)
  call MPI_Finalize(ierr)
  if (global_failures /= 0) call exit(1)

contains

  subroutine initialize_gpu()
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
      call record_failure("Fortran API test requires at least one visible CUDA device")
      return
    endif

    status = cudaSetDevice(mod(local_rank, num_devices))
    if (status /= cudaSuccess) call record_failure("cudaSetDevice failed")
  end subroutine initialize_gpu

  subroutine test_default_values()
    type(cudecompGridDescConfig) :: config
    type(cudecompGridDescAutotuneOptions) :: options

    call expect_success(cudecompGridDescConfigSetDefaults(config), "cudecompGridDescConfigSetDefaults")
    call expect_equal_int(config%rank_order, CUDECOMP_RANK_ORDER_DEFAULT, "default rank order")
    call expect_equal_int(config%transpose_comm_backend, CUDECOMP_TRANSPOSE_COMM_MPI_P2P, &
                          "default transpose backend")
    call expect_equal_int(config%halo_comm_backend, CUDECOMP_HALO_COMM_MPI, "default halo backend")
    call expect_int_array(config%pdims, [0, 0], "default pdims")
    call expect_int_array(config%gdims, [0, 0, 0], "default gdims")
    call expect_int_array(config%gdims_dist, [0, 0, 0], "default gdims_dist")
    call expect_int_array(reshape(config%transpose_mem_order, [9]), [-1, -1, -1, -1, -1, -1, -1, -1, -1], &
                          "default transpose_mem_order")
    if (any(config%transpose_axis_contiguous)) call record_failure("default axis-contiguous flags are not false")

    call expect_success(cudecompGridDescAutotuneOptionsSetDefaults(options), &
                        "cudecompGridDescAutotuneOptionsSetDefaults")
    call expect_equal_int(options%n_warmup_trials, 3, "default warmup trials")
    call expect_equal_int(options%n_trials, 5, "default trials")
    call expect_equal_int(options%grid_mode, CUDECOMP_AUTOTUNE_GRID_TRANSPOSE, "default grid mode")
    call expect_equal_int(options%dtype, CUDECOMP_DOUBLE, "default autotune dtype")
    call expect_equal_int(options%halo_axis, 1, "default Fortran halo axis")
    if (.not. options%allow_uneven_decompositions) call record_failure("default uneven decompositions flag is false")
    if (options%autotune_transpose_backend) call record_failure("default transpose backend autotune flag is true")
    if (options%autotune_halo_backend) call record_failure("default halo backend autotune flag is true")
  end subroutine test_default_values

  subroutine test_grid_descriptor_contracts(handle)
    type(cudecompHandle) :: handle
    type(cudecompGridDescConfig) :: config
    type(cudecompGridDescConfig) :: expected_config
    type(cudecompGridDescConfig) :: queried_config
    type(cudecompGridDescAutotuneOptions) :: options
    type(cudecompGridDesc) :: grid_desc
    integer :: res

    call setup_explicit_config(config)
    expected_config = config
    res = cudecompGridDescCreate(handle, grid_desc, config)
    call expect_success(res, "cudecompGridDescCreate without options")
    call expect_config_equal(config, expected_config, "config restored after create without options")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      call expect_success(cudecompGetGridDescConfig(handle, grid_desc, queried_config), &
                          "cudecompGetGridDescConfig without options")
      call expect_config_equal(queried_config, expected_config, "queried config without options")
      call expect_success(cudecompGridDescDestroy(handle, grid_desc), "destroy descriptor without options")
    endif

    call setup_explicit_config(config)
    expected_config = config
    call expect_success(cudecompGridDescAutotuneOptionsSetDefaults(options), &
                        "set defaults before create with options")
    options%n_warmup_trials = 0
    options%n_trials = 1
    options%dtype = CUDECOMP_FLOAT
    options%disable_nccl_backends = .true.
    options%disable_nvshmem_backends = .true.
    options%halo_axis = 3
    options%halo_extents = k_halo_extents
    options%halo_periods = [.false., .true., .false.]
    options%halo_padding = k_padding

    res = cudecompGridDescCreate(handle, grid_desc, config, options)
    call expect_success(res, "cudecompGridDescCreate with options")
    call expect_config_equal(config, expected_config, "config restored after create with options")
    call expect_equal_int(options%halo_axis, 3, "options halo_axis restored after create")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      call expect_success(cudecompGetGridDescConfig(handle, grid_desc, queried_config), &
                          "cudecompGetGridDescConfig with options")
      call expect_config_equal(queried_config, expected_config, "queried config with options")
      call expect_success(cudecompGridDescDestroy(handle, grid_desc), "destroy descriptor with options")
    endif
  end subroutine test_grid_descriptor_contracts

  subroutine test_descriptor_queries(handle)
    type(cudecompHandle) :: handle
    type(cudecompGridDescConfig) :: config
    type(cudecompGridDesc) :: grid_desc
    type(cudecompPencilInfo) :: pinfo
    integer :: res

    call setup_distributed_config(config)
    res = cudecompGridDescCreate(handle, grid_desc, config)
    call expect_success(res, "create descriptor for pencil queries")
    if (res /= CUDECOMP_RESULT_SUCCESS) return

    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo, 1), &
                        "cudecompGetPencilInfo without optional arrays")
    call expect_axis1_pencil_without_halo(pinfo)

    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo, 2, k_halo_extents, k_padding), &
                        "cudecompGetPencilInfo with halo and padding")
    call expect_int_array(pinfo%order, [1, 2, 3], "Fortran pencil order is one-based")
    call expect_int_array(pinfo%halo_extents, k_halo_extents, "explicit pencil halo extents")
    call expect_int_array(pinfo%padding, k_padding, "explicit pencil padding")
    if (minval(pinfo%lo) < 1) call record_failure("Fortran pencil lower bounds are not one-based")
    if (minval(pinfo%hi) < 1) call record_failure("Fortran pencil upper bounds are not one-based")

    call expect_success(cudecompGridDescDestroy(handle, grid_desc), "destroy descriptor for pencil queries")
  end subroutine test_descriptor_queries

  subroutine test_workspace_and_shifted_rank(handle)
    type(cudecompHandle) :: handle
    type(cudecompGridDescConfig) :: config
    type(cudecompGridDesc) :: grid_desc
    integer :: res
    integer(8) :: workspace_size
    integer :: shifted_rank
    integer, parameter :: expected_shifted_ranks(api_test_ranks) = [2, 3, -1, -1]

    call setup_distributed_config(config)
    res = cudecompGridDescCreate(handle, grid_desc, config)
    call expect_success(res, "create descriptor for workspace queries")
    if (res /= CUDECOMP_RESULT_SUCCESS) return

    workspace_size = -1_8
    call expect_success(cudecompGetTransposeWorkspaceSize(handle, grid_desc, workspace_size), &
                        "cudecompGetTransposeWorkspaceSize")
    if (workspace_size < 0_8) call record_failure("transpose workspace size is negative")

    workspace_size = -1_8
    call expect_success(cudecompGetHaloWorkspaceSize(handle, grid_desc, 2, k_halo_extents, workspace_size), &
                        "cudecompGetHaloWorkspaceSize")
    if (workspace_size < 0_8) call record_failure("halo workspace size is negative")

    shifted_rank = -2
    call expect_success(cudecompGetShiftedRank(handle, grid_desc, 1, 2, 1, .false., shifted_rank), &
                        "cudecompGetShiftedRank")
    call expect_equal_int(shifted_rank, expected_shifted_ranks(rank + 1), "one-based shifted rank query")

    call expect_success(cudecompGridDescDestroy(handle, grid_desc), "destroy descriptor for workspace queries")
  end subroutine test_workspace_and_shifted_rank

  subroutine test_dtype_sizes_and_strings()
    integer(8) :: dtype_size

    call expect_success(cudecompGetDataTypeSize(CUDECOMP_FLOAT, dtype_size), "float dtype size query")
    call expect_equal_integer8(dtype_size, 4_8, "float dtype size")
    call expect_success(cudecompGetDataTypeSize(CUDECOMP_DOUBLE, dtype_size), "double dtype size query")
    call expect_equal_integer8(dtype_size, 8_8, "double dtype size")
    call expect_success(cudecompGetDataTypeSize(CUDECOMP_FLOAT_COMPLEX, dtype_size), "float complex dtype size query")
    call expect_equal_integer8(dtype_size, 8_8, "float complex dtype size")
    call expect_success(cudecompGetDataTypeSize(CUDECOMP_DOUBLE_COMPLEX, dtype_size), &
                        "double complex dtype size query")
    call expect_equal_integer8(dtype_size, 16_8, "double complex dtype size")

    call expect_string(cudecompTransposeCommBackendToString(CUDECOMP_TRANSPOSE_COMM_MPI_P2P), "MPI_P2P", &
                       "transpose backend string")
    call expect_string(cudecompHaloCommBackendToString(CUDECOMP_HALO_COMM_MPI_BLOCKING), "MPI (blocking)", &
                       "halo backend string")
  end subroutine test_dtype_sizes_and_strings

  subroutine test_typed_malloc_free(handle)
    type(cudecompHandle) :: handle
    type(cudecompGridDescConfig) :: config
    type(cudecompGridDesc) :: grid_desc
    real(real32), pointer, device, contiguous :: r4(:)
    real(real64), pointer, device, contiguous :: r8(:)
    complex(real32), pointer, device, contiguous :: c4(:)
    complex(real64), pointer, device, contiguous :: c8(:)
    integer :: res

    nullify(r4)
    nullify(r8)
    nullify(c4)
    nullify(c8)

    call setup_distributed_config(config)
    res = cudecompGridDescCreate(handle, grid_desc, config)
    call expect_success(res, "create descriptor for typed allocation")
    if (res /= CUDECOMP_RESULT_SUCCESS) return

    call expect_success(cudecompMalloc(handle, grid_desc, r4, 4_8), "cudecompMalloc real32")
    if (.not. associated(r4)) call record_failure("real32 device pointer is not associated")
    call expect_success(cudecompFree(handle, grid_desc, r4), "cudecompFree real32")

    call expect_success(cudecompMalloc(handle, grid_desc, r8, 4_8), "cudecompMalloc real64")
    if (.not. associated(r8)) call record_failure("real64 device pointer is not associated")
    call expect_success(cudecompFree(handle, grid_desc, r8), "cudecompFree real64")

    call expect_success(cudecompMalloc(handle, grid_desc, c4, 4_8), "cudecompMalloc complex32")
    if (.not. associated(c4)) call record_failure("complex32 device pointer is not associated")
    call expect_success(cudecompFree(handle, grid_desc, c4), "cudecompFree complex32")

    call expect_success(cudecompMalloc(handle, grid_desc, c8, 4_8), "cudecompMalloc complex64")
    if (.not. associated(c8)) call record_failure("complex64 device pointer is not associated")
    call expect_success(cudecompFree(handle, grid_desc, c8), "cudecompFree complex64")

    call expect_success(cudecompGridDescDestroy(handle, grid_desc), "destroy descriptor for typed allocation")
  end subroutine test_typed_malloc_free

  subroutine setup_distributed_config(config)
    type(cudecompGridDescConfig) :: config

    call expect_success(cudecompGridDescConfigSetDefaults(config), "set distributed config defaults")
    config%gdims = k_gdims
    config%pdims = k_pdims
  end subroutine setup_distributed_config

  subroutine setup_explicit_config(config)
    type(cudecompGridDescConfig) :: config

    call setup_distributed_config(config)
    config%gdims_dist = k_gdims_dist
    config%rank_order = CUDECOMP_RANK_ORDER_COL_MAJOR
    config%transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A
    config%halo_comm_backend = CUDECOMP_HALO_COMM_MPI_BLOCKING
    config%transpose_axis_contiguous = [.true., .false., .true.]
    config%transpose_mem_order(:, 1) = [3, 2, 1]
    config%transpose_mem_order(:, 2) = [1, 3, 2]
    config%transpose_mem_order(:, 3) = [2, 1, 3]
  end subroutine setup_explicit_config

  subroutine expect_axis1_pencil_without_halo(pinfo)
    type(cudecompPencilInfo) :: pinfo
    integer :: expected_shape(3)
    integer :: expected_lo(3)
    integer :: expected_hi(3)
    integer(8) :: expected_size

    select case (rank)
    case (0)
      expected_shape = [9, 5, 6]
      expected_lo = [1, 1, 1]
      expected_hi = [9, 5, 6]
    case (1)
      expected_shape = [9, 5, 5]
      expected_lo = [1, 1, 7]
      expected_hi = [9, 5, 11]
    case (2)
      expected_shape = [9, 5, 6]
      expected_lo = [1, 6, 1]
      expected_hi = [9, 10, 6]
    case default
      expected_shape = [9, 5, 5]
      expected_lo = [1, 6, 7]
      expected_hi = [9, 10, 11]
    end select

    expected_size = product(expected_shape)
    call expect_int_array(pinfo%shape, expected_shape, "axis-1 shape without halo")
    call expect_int_array(pinfo%lo, expected_lo, "axis-1 lo without halo")
    call expect_int_array(pinfo%hi, expected_hi, "axis-1 hi without halo")
    call expect_int_array(pinfo%order, [1, 2, 3], "axis-1 order without halo")
    call expect_int_array(pinfo%halo_extents, [0, 0, 0], "axis-1 omitted halo defaults")
    call expect_int_array(pinfo%padding, [0, 0, 0], "axis-1 omitted padding defaults")
    call expect_equal_integer8(pinfo%size, expected_size, "axis-1 size without halo")
  end subroutine expect_axis1_pencil_without_halo

  subroutine expect_config_equal(actual, expected, context)
    type(cudecompGridDescConfig) :: actual
    type(cudecompGridDescConfig) :: expected
    character(len=*) :: context

    call expect_int_array(actual%gdims, expected%gdims, trim(context)//" gdims")
    call expect_int_array(actual%gdims_dist, expected%gdims_dist, trim(context)//" gdims_dist")
    call expect_int_array(actual%pdims, expected%pdims, trim(context)//" pdims")
    call expect_equal_int(actual%rank_order, expected%rank_order, trim(context)//" rank_order")
    call expect_equal_int(actual%transpose_comm_backend, expected%transpose_comm_backend, &
                          trim(context)//" transpose backend")
    call expect_equal_int(actual%halo_comm_backend, expected%halo_comm_backend, trim(context)//" halo backend")
    if (any(actual%transpose_axis_contiguous .neqv. expected%transpose_axis_contiguous)) then
      call record_failure(trim(context)//" axis-contiguous flags mismatch")
    endif
    call expect_int_array(reshape(actual%transpose_mem_order, [9]), reshape(expected%transpose_mem_order, [9]), &
                          trim(context)//" transpose_mem_order")
  end subroutine expect_config_equal

  subroutine expect_success(result, context)
    integer :: result
    character(len=*) :: context

    if (result /= CUDECOMP_RESULT_SUCCESS) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "returned", result
      failures = failures + 1
    endif
  end subroutine expect_success

  subroutine expect_equal_int(actual, expected, context)
    integer :: actual
    integer :: expected
    character(len=*) :: context

    if (actual /= expected) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "expected", expected, "actual", actual
      failures = failures + 1
    endif
  end subroutine expect_equal_int

  subroutine expect_equal_integer8(actual, expected, context)
    integer(8) :: actual
    integer(8) :: expected
    character(len=*) :: context

    if (actual /= expected) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "expected", expected, "actual", actual
      failures = failures + 1
    endif
  end subroutine expect_equal_integer8

  subroutine expect_int_array(actual, expected, context)
    integer :: actual(:)
    integer :: expected(:)
    character(len=*) :: context

    if (size(actual) /= size(expected)) then
      call record_failure(trim(context)//" array sizes differ")
    elseif (any(actual /= expected)) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "expected", expected, "actual", actual
      failures = failures + 1
    endif
  end subroutine expect_int_array

  subroutine expect_string(actual, expected, context)
    character(len=*) :: actual
    character(len=*) :: expected
    character(len=*) :: context

    if (actual /= expected) then
      write(*, *) "rank", rank, ": FAIL", trim(context), "expected", trim(expected), "actual", trim(actual)
      failures = failures + 1
    endif
  end subroutine expect_string

  subroutine record_failure(context)
    character(len=*) :: context

    write(*, *) "rank", rank, ": FAIL", trim(context)
    failures = failures + 1
  end subroutine record_failure

end program cudecomp_fortran_api_test
