! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

subroutine RUN_HALO_CASE()
  implicit none

  ! Pass explicit padding while using a non-default decomposition layout. Each
  ! scenario below runs X, Y, and Z halo updates for all update dimensions.
  call RUN_HALO_SCENARIO("custom_layout_padded_halo", k_halo_extents, k_explicit_padding, k_halo_periods, &
                         k_mixed_mem_order, k_default_axis_contiguous, CUDECOMP_RANK_ORDER_COL_MAJOR, .true.)
  ! Omit the Fortran-only optional padding argument and verify the default
  ! unpadded pencil shape still updates halos correctly.
  call RUN_HALO_SCENARIO("default_padding", k_halo_extents, k_zero_extents, k_halo_periods, k_default_mem_order, &
                         k_default_axis_contiguous, CUDECOMP_RANK_ORDER_DEFAULT, .false.)
end subroutine RUN_HALO_CASE

subroutine RUN_HALO_SCENARIO(scenario, halo_extents, padding, halo_periods, mem_order, axis_contiguous, rank_order, &
                             use_optional_padding)
  implicit none

  character(len=*), intent(in) :: scenario
  integer, intent(in) :: halo_extents(3)
  integer, intent(in) :: padding(3)
  logical, intent(in) :: halo_periods(3)
  integer, intent(in) :: mem_order(3, 3)
  logical, intent(in) :: axis_contiguous(3)
  integer, intent(in) :: rank_order
  logical, intent(in) :: use_optional_padding
  integer :: axis

  do axis = 1, 3
    call RUN_HALO_AXIS(scenario, axis, halo_extents, padding, halo_periods, mem_order, axis_contiguous, rank_order, &
                       use_optional_padding)
  enddo
end subroutine RUN_HALO_SCENARIO

subroutine RUN_HALO_AXIS(scenario, axis, halo_extents, padding, halo_periods, mem_order, axis_contiguous, rank_order, &
                         use_optional_padding)
  implicit none

  character(len=*), intent(in) :: scenario
  integer, intent(in) :: axis
  integer, intent(in) :: halo_extents(3)
  integer, intent(in) :: padding(3)
  logical, intent(in) :: halo_periods(3)
  integer, intent(in) :: mem_order(3, 3)
  logical, intent(in) :: axis_contiguous(3)
  integer, intent(in) :: rank_order
  logical, intent(in) :: use_optional_padding
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDesc) :: grid_desc
  type(cudecompPencilInfo) :: pinfo
  integer(int64) :: workspace_num_elements
  integer :: dim
  integer :: res
  logical :: grid_desc_created
  logical :: work_allocated
  ARRTYPE, allocatable :: initial(:)
  ARRTYPE, allocatable :: expected(:)
  ARRTYPE, allocatable :: actual(:)
  ARRTYPE, allocatable, device, target :: data_d(:)
  ARRTYPE, pointer, device, contiguous :: work_d(:)

  grid_desc_created = .false.
  work_allocated = .false.
  nullify(work_d)

  call setup_halo_config(config, axis_contiguous, mem_order, rank_order)
  res = cudecompGridDescCreate(handle, grid_desc, config)
  call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" cudecompGridDescCreate")
  if (res /= CUDECOMP_RESULT_SUCCESS) return
  grid_desc_created = .true.

  if (use_optional_padding) then
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo, axis, halo_extents, padding), &
                        DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" pencil info")
  else
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo, axis, halo_extents), &
                        DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" pencil info")
  endif
  call expect_success(cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents, workspace_num_elements), &
                      DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" halo workspace size")

  allocate(data_d(pinfo%size))
  call INITIALIZE_HALO_PENCIL(initial, pinfo)
  call INITIALIZE_HALO_REFERENCE(expected, pinfo, halo_periods)

  res = cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements)
  call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" cudecompMalloc workspace")
  if (res == CUDECOMP_RESULT_SUCCESS) work_allocated = .true.

  if (work_allocated) then
    data_d = UNSET_HALO_VALUE()
    data_d(1:pinfo%size) = initial
    work_d = 0

    do dim = 1, 3
      select case (axis)
      case (1)
        if (use_optional_padding) then
          res = cudecompUpdateHalosX(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, &
                                     dim, pinfo%padding)
        else
          res = cudecompUpdateHalosX(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, dim)
        endif
      case (2)
        if (use_optional_padding) then
          res = cudecompUpdateHalosY(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, &
                                     dim, pinfo%padding)
        else
          res = cudecompUpdateHalosY(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, dim)
        endif
      case (3)
        if (use_optional_padding) then
          res = cudecompUpdateHalosZ(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, &
                                     dim, pinfo%padding)
        else
          res = cudecompUpdateHalosZ(handle, grid_desc, data_d, work_d, DTYPE, pinfo%halo_extents, halo_periods, dim)
        endif
      end select
      call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" cudecompUpdateHalos")
    enddo

    allocate(actual(pinfo%size))
    actual = data_d(1:pinfo%size)
    call EXPECT_HALO_PENCIL_MATCH(expected, actual, pinfo, DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis))
    deallocate(actual)
  endif

  if (work_allocated) call expect_success(cudecompFree(handle, grid_desc, work_d), &
                                          DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)//" cudecompFree")
  if (grid_desc_created) call expect_success(cudecompGridDescDestroy(handle, grid_desc), &
                                             DTYPE_NAME//" "//trim(scenario)//" axis "//axis_name(axis)// &
                                             " cudecompGridDescDestroy")
end subroutine RUN_HALO_AXIS

subroutine INITIALIZE_HALO_PENCIL(values, pinfo)
  implicit none

  ARRTYPE, allocatable, intent(out) :: values(:)
  type(cudecompPencilInfo), intent(in) :: pinfo
  integer(int64) :: i
  integer(int64) :: local(3)
  integer(int64) :: global(3)

  allocate(values(pinfo%size))
  values = UNSET_HALO_VALUE()

  do i = 1_int64, pinfo%size
    call local_coordinate(i, pinfo, local)
    if (.not. is_internal_coordinate(pinfo, local)) cycle

    call global_coordinate(pinfo, local, global)
    values(i) = HALO_PENCIL_VALUE(global_linear_index(global))
  enddo
end subroutine INITIALIZE_HALO_PENCIL

subroutine INITIALIZE_HALO_REFERENCE(values, pinfo, halo_periods)
  implicit none

  ARRTYPE, allocatable, intent(out) :: values(:)
  type(cudecompPencilInfo), intent(in) :: pinfo
  logical, intent(in) :: halo_periods(3)
  integer(int64) :: i
  integer(int64) :: local(3)
  integer(int64) :: global(3)
  integer :: dim
  logical :: unset

  allocate(values(pinfo%size))
  values = UNSET_HALO_VALUE()

  do i = 1_int64, pinfo%size
    call local_coordinate(i, pinfo, local)
    call global_coordinate(pinfo, local, global)
    unset = is_padding_coordinate(pinfo, local)

    do dim = 1, 3
      if (global(dim) >= 1_int64 .and. global(dim) <= int(k_gdims(dim), int64)) cycle

      if (halo_periods(dim)) then
        global(dim) = wrap_index(global(dim), int(k_gdims(dim), int64))
      else
        unset = .true.
      endif
    enddo

    if (.not. unset) values(i) = HALO_PENCIL_VALUE(global_linear_index(global))
  enddo
end subroutine INITIALIZE_HALO_REFERENCE

function HALO_PENCIL_VALUE(linear_index) result(value)
  implicit none

  integer(int64), intent(in) :: linear_index
  ARRTYPE :: value

  value = linear_index
end function HALO_PENCIL_VALUE

function UNSET_HALO_VALUE() result(value)
  implicit none

  ARRTYPE :: value

  value = -1
end function UNSET_HALO_VALUE

subroutine EXPECT_HALO_PENCIL_MATCH(expected, actual, pinfo, context)
  implicit none

  ARRTYPE, intent(in) :: expected(:)
  ARRTYPE, intent(in) :: actual(:)
  type(cudecompPencilInfo), intent(in) :: pinfo
  character(len=*), intent(in) :: context
  integer(int64) :: i
  integer(int64) :: local(3)

  if (size(expected, kind=int64) /= pinfo%size .or. size(actual, kind=int64) /= pinfo%size) then
    call record_failure(trim(context)//" result size mismatch")
    return
  endif

  do i = 1_int64, pinfo%size
    if (expected(i) == actual(i)) cycle

    call local_coordinate(i, pinfo, local)
    write(*, *) "rank", rank, ": FAIL", trim(context), "mismatch at local coordinate", local, &
                "expected", expected(i), "actual", actual(i)
    failures = failures + 1
    return
  enddo
end subroutine EXPECT_HALO_PENCIL_MATCH
