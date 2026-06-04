! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

subroutine RUN_TRANSPOSE_CASE()
  implicit none

  ! Exercise non-default layout, out-of-place transposes, and explicit
  ! halo/padding optional arguments for every supported dtype.
  call RUN_TRANSPOSE_SCENARIO("custom_layout_padded_halo", k_explicit_halo_extents, k_explicit_halo_extents, &
                              k_explicit_halo_extents, k_explicit_padding_x, k_explicit_padding_y, &
                              k_explicit_padding_z, k_mixed_mem_order, k_default_axis_contiguous, &
                              CUDECOMP_RANK_ORDER_COL_MAJOR, .true., .true.)
  ! Omit the Fortran-only optional halo/padding arguments and use the default
  ! layout path with in-place transposes.
  call RUN_TRANSPOSE_SCENARIO("default_layout_in_place", k_zero_extents, k_zero_extents, k_zero_extents, &
                              k_zero_extents, k_zero_extents, k_zero_extents, k_default_mem_order, &
                              k_default_axis_contiguous, CUDECOMP_RANK_ORDER_DEFAULT, .false., .false.)
#ifdef RUN_TRANSPOSE_AXIS_CONTIGUOUS_CASE
  ! Keep axis-contiguous coverage small because it is a layout sanity check, not
  ! a dtype-specific numerical path.
  call RUN_TRANSPOSE_SCENARIO("axis_contiguous_in_place", k_zero_extents, k_zero_extents, k_zero_extents, &
                              k_zero_extents, k_zero_extents, k_zero_extents, k_default_mem_order, &
                              k_all_axis_contiguous, CUDECOMP_RANK_ORDER_DEFAULT, .false., .false.)
#endif
end subroutine RUN_TRANSPOSE_CASE

subroutine RUN_TRANSPOSE_SCENARIO(scenario, halo_extents_x, halo_extents_y, halo_extents_z, padding_x, padding_y, &
                                  padding_z, mem_order, axis_contiguous, rank_order, out_of_place, &
                                  use_optional_args)
  implicit none

  character(len=*), intent(in) :: scenario
  integer, intent(in) :: halo_extents_x(3)
  integer, intent(in) :: halo_extents_y(3)
  integer, intent(in) :: halo_extents_z(3)
  integer, intent(in) :: padding_x(3)
  integer, intent(in) :: padding_y(3)
  integer, intent(in) :: padding_z(3)
  integer, intent(in) :: mem_order(3, 3)
  logical, intent(in) :: axis_contiguous(3)
  integer, intent(in) :: rank_order
  logical, intent(in) :: out_of_place
  logical, intent(in) :: use_optional_args
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDesc) :: grid_desc
  type(cudecompPencilInfo) :: pinfo_x
  type(cudecompPencilInfo) :: pinfo_y
  type(cudecompPencilInfo) :: pinfo_z
  integer(int64) :: data_num_elements
  integer(int64) :: workspace_num_elements
  integer :: res
  logical :: grid_desc_created
  logical :: work_allocated
  ARRTYPE, allocatable :: xref(:)
  ARRTYPE, allocatable :: yref(:)
  ARRTYPE, allocatable :: zref(:)
  ARRTYPE, allocatable :: actual(:)
  ARRTYPE, allocatable, device, target :: data_a_d(:)
  ARRTYPE, allocatable, device, target :: data_b_d(:)
  ARRTYPE, pointer, device, contiguous :: input_d(:)
  ARRTYPE, pointer, device, contiguous :: output_d(:)
  ARRTYPE, pointer, device, contiguous :: work_d(:)

  grid_desc_created = .false.
  work_allocated = .false.
  nullify(input_d)
  nullify(output_d)
  nullify(work_d)

  call setup_transpose_config(config, axis_contiguous, mem_order, rank_order)
  res = cudecompGridDescCreate(handle, grid_desc, config)
  call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompGridDescCreate")
  if (res /= CUDECOMP_RESULT_SUCCESS) return
  grid_desc_created = .true.

  if (use_optional_args) then
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_x, 1, halo_extents_x, padding_x), &
                        DTYPE_NAME//" "//trim(scenario)//" X pencil info")
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_y, 2, halo_extents_y, padding_y), &
                        DTYPE_NAME//" "//trim(scenario)//" Y pencil info")
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_z, 3, halo_extents_z, padding_z), &
                        DTYPE_NAME//" "//trim(scenario)//" Z pencil info")
  else
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_x, 1), &
                        DTYPE_NAME//" "//trim(scenario)//" X pencil info")
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_y, 2), &
                        DTYPE_NAME//" "//trim(scenario)//" Y pencil info")
    call expect_success(cudecompGetPencilInfo(handle, grid_desc, pinfo_z, 3), &
                        DTYPE_NAME//" "//trim(scenario)//" Z pencil info")
  endif
  call expect_success(cudecompGetTransposeWorkspaceSize(handle, grid_desc, workspace_num_elements), &
                      DTYPE_NAME//" "//trim(scenario)//" transpose workspace size")

  data_num_elements = max(pinfo_x%size, max(pinfo_y%size, pinfo_z%size))
  allocate(data_a_d(data_num_elements))
  if (out_of_place) allocate(data_b_d(data_num_elements))

  call INITIALIZE_TRANSPOSE_PENCIL(xref, pinfo_x)
  call INITIALIZE_TRANSPOSE_PENCIL(yref, pinfo_y)
  call INITIALIZE_TRANSPOSE_PENCIL(zref, pinfo_z)

  res = cudecompMalloc(handle, grid_desc, work_d, workspace_num_elements)
  call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompMalloc workspace")
  if (res == CUDECOMP_RESULT_SUCCESS) work_allocated = .true.

  if (work_allocated) then
    data_a_d = UNSET_TRANSPOSE_VALUE()
    if (out_of_place) data_b_d = UNSET_TRANSPOSE_VALUE()
    data_a_d(1:pinfo_x%size) = xref
    input_d => data_a_d
    if (out_of_place) then
      output_d => data_b_d
    else
      output_d => data_a_d
    endif
    work_d = 0
    if (use_optional_args) then
      res = cudecompTransposeXToY(handle, grid_desc, input_d, output_d, work_d, DTYPE, pinfo_x%halo_extents, &
                                  pinfo_y%halo_extents, pinfo_x%padding, pinfo_y%padding)
    else
      res = cudecompTransposeXToY(handle, grid_desc, input_d, output_d, work_d, DTYPE)
    endif
    call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompTransposeXToY")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      allocate(actual(pinfo_y%size))
      actual = output_d(1:pinfo_y%size)
      call EXPECT_TRANSPOSE_PENCIL_MATCH(yref, actual, pinfo_y, DTYPE_NAME//" "//trim(scenario)//" XToY")
      deallocate(actual)
    endif

    if (out_of_place) then
      input_d => data_b_d
      output_d => data_a_d
    else
      input_d => data_a_d
      output_d => data_a_d
    endif
    work_d = 0
    if (use_optional_args) then
      res = cudecompTransposeYToZ(handle, grid_desc, input_d, output_d, work_d, DTYPE, pinfo_y%halo_extents, &
                                  pinfo_z%halo_extents, pinfo_y%padding, pinfo_z%padding)
    else
      res = cudecompTransposeYToZ(handle, grid_desc, input_d, output_d, work_d, DTYPE)
    endif
    call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompTransposeYToZ")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      allocate(actual(pinfo_z%size))
      actual = output_d(1:pinfo_z%size)
      call EXPECT_TRANSPOSE_PENCIL_MATCH(zref, actual, pinfo_z, DTYPE_NAME//" "//trim(scenario)//" YToZ")
      deallocate(actual)
    endif

    if (out_of_place) then
      input_d => data_a_d
      output_d => data_b_d
    else
      input_d => data_a_d
      output_d => data_a_d
    endif
    work_d = 0
    if (use_optional_args) then
      res = cudecompTransposeZToY(handle, grid_desc, input_d, output_d, work_d, DTYPE, pinfo_z%halo_extents, &
                                  pinfo_y%halo_extents, pinfo_z%padding, pinfo_y%padding)
    else
      res = cudecompTransposeZToY(handle, grid_desc, input_d, output_d, work_d, DTYPE)
    endif
    call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompTransposeZToY")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      allocate(actual(pinfo_y%size))
      actual = output_d(1:pinfo_y%size)
      call EXPECT_TRANSPOSE_PENCIL_MATCH(yref, actual, pinfo_y, DTYPE_NAME//" "//trim(scenario)//" ZToY")
      deallocate(actual)
    endif

    if (out_of_place) then
      input_d => data_b_d
      output_d => data_a_d
    else
      input_d => data_a_d
      output_d => data_a_d
    endif
    work_d = 0
    if (use_optional_args) then
      res = cudecompTransposeYToX(handle, grid_desc, input_d, output_d, work_d, DTYPE, pinfo_y%halo_extents, &
                                  pinfo_x%halo_extents, pinfo_y%padding, pinfo_x%padding)
    else
      res = cudecompTransposeYToX(handle, grid_desc, input_d, output_d, work_d, DTYPE)
    endif
    call expect_success(res, DTYPE_NAME//" "//trim(scenario)//" cudecompTransposeYToX")
    if (res == CUDECOMP_RESULT_SUCCESS) then
      allocate(actual(pinfo_x%size))
      actual = output_d(1:pinfo_x%size)
      call EXPECT_TRANSPOSE_PENCIL_MATCH(xref, actual, pinfo_x, DTYPE_NAME//" "//trim(scenario)//" YToX")
      deallocate(actual)
    endif
  endif

  if (work_allocated) call expect_success(cudecompFree(handle, grid_desc, work_d), &
                                          DTYPE_NAME//" "//trim(scenario)//" cudecompFree")
  if (grid_desc_created) call expect_success(cudecompGridDescDestroy(handle, grid_desc), &
                                             DTYPE_NAME//" "//trim(scenario)//" cudecompGridDescDestroy")
end subroutine RUN_TRANSPOSE_SCENARIO

subroutine INITIALIZE_TRANSPOSE_PENCIL(values, pinfo)
  implicit none

  ARRTYPE, allocatable, intent(out) :: values(:)
  type(cudecompPencilInfo), intent(in) :: pinfo
  integer(int64) :: i
  integer(int64) :: local(3)
  integer(int64) :: global(3)

  allocate(values(pinfo%size))
  values = UNSET_TRANSPOSE_VALUE()

  do i = 1_int64, pinfo%size
    call local_coordinate(i, pinfo, local)
    if (.not. is_internal_coordinate(pinfo, local)) cycle

    call global_coordinate(pinfo, local, global)
    values(i) = TRANSPOSE_PENCIL_VALUE(global_linear_index(global))
  enddo
end subroutine INITIALIZE_TRANSPOSE_PENCIL

function TRANSPOSE_PENCIL_VALUE(linear_index) result(value)
  implicit none

  integer(int64), intent(in) :: linear_index
  ARRTYPE :: value

  value = linear_index
end function TRANSPOSE_PENCIL_VALUE

function UNSET_TRANSPOSE_VALUE() result(value)
  implicit none

  ARRTYPE :: value

  value = -1
end function UNSET_TRANSPOSE_VALUE

subroutine EXPECT_TRANSPOSE_PENCIL_MATCH(expected, actual, pinfo, context)
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
    call local_coordinate(i, pinfo, local)
    if (.not. is_internal_coordinate(pinfo, local)) cycle
    if (expected(i) == actual(i)) cycle

    write(*, *) "rank", rank, ": FAIL", trim(context), "mismatch at local coordinate", local, &
                "expected", expected(i), "actual", actual(i)
    failures = failures + 1
    return
  enddo
end subroutine EXPECT_TRANSPOSE_PENCIL_MATCH
