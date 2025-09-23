! SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
! http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! Simple subroutine to check cuDecomp errors
subroutine CHECK_CUDECOMP_EXIT(istat)
  use cudecomp
  integer :: istat
  if (istat /= CUDECOMP_RESULT_SUCCESS) then
    print*, "CUDECOMP Error. Exiting."
    call exit(1)
  endif
end subroutine CHECK_CUDECOMP_EXIT

program main
  use cudafor
  use mpi
  use cudecomp
  use, intrinsic :: iso_fortran_env, only: real64

  implicit none

  ! MPI
  integer :: rank, local_rank, nranks, ierr
  integer :: local_comm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options
  type(cudecompGridDesc) :: grid_desc
  type(cudecompPencilInfo) :: pinfo_x, pinfo_y, pinfo_z
  integer :: istat

  ! data
  real(real64), allocatable, target :: data(:)
  real(real64), allocatable, device, target :: data_d(:)
  real(real64), pointer, device, contiguous :: transpose_work_d(:), halo_work_d(:)
  real(real64), pointer, contiguous :: data_x(:,:,:), data_y(:,:,:), data_z(:,:,:)
  real(real64), pointer, device, contiguous :: data_x_d(:,:,:), data_y_d(:,:,:), data_z_d(:,:,:)
  integer(8) :: data_num_elements, transpose_work_num_elements, halo_work_num_elements
  logical :: halo_periods(3)

  integer :: i, j, k
  integer :: gx(3)

  ! Initialize MPI and start up cuDecomp
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  if (nranks /= 4) then
    print*, "ERROR: This example requires 4 ranks to run. Exiting..."
    call exit(1)
  endif

  call MPI_Comm_split_Type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
  call MPI_Comm_rank(local_comm, local_rank, ierr)
  ierr = cudaSetDevice(local_rank)

  istat = cudecompInit(handle, MPI_COMM_WORLD)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Create cuDecomp grid descriptor
  istat = cudecompGridDescConfigSetDefaults(config)
  call CHECK_CUDECOMP_EXIT(istat)
  config%pdims = [2, 2] ! [P_rows, P_cols]
  config%gdims = [64, 64, 64] ! [X, Y, Z]

  config%transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P
  config%halo_comm_backend = CUDECOMP_HALO_COMM_MPI

  config%transpose_axis_contiguous = [.true., .true., .true.]

  istat = cudecompGridDescCreate(handle, grid_desc, config)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Print information on configuration
  if (rank == 0) then
    write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
    write(*,"('running using ', a, ' transpose backend ...')") &
              cudecompTransposeCommBackendToString(config%transpose_comm_backend)
    write(*,"('running using ', a, ' halo backend ...')") &
              cudecompHaloCommBackendToString(config%halo_comm_backend)
  endif

  ! Allocating pencil memory

  ! Get X-pencil information (with halo elements)
  istat = cudecompGetPencilInfo(handle, grid_desc, pinfo_x, 1, [1, 1, 1])
  call CHECK_CUDECOMP_EXIT(istat)

  ! Get Y-pencil information
  istat = cudecompGetPencilInfo(handle, grid_desc, pinfo_y, 2)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Get Z-pencil information
  istat = cudecompGetPencilInfo(handle, grid_desc, pinfo_z, 3)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Allocate pencil memory
  data_num_elements = max(pinfo_x%size, pinfo_y%size, pinfo_z%size)

  ! Allocate device buffer
  allocate(data_d(data_num_elements))

  ! Allocate host buffer
  allocate(data(data_num_elements))

  ! Initializing pencil data
  ! Host pointers
  data_x(1:pinfo_x%shape(1), 1:pinfo_x%shape(2), 1:pinfo_x%shape(3)) => data(:)
  data_y(1:pinfo_y%shape(1), 1:pinfo_y%shape(2), 1:pinfo_y%shape(3)) => data(:)
  data_z(1:pinfo_z%shape(1), 1:pinfo_z%shape(2), 1:pinfo_z%shape(3)) => data(:)

  ! Device pointers
  data_x_d(1:pinfo_x%shape(1), 1:pinfo_x%shape(2), 1:pinfo_x%shape(3)) => data_d(:)
  data_y_d(1:pinfo_y%shape(1), 1:pinfo_y%shape(2), 1:pinfo_y%shape(3)) => data_d(:)
  data_z_d(1:pinfo_z%shape(1), 1:pinfo_z%shape(2), 1:pinfo_z%shape(3)) => data_d(:)

  ! Initialize on host + H2D memcopy
  do k = 1, pinfo_x%shape(3)
    do j = 1, pinfo_x%shape(2)
      do i = 1, pinfo_x%shape(1)
        ! Compute global grid coordinates. To compute these, we offset the local coordinates
        ! using the lower bound, lo, and use the order array to map the local coordinate order
        ! to the global coordinate order.
        gx(pinfo_x%order(1)) = i + pinfo_x%lo(1) - 1
        gx(pinfo_x%order(2)) = j + pinfo_x%lo(2) - 1
        gx(pinfo_x%order(3)) = k + pinfo_x%lo(3) - 1

        ! Since the X-pencil also has halo elements, we apply an additional offset for the halo
        ! elements in each direction, again using the order array to apply the extent to the
        ! appropriate global coordinate
        gx(pinfo_x%order(1)) =  gx(pinfo_x%order(1)) - pinfo_x%halo_extents(pinfo_x%order(1))
        gx(pinfo_x%order(2)) =  gx(pinfo_x%order(2)) - pinfo_x%halo_extents(pinfo_x%order(2))
        gx(pinfo_x%order(3)) =  gx(pinfo_x%order(3)) - pinfo_x%halo_extents(pinfo_x%order(3))

        ! Finally, we can set the buffer element, for example using a function based on the
        ! global coordinates.
        data_x(i,j,k) = gx(1) + gx(2) + gx(3)

      enddo
    enddo
  enddo

  ! Copy host data to device
  data_d = data

  ! Initialize on device
  !$acc parallel loop collapse(3) private(gx)
  do k = 1, pinfo_x%shape(3)
    do j = 1, pinfo_x%shape(2)
      do i = 1, pinfo_x%shape(1)
        ! Compute global grid coordinates. To compute these, we offset the local coordinates
        ! using the lower bound, lo, and use the order array to map the local coordinate order
        ! to the global coordinate order.
        gx(pinfo_x%order(1)) = i + pinfo_x%lo(1) - 1
        gx(pinfo_x%order(2)) = j + pinfo_x%lo(2) - 1
        gx(pinfo_x%order(3)) = k + pinfo_x%lo(3) - 1

        ! Since the X-pencil also has halo elements, we apply an additional offset for the halo
        ! elements in each direction, again using the order array to apply the extent to the
        ! appropriate global coordinate
        gx(pinfo_x%order(1)) =  gx(pinfo_x%order(1)) - pinfo_x%halo_extents(pinfo_x%order(1))
        gx(pinfo_x%order(2)) =  gx(pinfo_x%order(2)) - pinfo_x%halo_extents(pinfo_x%order(2))
        gx(pinfo_x%order(3)) =  gx(pinfo_x%order(3)) - pinfo_x%halo_extents(pinfo_x%order(3))

        ! Finally, we can set the buffer element, for example using a function based on the
        ! global coordinates.
        data_x_d(i,j,k) = gx(1) + gx(2) + gx(3)

      enddo
    enddo
  enddo

  ! Allocating cuDecomp workspace

  ! Get workspace sizes
  istat = cudecompGetTransposeWorkspaceSize(handle, grid_desc, transpose_work_num_elements)
  call CHECK_CUDECOMP_EXIT(istat)
  istat = cudecompGetHaloWorkspaceSize(handle, grid_desc, 1, [1,1,1], halo_work_num_elements)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Allocate using cudecompMalloc
  ! Note: *_work_d arrays are of type consistent with cudecompDataType to be used (CUDECOMP_DOUBLE). Otherwise,
  ! must adjust workspace_num_elements to allocate enough workspace.
  istat = cudecompMalloc(handle, grid_desc, transpose_work_d, transpose_work_num_elements)
  call CHECK_CUDECOMP_EXIT(istat)
  istat = cudecompMalloc(handle, grid_desc, halo_work_d, halo_work_num_elements)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Transposing data

  ! Transpose from X-pencils to Y-pencils.
  istat = cudecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE, pinfo_x%halo_extents, [0,0,0])
  call CHECK_CUDECOMP_EXIT(istat)

  ! Transpose from Y-pencils to Z-pencils.
  istat = cudecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Transpose from Z-pencils to Y-pencils.
  istat = cudecompTransposeZToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Transpose from Y-pencils to X-pencils.
  istat = cudecompTransposeYToX(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_DOUBLE, [0,0,0], pinfo_x%halo_extents)
  call CHECK_CUDECOMP_EXIT(istat)


  ! Updating halos

  ! Setting for periodic halos in all directions
  halo_periods = [.true., .true., .true.]

  ! Update X-pencil halos in X direction
  istat = cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 1)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Update X-pencil halos in Y direction
  istat = cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 2)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Update X-pencil halos in Z direction
  istat = cudecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, CUDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 3)
  call CHECK_CUDECOMP_EXIT(istat)

  ! Cleanup resources
  deallocate(data)
  deallocate(data_d)
  istat = cudecompFree(handle, grid_desc, transpose_work_d)
  call CHECK_CUDECOMP_EXIT(istat)
  istat = cudecompFree(handle, grid_desc, halo_work_d)
  call CHECK_CUDECOMP_EXIT(istat)
  istat = cudecompGridDescDestroy(handle, grid_desc)
  call CHECK_CUDECOMP_EXIT(istat)
  istat = cudecompFinalize(handle)
  call CHECK_CUDECOMP_EXIT(istat)

  call MPI_Finalize(ierr)

end program
