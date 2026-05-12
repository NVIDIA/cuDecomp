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

! Simple subroutine to check HIP errors
subroutine CHECK_HIP_EXIT(istat)
  use hipfort
  integer :: istat
  if (istat /= hipSuccess) then
    print*, "HIP Error. Exiting."
    call exit(1)
  endif
end subroutine CHECK_HIP_EXIT

! Simple subroutine to check hipDecomp errors
subroutine CHECK_HIPDECOMP_EXIT(istat)
  use hipdecomp
  integer :: istat
  if (istat /= HIPDECOMP_RESULT_SUCCESS) then
    print*, "HIPDECOMP Error. Exiting."
    call exit(1)
  endif
end subroutine CHECK_HIPDECOMP_EXIT

program main
  use hipfort
  use mpi
  use hipdecomp
  use, intrinsic :: iso_fortran_env, only: real64

  implicit none

  ! MPI
  integer :: rank, local_rank, nranks, ierr
  integer :: local_comm, device_count

  ! hipdecomp
  type(hipdecompHandle) :: handle
  type(hipdecompGridDescConfig) :: config
  type(hipdecompGridDescAutotuneOptions) :: options
  type(hipdecompGridDesc) :: grid_desc
  type(hipdecompPencilInfo) :: pinfo_x, pinfo_y, pinfo_z
  integer :: istat
  logical :: disable_nccl_backends

  ! data
  real(real64), allocatable, target :: data(:)
  real(real64), pointer :: data_d(:)
  real(real64), pointer, contiguous :: transpose_work_d(:), halo_work_d(:)
  real(real64), pointer, contiguous :: data_x(:,:,:), data_y(:,:,:), data_z(:,:,:)
  real(real64), pointer :: data_x_d(:,:,:), data_y_d(:,:,:), data_z_d(:,:,:)
  integer(8) :: data_num_elements, transpose_work_num_elements, halo_work_num_elements
  logical :: halo_periods(3)

  integer :: i, j, k
  integer :: gx(3)

  ! Initialize MPI and start up hipDecomp
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  call MPI_Comm_split_Type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
  call MPI_Comm_rank(local_comm, local_rank, ierr)

  ierr = hipGetDeviceCount(device_count)
  ierr = hipSetDevice(mod(local_rank, device_count))

  ! Cannot use NCCL if multiple ranks run on the same GPU
  disable_nccl_backends = .false.
  if (local_rank >= device_count) disable_nccl_backends = .true.
  call MPI_Allreduce(MPI_IN_PLACE, disable_nccl_backends, 1, MPI_LOGICAL, MPI_LOR, MPI_COMM_WORLD, ierr)

  istat = hipdecompInit(handle, MPI_COMM_WORLD)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Create hipDecomp grid descriptor
  istat = hipdecompGridDescConfigSetDefaults(config)
  call CHECK_HIPDECOMP_EXIT(istat)
  ! Set pdims entries to 0 to enable process grid autotuning
  config%pdims = [0, 0] ! [P_rows, P_cols]
  config%gdims = [64, 64, 64] ! [X, Y, Z]

  config%transpose_comm_backend = HIPDECOMP_TRANSPOSE_COMM_MPI_P2P
  config%halo_comm_backend = HIPDECOMP_HALO_COMM_MPI

  config%transpose_axis_contiguous = [.true., .true., .true.]

  ! Set up autotune options structure
  istat = hipdecompGridDescAutotuneOptionsSetDefaults(options)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! General options
  options%n_warmup_trials = 3
  options%n_trials = 5
  options%dtype = HIPDECOMP_DOUBLE
  options%disable_nccl_backends = disable_nccl_backends
  options%disable_nvshmem_backends = .false.
  options%skip_threshold = 0.0

  ! Process grid autotuning options
  options%grid_mode = HIPDECOMP_AUTOTUNE_GRID_TRANSPOSE
  options%allow_uneven_decompositions = .true.

  ! Transpose communication backend autotuning options
  options%autotune_transpose_backend = .true.
  options%transpose_use_inplace_buffers(1) = .true. ! use in-place buffers for X-to-Y transpose
  options%transpose_use_inplace_buffers(2) = .true. ! use in-place buffers for Y-to-Z transpose
  options%transpose_use_inplace_buffers(3) = .true. ! use in-place buffers for Z-to-Y transpose
  options%transpose_use_inplace_buffers(4) = .true. ! use in-place buffers for Y-to-X transpose
  options%transpose_op_weights(1) = 1.0 ! apply 1.0 multiplier to X-to-Y transpose timings
  options%transpose_op_weights(2) = 1.0 ! apply 1.0 multiplier to Y-to-Z transpose timings
  options%transpose_op_weights(3) = 1.0 ! apply 1.0 multiplier to Z-to-Y transpose timings
  options%transpose_op_weights(4) = 1.0 ! apply 1.0 multiplier to Y-to-X transpose timings
  options%transpose_input_halo_extents(:, 1) = [1, 1, 1] ! set input_halo_extent to [1, 1, 1] for X-to-Y transpose
  options%transpose_output_halo_extents(:, 4) = [1, 1, 1] ! set output_halo_extent to [1, 1, 1] for Y-to-X transpose

  ! Halo communication backend autotuning options
  options%autotune_halo_backend = .true.

  options%halo_axis = 1

  options%halo_extents = [1, 1, 1]

  options%halo_periods = [.true., .true., .true.]

  istat = hipdecompGridDescCreate(handle, grid_desc, config, options)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Print information on configuration (updated by autotuner)
  if (rank == 0) then
    write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
    write(*,"('running using ', a, ' transpose backend ...')") &
              hipdecompTransposeCommBackendToString(config%transpose_comm_backend)
    write(*,"('running using ', a, ' halo backend ...')") &
              hipdecompHaloCommBackendToString(config%halo_comm_backend)
  endif

  ! Allocating pencil memory

  ! Get X-pencil information (with halo elements)
  istat = hipdecompGetPencilInfo(handle, grid_desc, pinfo_x, 1, [1, 1, 1])
  call CHECK_HIPDECOMP_EXIT(istat)


  ! Get Y-pencil information
  istat = hipdecompGetPencilInfo(handle, grid_desc, pinfo_y, 2)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Get Z-pencil information
  istat = hipdecompGetPencilInfo(handle, grid_desc, pinfo_z, 3)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Allocate pencil memory
  data_num_elements = max(pinfo_x%size, pinfo_y%size, pinfo_z%size)

  ! Allocate device buffer
  call CHECK_HIP_EXIT(hipMalloc(data_d, data_num_elements))

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
  call CHECK_HIP_EXIT(hipMemcpy(data_d, data, size(data), hipMemcpyHostToDevice))

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

  ! Allocating hipDecomp workspace

  ! Get workspace sizes
  istat = hipdecompGetTransposeWorkspaceSize(handle, grid_desc, transpose_work_num_elements)
  call CHECK_HIPDECOMP_EXIT(istat)
  istat = hipdecompGetHaloWorkspaceSize(handle, grid_desc, 1, [1,1,1], halo_work_num_elements)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Allocate using hipdecompMalloc
  ! Note: *_work_d arrays are of type consistent with hipdecompDataType to be used (HIPDECOMP_DOUBLE). Otherwise,
  ! must adjust workspace_num_elements to allocate enough workspace.
  istat = hipdecompMalloc(handle, grid_desc, transpose_work_d, transpose_work_num_elements)
  call CHECK_HIPDECOMP_EXIT(istat)
  istat = hipdecompMalloc(handle, grid_desc, halo_work_d, halo_work_num_elements)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Transposing data

  ! Transpose from X-pencils to Y-pencils.
  istat = hipdecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE, pinfo_x%halo_extents, [0,0,0])
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Transpose from Y-pencils to Z-pencils.
  istat = hipdecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Transpose from Z-pencils to Y-pencils.
  istat = hipdecompTransposeZToY(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Transpose from Y-pencils to X-pencils.
  istat = hipdecompTransposeYToX(handle, grid_desc, data_d, data_d, transpose_work_d, HIPDECOMP_DOUBLE, [0,0,0], pinfo_x%halo_extents)
  call CHECK_HIPDECOMP_EXIT(istat)


  ! Updating halos

  ! Setting for periodic halos in all directions
  halo_periods = [.true., .true., .true.]

  ! Update X-pencil halos in X direction
  istat = hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 1)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Update X-pencil halos in Y direction
  istat = hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 2)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Update X-pencil halos in Z direction
  istat = hipdecompUpdateHalosX(handle, grid_desc, data_d, halo_work_d, HIPDECOMP_DOUBLE, pinfo_x%halo_extents, halo_periods, 3)
  call CHECK_HIPDECOMP_EXIT(istat)

  ! Cleanup resources
  deallocate(data)
  call CHECK_HIP_EXIT(hipFree(data_d))
  istat = hipdecompFree(handle, grid_desc, transpose_work_d)
  call CHECK_HIPDECOMP_EXIT(istat)
  istat = hipdecompFree(handle, grid_desc, halo_work_d)
  call CHECK_HIPDECOMP_EXIT(istat)
  istat = hipdecompGridDescDestroy(handle, grid_desc)
  call CHECK_HIPDECOMP_EXIT(istat)
  istat = hipdecompFinalize(handle)
  call CHECK_HIPDECOMP_EXIT(istat)

  call MPI_Finalize(ierr)

end program
