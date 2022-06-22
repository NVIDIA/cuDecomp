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

! Solves poisson equation
!
!   u_xx + u_yy + u_zz= phi(x,y,z)
!
! where
!
!   phi(x,y,z) = sin(2*pi*Mx*x)*sin(2*pi*My*y)*sin(2*pi*Mz*z)
!
! on domain 0 <= x <= 1, 0 <= y <= 1

program main
  use cudafor
  use cudecomp
  use cufft
  use mpi

  implicit none

  ! Command line arguments
  ! grid dimensions
  integer :: nx, ny, nz
  integer :: comm_backend
  integer :: pr, pc

  ! grid size
  real(8), parameter :: Lx = 1.0, Ly = 1.0, Lz = 1.0
  ! modes for phi
  integer, parameter :: Mx = 2, My = 3, Mz = 4
  real(8), parameter :: twopi = 8.0_8*atan(1.0_8)

  ! grid
  real(8), allocatable :: x(:), y(:), z(:)
  real(8) :: hx, hy, hz
  ! wavenumbers
  real(8), allocatable :: kx(:), ky(:), kz(:)
  real(8), device, allocatable :: kx_d(:), ky_d(:), kz_d(:)

  complex(8), allocatable :: phi(:), ua(:,:,:)
  complex(8), device, allocatable :: phi_d(:)
  complex(8), pointer, device, contiguous :: work_d(:)

  ! MPI
  integer :: rank, ranks, ierr
  integer :: localRank, localComm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDesc) :: grid_desc
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options

  integer :: pdims(2)
  integer :: gdims(3)
  integer :: npx, npy, npz
  type(cudecompPencilInfo) :: piX, piY, piZ
  integer(8) :: nElemX, nElemY, nElemZ, nElemWork

  ! CUFFT
  integer :: planX, planY, planZ
  integer :: batchsize
  integer :: status

  logical :: skip_next
  character(len=16) :: arg
  integer :: i, j, k

  ! MPI initialization

  call mpi_init(ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_init failed: ', ierr
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank failed: ', ierr
  call mpi_comm_size(MPI_COMM_WORLD, ranks, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_size failed: ', ierr

  call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, localComm, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_split_type failed: ', ierr
  call mpi_comm_rank(localComm, localRank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank on local rank failed: ', ierr
  ierr = cudaSetDevice(localRank)

  ! Parse command-line arguments
  nx = 64
  ny = 64
  nz = 64
  pr = 0
  pc = 0
  comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P

  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--nx')
        call get_command_argument(i+1, arg)
        read(arg, *) nx
        skip_next = .true.
      case('--ny')
        call get_command_argument(i+1, arg)
        read(arg, *) ny
        skip_next = .true.
      case('--nz')
        call get_command_argument(i+1, arg)
        read(arg, *) nz
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
      case default
        print*, "Unknown argument."
        call exit(1)
    end select
  end do

  ! check for valid Mx/My/Mz
  if (Mx==0 .or. Mx==0 .or. My==0) then
     if (rank == 0) write(*,*) 'Mx/My/Mz is 0 thus solution is u=0'
     call MPI_Finalize(ierr)
     stop
  else if (rank == 0) then
     write(*,"('Running with Mx, My, Mz = ', i0, ', ', i0, ', ', i0)") Mx, My, Mz
     write(*,"('Running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid ...')") &
          nx, ny, nz
  end if

  ! cudecomp initialization

  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config))
  pdims = [pr, pc]
  config%pdims = pdims
  gdims = [nx, ny, nz]
  config%gdims = gdims
  config%transpose_comm_backend = comm_backend
  config%transpose_axis_contiguous = .true.

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
  options%dtype = CUDECOMP_DOUBLE_COMPLEX
  if (comm_backend == 0) then
    options%autotune_transpose_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

  if (rank == 0) then
     write(*,"('Running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
     write(*,"('Using ', a, ' backend ...')") cudecompTransposeCommBackendToString(config%transpose_comm_backend)
  end if

  ! get pencil info
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piX, 1))
  nElemX = piX%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piY, 2))
  nElemY = piY%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piZ, 3))
  nElemZ = piZ%size

  ! get workspace size
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, nElemWork))

  ! CUFFT initialization

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planX, nx, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan'

  batchSize = piY%shape(2)*piY%shape(3)
  status = cufftPlan1D(planY, ny, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan'

  batchSize = piZ%shape(2)*piZ%shape(3)
  status = cufftPlan1D(planZ, nz, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan'


  ! Physical grid

  allocate(x(nx), y(ny), z(nz))

  hx = Lx/nx
  do i = 1, nx
     x(i) = hx*i
  enddo

  hy = Ly/ny
  do j = 1, ny
     y(j) = hy*j
  enddo

  hz = Lz/nz
  do k = 1, nz
     z(k) = hz*k
  enddo

  ! Wavenumbers

  allocate(kx(nx), ky(ny), kz(nz))

  do i = 1, nx/2
     kx(i) = (i-1)*(twoPi/Lx)
  enddo
  do i = nx/2+1, nx
     kx(i) = (i-1-nx)*(twoPi/Lx)
  enddo
  allocate(kx_d, source=kx)

  do j = 1, ny/2
     ky(j) = (j-1)*(twoPi/Ly)
  enddo
  do j = ny/2+1, ny
     ky(j) = (j-1-ny)*(twoPi/Ly)
  enddo
  allocate(ky_d, source=ky)

  do k = 1, nz/2
     kz(k) = (k-1)*(twoPi/Lz)
  enddo
  do k = nz/2+1, nz
     kz(k) = (k-1-nz)*(twoPi/Lz)
  enddo
  allocate(kz_d, source=kz)

  ! allocate arrays

  allocate(phi(max(nElemX, nElemY, nElemZ)))
  allocate(phi_d, mold=phi)
  allocate(ua(nx, piX%shape(2), piX%shape(3)))
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, nElemWork))

  ! initialize phi and analytical solution
  block
    complex(8), pointer :: phi3(:,:,:)
    integer :: jl, kl, jg, kg
    npx = piX%shape(1)
    npy = piX%shape(2)
    npz = piX%shape(3)
    call c_f_pointer(c_loc(phi), phi3, [npx, npy, npz])

    do kl = 1, npz
       kg = piX%lo(3) + kl - 1
       do jl = 1, npy
          jg = piX%lo(2) + jl - 1
          do i = 1, nx
             phi3(i,jl,kl) = cmplx(sin(twoPi*Mx*x(i))*sin(twoPi*My*y(jg))*sin(twoPi*Mz*z(kg)),0.0)
             ua(i,jl,kl) = -phi3(i,jl,kl)/(twoPi**2*(Mx**2 + My**2 + Mz**2))
          enddo
       enddo
    enddo
  end block

  ! H2D transfer
  phi_d = phi

  ! phi(x,y,z) -> phi(kx,y,z)
  status = cufftExecZ2Z(planX, phi_d, phi_d, CUFFT_FORWARD)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X forward error: ', status
  ! phi(kx,y,z) -> phi(y,z,kx)
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(y,z,kx) -> phi(ky,z,kx)
  status = cufftExecZ2Z(planY, phi_d, phi_d, CUFFT_FORWARD)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y forward error: ', status
  ! phi(ky,z,kx) -> phi(z,kx,ky)
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(z,kx,ky) -> phi(kz,kx,ky)
  status = cufftExecZ2Z(planZ, phi_d, phi_d, CUFFT_FORWARD)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z forward error: ', status

  block
    complex(8), device, pointer :: phi3d(:,:,:)
    real(8) :: k2
    integer :: il, jl, ig, jg
    integer :: offsets(3), xoff, yoff
    integer :: np(3)
    np(piZ%order(1)) = piZ%shape(1)
    np(piZ%order(2)) = piZ%shape(2)
    np(piZ%order(3)) = piZ%shape(3)
    call c_f_pointer(c_devloc(phi_d), phi3d, piZ%shape)


    ! divide by -K**2, and normalize

    offsets(piZ%order(1)) = piZ%lo(1) - 1
    offsets(piZ%order(2)) = piZ%lo(2) - 1
    offsets(piZ%order(3)) = piZ%lo(3) - 1

    xoff = offsets(1)
    yoff = offsets(2)
    npx = np(1)
    npy = np(2)
    !$cuf kernel do (2)
    do jl = 1, npy
       do il = 1, npx
          jg = yoff + jl
          ig = xoff + il
          do k = 1, nz
             k2 = kx_d(ig)**2 + ky_d(jg)**2 + kz_d(k)**2
             phi3d(k,il,jl) = -phi3d(k,il,jl)/k2/(nx*ny*nz)
          enddo
       enddo
    enddo

    ! specify mean (corrects division by zero wavenumber above)
    if (xoff == 0 .and. yoff == 0) phi3d(1,1,1) = 0.0

  end block

  ! phi(kz,kx,ky) -> phi(z,kx,ky)
  status = cufftExecZ2Z(planZ, phi_d, phi_d, CUFFT_INVERSE)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z inverse error: ', status
  ! phi(z,kx,ky) -> phi(ky,z,kx)
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(ky,z,kx) -> phi(y,z,kx)
  status = cufftExecZ2Z(planY, phi_d, phi_d, CUFFT_INVERSE)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y inverse error: ', status
  ! phi(y,z,kx) -> phi(kx,y,z)
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(kx,y,z) -> phi(x,y,z)
  status = cufftExecZ2Z(planX, phi_d, phi_d, CUFFT_INVERSE)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X inverse error: ', status

  phi = phi_d

  ! check results

  block
    complex(8), pointer :: phi3(:,:,:)
    real(8) :: err, maxErr = -1.0
    integer :: jl, kl
    npx = piX%shape(1)
    npy = piX%shape(2)
    npz = piX%shape(3)
    call c_f_pointer(c_loc(phi), phi3, [npx, npy, npz])


    do kl = 1, npz
       do jl = 1, npy
          do i = 1, nx
             err = abs(ua(i,jl,kl)-phi3(i,jl,kl))
             if (err > maxErr) maxErr = err
          enddo
       enddo
    enddo

    write(*,"('[', i0, '] Max Error: ', e12.6)") rank, maxErr
  end block

  ! cleanup

  status = cufftDestroy(planX)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X plan destroy: ', status
  status = cufftDestroy(planY)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y plan destroy: ', status
  status = cufftDestroy(planZ)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z plan destroy: ', status

  deallocate(x, y, z)
  deallocate(kx, ky, kz)
  deallocate(kx_d, ky_d, kz_d)
  deallocate(phi, phi_d, ua)

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))

  call mpi_finalize(ierr)

end program main
