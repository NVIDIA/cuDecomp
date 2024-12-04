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

module precision
  use mpi, only: MPI_REAL,MPI_DOUBLE_PRECISION
  use cudecomp, only: CUDECOMP_DOUBLE, CUDECOMP_DOUBLE_COMPLEX, CUDECOMP_FLOAT, CUDECOMP_FLOAT_COMPLEX
  use cufft, only: CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z, CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
  use iso_fortran_env, only: int64,real32,real64
  implicit none
  integer, parameter, public :: i8 = int64
#ifdef SINGLE
#define cufftExecD2Z cufftExecR2C
#define cufftExecZ2D cufftExecC2R
#define cufftExecZ2Z cufftExecC2C
  integer, parameter, public :: rp = real32
  integer, parameter, public :: MPI_REAL_RP = MPI_REAL
  integer, parameter, public :: CUDECOMP_REAL_RP = CUDECOMP_FLOAT
  integer, parameter, public :: CUDECOMP_REAL_COMPLEX_RP = CUDECOMP_FLOAT_COMPLEX
  integer, parameter, public :: CUFFT_R2C_RP = CUFFT_R2C
  integer, parameter, public :: CUFFT_C2R_RP = CUFFT_C2R
  integer, parameter, public :: CUFFT_C2C_RP = CUFFT_C2C
#else
  integer, parameter, public :: rp = real64
  integer, parameter, public :: MPI_REAL_RP = MPI_DOUBLE_PRECISION
  integer, parameter, public :: CUDECOMP_REAL_RP = CUDECOMP_DOUBLE
  integer, parameter, public :: CUDECOMP_REAL_COMPLEX_RP = CUDECOMP_DOUBLE_COMPLEX
  integer, parameter, public :: CUFFT_R2C_RP = CUFFT_D2Z
  integer, parameter, public :: CUFFT_C2R_RP = CUFFT_Z2D
  integer, parameter, public :: CUFFT_C2C_RP = CUFFT_Z2Z
#endif
end module precision

program taylor_green
  use precision
  use mpi
  use cudecomp
  use cufft
  implicit none

  integer :: commType = 0
  integer :: rank, ranks, ierr, numDev
  integer :: localRank, localComm
  integer :: pdims(2) = [0,0]

  type(cudecompHandle) :: handle
  type(cudecompGridDesc) :: gridDescR, gridDescC
  type(cudecompGridDescConfig) :: gridDescConfigR, gridDescConfigC
  type(cudecompGridDescAutotuneOptions) :: optionsR, optionsC
  integer :: gdimR(3), gdimC(3)
  type(cudecompPencilInfo) :: piXR, piXC, piYC, piZC
  integer(i8) :: nElemXR, nElemXC, nElemYC, nElemZC, nElemWorkC, workSize
  logical :: axisContiguous = .true.

  real(rp) :: pi = 4.0_rp*atan(1.0_rp)
  integer :: i, j, k, status
  integer :: il, ig, kl, jl, kg, jg
  logical :: printPencilInfo = .false.
  complex(rp), pointer, device, contiguous :: workC(:)

  integer :: planD2ZX, planZ2DX, planZ2ZY, planZ2ZZ
  integer :: batchSize
  integer(i8) :: sizeD2ZX, sizeZ2DX, sizeZ2ZY, sizeZ2ZZ
  integer(i8) :: nElemWorkCUFFT, nElemWorkDecomp

  logical :: skip_next
  character(len=16) :: arg

  integer :: N=256
  integer :: kmax,niter=1000,stat_freq=100
  real(rp) :: nu=0.000625
  real(rp) :: dt=0.001
  real(rp) :: maxtime=0.0
  real(rp) :: cfl=0.0
  real(rp) :: spec_freq=0.0
  real(rp) :: dx,dy,dz
  real(rp) :: ke,enst
  real(rp), allocatable, managed :: kx(:),ky(:),kz(:)

  real(rp) :: N3
  real(rp) :: sumsql,sumsq,velmaxl,velmax
  real(rp) :: u,v,w
  real(rp) :: kmaxr
  integer :: num_shells, spec_count
  real(rp), allocatable, managed :: ek(:)

  real(8) :: ts,te,ts_step,te_step
  real(rp) :: flowtime=0._rp
  integer :: count=0

  real(rp), allocatable, target, managed :: workspace(:)
  real(rp), allocatable, target, managed :: dworkspace(:)
  real(rp), allocatable, target, managed :: hworkspace(:,:)
  real(rp), pointer, managed :: Ur(:,:,:),Vr(:,:,:),Wr(:,:,:)
  complex(rp), pointer, managed :: Uc(:,:,:),Vc(:,:,:),Wc(:,:,:)
  real(rp), pointer, managed :: dUr(:,:,:),dVr(:,:,:),dWr(:,:,:)
  complex(rp), pointer, managed :: dUc(:,:,:),dVc(:,:,:),dWc(:,:,:)
  real(rp), pointer, managed :: Uh1r(:,:,:),Vh1r(:,:,:),Wh1r(:,:,:)
  real(rp), pointer, managed :: Uh2r(:,:,:),Vh2r(:,:,:),Wh2r(:,:,:)
  real(rp), pointer, managed :: Uh3r(:,:,:),Vh3r(:,:,:),Wh3r(:,:,:)
  complex(rp), pointer, managed :: Uh1c(:,:,:),Vh1c(:,:,:),Wh1c(:,:,:)
  complex(rp), pointer, managed :: Uh2c(:,:,:),Vh2c(:,:,:),Wh2c(:,:,:)
  complex(rp), pointer, managed :: Uh3c(:,:,:),Vh3c(:,:,:),Wh3c(:,:,:)

  call initialize
  if( maxtime > 0.0_rp ) niter = 1000000
  if( cfl > 0.0_rp ) dt = 0.0_rp
  call print_stats
  if( spec_freq > 0.0_rp ) call write_spectrum_sample(0)

  status=cudaDeviceSynchronize()
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  ts = MPI_Wtime()
  ts_step = MPI_Wtime()

  do i=1,niter

    call step
    count = count + 1

    if( spec_freq > 0.0_rp .and. flowtime >= real(spec_count + 1,rp)*spec_freq ) then
      call write_spectrum_sample(spec_count+1)
      spec_count = spec_count + 1  
    end if

    if( mod(i,stat_freq) == 0 ) then
      status=cudaDeviceSynchronize()
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      te_step = MPI_Wtime()
      if (rank == 0) write(*,"(' Average iteration time: ',F12.6,' ms')") (te_step - ts_step) * 1000._rp / real(count,rp)
      count = 0

      call print_stats

      status=cudaDeviceSynchronize()
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      ts_step = MPI_Wtime()
    end if

    if( maxtime > 0._rp .and. flowtime >= maxtime ) exit

  end do

  status=cudaDeviceSynchronize()
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  te = MPI_Wtime()
  if (rank == 0) write(*,"(' Simulation time: ',F12.6,' s')") te - ts

  call finalize

  status=cudaDeviceSynchronize()
  call mpi_finalize(ierr)

contains

subroutine initializeField(Ur,Vr,Wr,dx,dy,dz,piXR)
  real(rp), managed :: Ur(:,:,:),Vr(:,:,:),Wr(:,:,:)
  type(cudecompPencilInfo) :: piXR
  integer :: il,ig,jl,jg,kl,kg
  real(rp) :: dx,dy,dz

!$acc parallel loop collapse(3)
  do kl=1,piXR%shape(3)
    do jl=1,piXr%shape(2)
      do il=1,piXR%shape(1)-2
        kg= kl + piXR%lo(3) - 1
        jg= jl + piXR%lo(2) - 1
        ig= il + piXR%lo(1) - 1

        Ur(il,jl,kl)=sin((ig-1)*dx) * cos((jg-1)*dy) * cos((kg-1)*dz)
        Vr(il,jl,kl)=-cos((ig-1)*dx) * sin((jg-1)*dy) * cos((kg-1)*dz)
        Wr(il,jl,kl)=0._rp
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()
end subroutine

subroutine curl(U,V,W,dU,dV,dW)
  complex(rp), managed :: U(:,:,:),V(:,:,:),W(:,:,:)
  complex(rp), managed :: dU(:,:,:),dV(:,:,:),dW(:,:,:)
  integer :: il,ig,jl,jg,kl,kg

!$acc parallel loop collapse(3)
  do jl=1,piZC%shape(3)
    do il=1,piZC%shape(2)
      do kl=1,piZC%shape(1)
        kg= kl + piZC%lo(1) - 1
        ig= il + piZC%lo(2) - 1
        jg= jl + piZC%lo(3) - 1

        dU(kl,il,jl)=cmplx(0.,1.,rp)*( ky(jg)*W(kl,il,jl)-kz(kg)*V(kl,il,jl) )
        dV(kl,il,jl)=cmplx(0.,1.,rp)*( kz(kg)*U(kl,il,jl)-kx(ig)*W(kl,il,jl) )
        dW(kl,il,jl)=cmplx(0.,1.,rp)*( kx(ig)*V(kl,il,jl)-ky(jg)*U(kl,il,jl) )
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()
end subroutine

subroutine cross(U,V,W,dU,dV,dW)
  real(rp), managed :: U(:,:,:),V(:,:,:),W(:,:,:)
  real(rp), managed :: dU(:,:,:),dV(:,:,:),dW(:,:,:)
  real(rp) :: Uu,Uv,Uw,dUu,dUv,dUw
  integer :: il,ig,jl,jg,kl,kg

!$acc parallel loop collapse(3)
  do kl=1,piXR%shape(3)
    do jl=1,piXR%shape(2)
      do il=1,piXR%shape(1)-2
        Uu = U(il,jl,kl)/N3
        Uv = V(il,jl,kl)/N3
        Uw = W(il,jl,kl)/N3
        dUu = dU(il,jl,kl)/N3
        dUv = dV(il,jl,kl)/N3
        dUw = dW(il,jl,kl)/N3
        dU(il,jl,kl) = Uv*dUw - Uw*dUv
        dV(il,jl,kl) = Uw*dUu - Uu*dUw
        dW(il,jl,kl) = Uu*dUv - Uv*dUu  
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()
end subroutine

subroutine compute_dU(U,V,W,dU,dV,dW)
  complex(rp), managed :: U(:,:,:),V(:,:,:),W(:,:,:)
  complex(rp), managed :: dU(:,:,:),dV(:,:,:),dW(:,:,:)
  real(rp) :: k2
  complex(rp) :: dUu,dUv,dUw,Ph
  logical :: alias
  integer :: il,ig,jl,jg,kl,kg

!$acc parallel loop collapse(3)
  do jl=1,piZC%shape(3)
    do il=1,piZC%shape(2)
      do kl=1,piZC%shape(1)
        kg= kl + piZC%lo(1) - 1
        ig= il + piZC%lo(2) - 1
        jg= jl + piZC%lo(3) - 1

        k2 = kx(ig)*kx(ig) + ky(jg)*ky(jg) + kz(kg)*kz(kg)
        alias = (abs(kx(ig)) >= kmaxr .or. abs(ky(jg)) >= kmaxr .or. abs(kz(kg)) >= kmaxr)

        dUu = dU(kl,il,jl)
        dUv = dV(kl,il,jl)
        dUw = dW(kl,il,jl)
        Ph = kx(ig)*dUu + ky(jg)*dUv + kz(kg)*dUw
        if( k2 /= 0._rp ) Ph = Ph/k2

        if( alias ) then
          dU(kl,il,jl) = -nu*k2*U(kl,il,jl)
          dV(kl,il,jl) = -nu*k2*V(kl,il,jl)
          dW(kl,il,jl) = -nu*k2*W(kl,il,jl)
        else
          dU(kl,il,jl) = dU(kl,il,jl)-Ph*kx(ig)-nu*k2*U(kl,il,jl)
          dV(kl,il,jl) = dV(kl,il,jl)-Ph*ky(jg)-nu*k2*V(kl,il,jl)
          dW(kl,il,jl) = dW(kl,il,jl)-Ph*kz(kg)-nu*k2*W(kl,il,jl)
        end if
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()
end subroutine

subroutine forward(work)
  real(rp), managed :: work(:)
  integer :: i
  integer(i8) :: start

  do i=1,3
    start=1+(i-1)*workSize
    status = cufftExecD2Z(planD2ZX, work(start), work(start))
    CHECK_CUDECOMP_EXIT(cudecompTransposeXtoY(handle, gridDescC, work(start), work(start), workC, CUDECOMP_REAL_COMPLEX_RP))
    status = cufftExecZ2Z(planZ2ZY, work(start), work(start), CUFFT_FORWARD)
    CHECK_CUDECOMP_EXIT(cudecompTransposeYtoZ(handle, gridDescC, work(start), work(start), workC, CUDECOMP_REAL_COMPLEX_RP))
    status = cufftExecZ2Z(planZ2ZZ, work(start), work(start), CUFFT_FORWARD)
  end do
end subroutine

subroutine backward(work)
  real(rp), managed :: work(:)
  integer :: i
  integer(i8) :: start

  do i=1,3
    start=1+(i-1)*workSize
    status = cufftExecZ2Z(planZ2ZZ, work(start), work(start), CUFFT_INVERSE)
    CHECK_CUDECOMP_EXIT(cudecompTransposeZtoY(handle, gridDescC, work(start), work(start), workC, CUDECOMP_REAL_COMPLEX_RP))
    status = cufftExecZ2Z(planZ2ZY, work(start), work(start), CUFFT_INVERSE)
    CHECK_CUDECOMP_EXIT(cudecompTransposeYtoX(handle, gridDescC, work(start), work(start), workC, CUDECOMP_REAL_COMPLEX_RP))
    status = cufftExecZ2D(planZ2DX, work(start), work(start))
  end do
end subroutine

subroutine print_stats
  call curl(Uh1c,Vh1c,Wh1c,dUc,dVc,dWc)
  call backward(dworkspace)

  sumsql = 0._rp
!$acc parallel loop collapse(3) reduction(+:sumsql)
  do kl=1,piXR%shape(3)
    do jl=1,piXr%shape(2)
      do il=1,piXR%shape(1)-2
        u = dUr(il,jl,kl)/N3
        v = dVr(il,jl,kl)/N3
        w = dWr(il,jl,kl)/N3
        sumsql = sumsql + u*u + v*v + w*w
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()

  sumsq = 0._rp
  call mpi_reduce(sumsql, sumsq, 1, MPI_REAL_RP, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  enst = 0.5_rp * sumsq / N3

  sumsql = 0._rp
!$acc parallel loop collapse(3) reduction(+:sumsql)
  do kl=1,piXR%shape(3)
    do jl=1,piXr%shape(2)
      do il=1,piXR%shape(1)-2
        u = Ur(il,jl,kl)/N3
        v = Vr(il,jl,kl)/N3
        w = Wr(il,jl,kl)/N3
        sumsql = sumsql + u*u + v*v + w*w
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()

  sumsq = 0._rp
  call mpi_reduce(sumsql, sumsq, 1, MPI_REAL_RP, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  ke = 0.5_rp * sumsq / N3

  if (rank == 0) write(*,"(' flowtime: ',F8.5,' ke: ',F16.14,' enstrophy: ',F17.14,' dt: ',F16.14)") flowtime,ke,enst,dt
end subroutine

subroutine write_spectrum_sample(idx)
  implicit none
  integer, intent(in) :: idx
  integer :: i
  character(len=100) :: filename
  character(len=8) :: idx_str
  real(rp) :: scale_factor
  integer :: ios
  integer :: il,ig,jl,jg,kl,kg
  integer :: shell
  real(rp) :: kk, uu
  complex(rp) :: Uhu,Uhv,Uhw

!$acc kernels
  ek = 0._rp
!$acc end kernels

!$acc parallel loop collapse(3)
  do jl=1,piZC%shape(3)
    do il=1,piZC%shape(2)
      do kl=1,piZC%shape(1)
        kg= kl + piZC%lo(1) - 1
        ig= il + piZC%lo(2) - 1
        jg= jl + piZC%lo(3) - 1

        Uhu = Uh1c(kl,il,jl)
        Uhv = Vh1c(kl,il,jl)
        Uhw = Wh1c(kl,il,jl)

        kk = kx(ig)*kx(ig) + ky(jg)*ky(jg) + kz(kg)*kz(kg)
        shell = int(sqrt(kk) + 0.5_rp) + 1
        uu = abs(Uhu)*abs(Uhu) + abs(Uhv)*abs(Uhv) + abs(Uhw)*abs(Uhw)
        if( kx(ig) .eq. 0._rp ) uu = 0.5_rp*uu
!$acc atomic update
        ek(shell) = ek(shell) + uu
      end do
    end do
  end do
!$acc end parallel

  status=cudaDeviceSynchronize()

  if (rank == 0) then
    call mpi_reduce(MPI_IN_PLACE, ek, num_shells, MPI_REAL_RP, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  else
    call mpi_reduce(ek, ek, num_shells, MPI_REAL_RP, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  end if

  if (rank == 0) then
    write(idx_str, '(I8.8)') idx
    filename = 'spectrum_' // trim(adjustl(idx_str)) // '.txt'

    print*, 'writing spectrum sample to ', trim(filename), '...'
    open(unit=10, file=trim(filename), status='unknown', iostat=ios)
    if (ios /= 0) then
      print*, 'Error opening file ', trim(filename)
      return
    end if

    write(10, '(A, F14.6)') 'time: ', flowtime
    scale_factor = 1.0_rp / real(N3,rp)
    do i = 1, num_shells
      write(10, '(E24.16)') ek(i) * scale_factor
    end do

    close(10)
  end if
end subroutine

subroutine compute_rhs
  call curl(Uh2c,Vh2c,Wh2c,dUc,dVc,dWc)
  call backward(dworkspace)
  call cross(Ur,Vr,Wr,dUr,dVr,dWr)
  call forward(dworkspace)
  kmaxr = 2.0_rp/3.0_rp * real(N/2+1,rp)
  call compute_dU(Uh2c,Vh2c,Wh2c,dUc,dVc,dWc)
end subroutine

subroutine RK4
!$acc kernels
  Uh2c = Uh1c
  Vh2c = Vh1c
  Wh2c = Wh1c
!$acc end kernels

  call compute_rhs

!$acc kernels
  Uh2c = Uh1c + 0.5_rp*dt*dUc
  Vh2c = Vh1c + 0.5_rp*dt*dVc
  Wh2c = Wh1c + 0.5_rp*dt*dWc
!$acc end kernels

!$acc kernels
  Uc = Uh2c
  Vc = Vh2c
  Wc = Wh2c
!$acc end kernels

  call backward(workspace)

!$acc kernels
  Uh3c = Uh1c + (1._rp/6._rp)*dt*dUc
  Vh3c = Vh1c + (1._rp/6._rp)*dt*dVc
  Wh3c = Wh1c + (1._rp/6._rp)*dt*dWc
!$acc end kernels

  call compute_rhs

!$acc kernels
  Uh2c = Uh1c + 0.5_rp*dt*dUc
  Vh2c = Vh1c + 0.5_rp*dt*dVc
  Wh2c = Wh1c + 0.5_rp*dt*dWc
!$acc end kernels

!$acc kernels
  Uc = Uh2c
  Vc = Vh2c
  Wc = Wh2c
!$acc end kernels

  call backward(workspace)

!$acc kernels
  Uh3c = Uh3c + (1._rp/3._rp)*dt*dUc
  Vh3c = Vh3c + (1._rp/3._rp)*dt*dVc
  Wh3c = Wh3c + (1._rp/3._rp)*dt*dWc
!$acc end kernels

  call compute_rhs

!$acc kernels
  Uh2c = Uh1c + 1._rp*dt*dUc
  Vh2c = Vh1c + 1._rp*dt*dVc
  Wh2c = Wh1c + 1._rp*dt*dWc
!$acc end kernels

!$acc kernels
  Uc = Uh2c
  Vc = Vh2c
  Wc = Wh2c
!$acc end kernels

  call backward(workspace)

!$acc kernels
  Uh3c = Uh3c + (1._rp/3._rp)*dt*dUc
  Vh3c = Vh3c + (1._rp/3._rp)*dt*dVc
  Wh3c = Wh3c + (1._rp/3._rp)*dt*dWc
!$acc end kernels

  call compute_rhs

!$acc kernels
  Uh1c = Uh3c + (1._rp/6._rp)*dt*dUc
  Vh1c = Vh3c + (1._rp/6._rp)*dt*dVc
  Wh1c = Wh3c + (1._rp/6._rp)*dt*dWc
!$acc end kernels

!$acc kernels
  Uc = Uh1c
  Vc = Vh1c
  Wc = Wh1c
!$acc end kernels

  call backward(workspace)
end subroutine

subroutine get_dt
  if( cfl > 0.0_rp ) then
    velmaxl = 0._rp
!$acc parallel loop collapse(3) reduction(max:velmaxl)
    do kl=1,piXR%shape(3)
      do jl=1,piXr%shape(2)
        do il=1,piXR%shape(1)-2
          u = Ur(il,jl,kl)/N3
          v = Vr(il,jl,kl)/N3
          w = Wr(il,jl,kl)/N3
          velmaxl = max(sqrt(u*u + v*v + w*w), velmaxl)
        end do
      end do
    end do
!$acc end parallel

    status=cudaDeviceSynchronize()

    velmax = 0._rp
    call mpi_allreduce(velmaxl, velmax, 1, MPI_REAL_RP, MPI_MAX, MPI_COMM_WORLD, ierr)

    dt = cfl * dx / velmax
  end if
end subroutine

subroutine step
  call get_dt
  call RK4
  flowtime = flowtime + dt
end subroutine

subroutine initialize
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
  ierr = cudaGetDeviceCount(numDev)
  ierr = cudaSetDevice(mod(localRank,numDev))

  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--N')
        call get_command_argument(i+1, arg)
        read(arg, *) N
        skip_next = .true.
      case('--niter')
        call get_command_argument(i+1, arg)
        read(arg, *) niter
        skip_next = .true.
      case('--max_flowtime')
        call get_command_argument(i+1, arg)
        read(arg, *) maxtime
        skip_next = .true.
      case('--printfreq')
        call get_command_argument(i+1, arg)
        read(arg, *) stat_freq
        skip_next = .true.
      case('--specfreq')
        call get_command_argument(i+1, arg)
        read(arg, *) spec_freq
        skip_next = .true.
      case('--nu')
        call get_command_argument(i+1, arg)
        read(arg, *) nu
        skip_next = .true.
      case('--dt')
        call get_command_argument(i+1, arg)
        read(arg, *) dt
        skip_next = .true.
      case('--cfl')
        call get_command_argument(i+1, arg)
        read(arg, *) cfl
        skip_next = .true.
      case('--help')
        if(rank==0) print*, "Command line arguments: --N --niter --max_flowtime --printfreq --specfreq --nu --dt --cfl --help"
        call MPI_Finalize(ierr)
        call exit(0)
      case default
        if(rank==0) print*, "Unknown argument."
        call MPI_Finalize(ierr)
        call exit(1)
    end select
  end do

  if (rank == 0) then
    write(*,"('running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid...')") N, N, N
  end if

  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(gridDescConfigC))
  gridDescConfigC%pdims = pdims
  gdimC = [N/2+1, N, N]
  gridDescConfigC%gdims = gdimC
  gridDescConfigC%transpose_comm_backend = commType
  gridDescConfigC%transpose_axis_contiguous = axisContiguous

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(optionsC))
  optionsC%dtype = CUDECOMP_REAL_COMPLEX_RP
  if (commType == 0) then
    optionsC%autotune_transpose_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, gridDescC, gridDescConfigC, optionsC))

  pdims = gridDescConfigC%pdims

  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(gridDescConfigR))
  gridDescConfigR%pdims = pdims
  gdimR = [(N/2+1)*2, N, N]
  gridDescConfigR%gdims = gdimR
  gridDescConfigR%transpose_comm_backend = gridDescConfigC%transpose_comm_backend
  gridDescConfigR%transpose_axis_contiguous = axisContiguous

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(optionsR))
  optionsR%dtype = CUDECOMP_REAL_RP

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, gridDescR, gridDescConfigR, optionsR))

  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, gridDescR, piXR, 1))
  nElemXR = product(piXR%shape)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, gridDescC, piXC, 1))
  nElemXC = product(piXC%shape)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, gridDescC, piYC, 2))
  nElemYC = product(piYC%shape)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, gridDescC, piZC, 3))
  nElemZC = product(piZC%shape)

  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, gridDescC, nElemWorkDecomp))

  if (printPencilInfo) then
    write(*,"('[',i0,'] piXR%shape = ',i0,' x ',i0,' x ',i0)") rank, piXR%shape
    write(*,"('[',i0,'] piXC%shape = ',i0,' x ',i0,' x ',i0)") rank, piXC%shape
    write(*,"('[',i0,'] piYC%shape = ',i0,' x ',i0,' x ',i0)") rank, piYC%shape
    write(*,"('[',i0,'] piZC%shape = ',i0,' x ',i0,' x ',i0)") rank, piZC%shape

    write(*,"('[',i0,'] piXR%lo = ',i0,' x ',i0,' x ',i0)") rank, piXR%lo
    write(*,"('[',i0,'] piXC%lo = ',i0,' x ',i0,' x ',i0)") rank, piXC%lo
    write(*,"('[',i0,'] piYC%lo = ',i0,' x ',i0,' x ',i0)") rank, piYC%lo
    write(*,"('[',i0,'] piZC%lo = ',i0,' x ',i0,' x ',i0)") rank, piZC%lo

    write(*,"('[',i0,'] piXR%order = ',i0,' x ',i0,' x ',i0)") rank, piXR%order
    write(*,"('[',i0,'] piXC%order = ',i0,' x ',i0,' x ',i0)") rank, piXC%order
    write(*,"('[',i0,'] piYC%order = ',i0,' x ',i0,' x ',i0)") rank, piYC%order
    write(*,"('[',i0,'] piZC%order = ',i0,' x ',i0,' x ',i0)") rank, piZC%order

    write(*,"('[',i0,'] piXR%size = ',i0)") rank, piXR%size
    write(*,"('[',i0,'] piXC%size = ',i0)") rank, piXC%size
    write(*,"('[',i0,'] piYC%size = ',i0)") rank, piYC%size
    write(*,"('[',i0,'] piZC%size = ',i0)") rank, piZC%size
  end if

  batchSize = piXR%shape(2)*piXR%shape(3)
  status = cufftCreate(planD2ZX)
  status = cufftSetAutoAllocation(planD2ZX, 0)
  status = cufftMakePlan1D(planD2ZX, N, CUFFT_R2C_RP, batchSize, sizeD2ZX)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X R2C plan'

  batchSize = piXC%shape(2)*piXC%shape(3)
  status = cufftCreate(planZ2DX)
  status = cufftSetAutoAllocation(planZ2DX, 0)
  status = cufftMakePlan1D(planZ2DX, N, CUFFT_C2R_RP, batchSize, sizeZ2DX)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X C2R plan'

  batchSize = piYC%shape(2)*piYC%shape(3)
  status = cufftCreate(planZ2ZY)
  status = cufftSetAutoAllocation(planZ2ZY, 0)
  status = cufftMakePlan1D(planZ2ZY, N, CUFFT_C2C_RP, batchSize, sizeZ2ZY)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan'

  batchSize = piZC%shape(2)*piZC%shape(3)
  status = cufftCreate(planZ2ZZ)
  status = cufftSetAutoAllocation(planZ2ZZ, 0)
  status = cufftMakePlan1D(planZ2ZZ, N, CUFFT_C2C_RP, batchSize, sizeZ2ZZ)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan'

  nElemWorkCufft = max(sizeD2ZX,sizeZ2DX,sizeZ2ZY,sizeZ2ZZ)/(2*sizeof(1.0_rp))
  nElemWorkC = max(nElemWorkCufft,nElemWorkDecomp)
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, gridDescC, workC, nElemWorkC))

  status = cufftSetWorkArea(planD2ZX, workC)
  status = cufftSetWorkArea(planZ2DX, workC)
  status = cufftSetWorkArea(planZ2ZY, workC)
  status = cufftSetWorkArea(planZ2ZZ, workC)

  workSize=max(2*nElemXC, 2*nElemYC, 2*nElemZC, nElemXR)
  allocate(workspace(3*workSize))
  allocate(dworkspace(3*workSize))
  allocate(hworkspace(3*worksize,3))

  call c_f_pointer(c_loc(workspace              ),Ur    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(workspace              ),Uc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(workspace(1+  workSize)),Vr    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(workspace(1+  workSize)),Vc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(workspace(1+2*workSize)),Wr    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(workspace(1+2*workSize)),Wc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])

  call c_f_pointer(c_loc(dworkspace              ),dUr    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(dworkspace              ),dUc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(dworkspace(1+  workSize)),dVr    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(dworkspace(1+  workSize)),dVc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(dworkspace(1+2*workSize)),dWr    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(dworkspace(1+2*workSize)),dWc    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])

  call c_f_pointer(c_loc(hworkspace(1           ,1)),Uh1r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1           ,1)),Uh1c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,1)),Vh1r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,1)),Vh1c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,1)),Wh1r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,1)),Wh1c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])

  call c_f_pointer(c_loc(hworkspace(1           ,2)),Uh2r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1           ,2)),Uh2c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,2)),Vh2r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,2)),Vh2c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,2)),Wh2r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,2)),Wh2c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])

  call c_f_pointer(c_loc(hworkspace(1           ,3)),Uh3r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1           ,3)),Uh3c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,3)),Vh3r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+  workSize,3)),Vh3c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,3)),Wh3r    ,[piXR%shape(1),piXR%shape(2),piXR%shape(3)])
  call c_f_pointer(c_loc(hworkspace(1+2*workSize,3)),Wh3c    ,[piZC%shape(1),piZC%shape(2),piZC%shape(3)])

  kmax=N/2
  allocate(kx(N/2+1),ky(N),kz(N))

  kx = real((/ (i-1,i=1,kmax), -kmax /), rp)
  ky = real((/ (i-1,i=1,kmax),(i-1-N,i=kmax+1,N) /), rp)
  kz = real((/ (i-1,i=1,kmax),(i-1-N,i=kmax+1,N) /), rp)

  dx = 2._rp*pi/real(N,rp)
  dy = 2._rp*pi/real(N,rp)
  dz = 2._rp*pi/real(N,rp)

  call initializeField(Ur, Vr, Wr, dx, dy, dz, pixR)

!$acc kernels
  Uh1r = Ur
  Vh1r = Vr
  Wh1r = Wr
!$acc end kernels

  call forward(hworkspace(:,1))   
  status=cudaDeviceSynchronize()

  N3 = real(N,rp)*real(N,rp)*real(N,rp)
!$acc kernels
  Ur = Ur*N3
  Vr = Vr*N3
  Wr = Wr*N3
!$acc end kernels

  if(spec_freq > 0._rp) then
    num_shells = int(sqrt(9.0_rp*N*N + 4.0_rp*N + 4.0_rp) / 4.0_rp) + 1
    allocate(ek(num_shells))
    spec_count = 0
  end if
end subroutine

subroutine finalize
  deallocate(kx,ky,kz)
  deallocate(workspace)
  deallocate(dworkspace)
  deallocate(hworkspace)

  status = cufftDestroy(planD2ZX)
  status = cufftDestroy(planZ2DX)
  status = cufftDestroy(planZ2ZY)
  status = cufftDestroy(planZ2ZZ)

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, gridDescC, workC))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, gridDescC))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, gridDescR))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))
end subroutine

end program
