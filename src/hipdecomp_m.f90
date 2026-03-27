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

module hipdecomp
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: int64, real32, real64

  ! enumerators

  ! enum for hipDecomp transpose backend options
  enum, bind(c) ! hipdecompTransposeCommBackend
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_MPI_P2P = 1
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_MPI_P2P_PL = 2
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_MPI_A2A = 3
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_NCCL = 4
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_NCCL_PL = 5
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_NVSHMEM = 6
    enumerator :: HIPDECOMP_TRANSPOSE_COMM_NVSHMEM_PL = 7
  end enum

  ! enum for hipDecomp halo backend options
  enum, bind(c) ! hipdecompHaloCommBackend
    enumerator :: HIPDECOMP_HALO_COMM_MPI = 1
    enumerator :: HIPDECOMP_HALO_COMM_MPI_BLOCKING = 2
    enumerator :: HIPDECOMP_HALO_COMM_NCCL = 3
    enumerator :: HIPDECOMP_HALO_COMM_NVSHMEM = 4
    enumerator :: HIPDECOMP_HALO_COMM_NVSHMEM_BLOCKING = 5
  end enum

  ! enum for hipDecomp grid autotune setting
  enum, bind(c) ! hipdecompAutotuneGridMode
    enumerator :: HIPDECOMP_AUTOTUNE_GRID_TRANSPOSE = 0
    enumerator :: HIPDECOMP_AUTOTUNE_GRID_HALO = 1
  end enum

  ! enum for hipDecomp supported data types
  enum, bind(c) ! hipdecompDataType
    enumerator :: HIPDECOMP_FLOAT = -1
    enumerator :: HIPDECOMP_DOUBLE = -2
    enumerator :: HIPDECOMP_FLOAT_COMPLEX = -3
    enumerator :: HIPDECOMP_DOUBLE_COMPLEX = -4
  end enum

  ! enum for hipDecomp return values
  enum, bind(c) ! hipdecompResult
    enumerator :: HIPDECOMP_RESULT_SUCCESS = 0
    enumerator :: HIPDECOMP_RESULT_INVALID_USAGE = 1
    enumerator :: HIPDECOMP_RESULT_NOT_SUPPORTED = 2
    enumerator :: HIPDECOMP_RESULT_INTERNAL_ERROR = 3
    enumerator :: HIPDECOMP_RESULT_CUDA_ERROR = 4
    enumerator :: HIPDECOMP_RESULT_CUTENSOR_ERROR = 5
    enumerator :: HIPDECOMP_RESULT_MPI_ERROR = 6
    enumerator :: HIPDECOMP_RESULT_NCCL_ERROR = 7
    enumerator :: HIPDECOMP_RESULT_NVSHMEM_ERROR = 8
    enumerator :: HIPDECOMP_RESULT_NVML_ERROR = 9
  end enum

  ! types

  ! Opaque handle to hipDecomp handle
  type, bind(c) :: hipdecompHandle
    type(c_ptr) :: member
  end type hipdecompHandle

  ! Opaque handle to hipDecomp grid descriptor
  type, bind(c) :: hipdecompGridDesc
    type(c_ptr) :: member
  end type hipdecompGridDesc

  ! Structure defining configuration options for grid descriptor creation
  type, bind(c) :: hipdecompGridDescConfig
    ! Grid information
    integer(c_int32_t) :: gdims(3) ! dimensions of data grid
    integer(c_int32_t) :: gdims_dist(3) ! dimensions of data grid for distribution
    integer(c_int32_t) :: pdims(2) ! dimensions of process grid

    ! Transpose Options
    integer(c_int32_t) :: transpose_comm_backend
    logical(c_bool) :: transpose_axis_contiguous(3) ! flag (by axis) indicating whether pencil memory should be contiguous along pencil axis
    integer(c_int32_t) :: transpose_mem_order(3,3) ! user-specified memory ordering by axis, overrides transpose_axis_contiguous setting

    ! Halo Options
    integer(c_int32_t) :: halo_comm_backend ! communication backend to use for halo communication

  end type hipdecompGridDescConfig

  ! Structure defining autotuning options for grid descriptor creation
  type, bind(c) :: hipdecompGridDescAutotuneOptions
    ! General options
    integer(c_int32_t) :: n_warmup_trials ! number of warmup trials to run for each tested configuration during autotuning
    integer(c_int32_t) :: n_trials ! number of timed trials to run for each tested configuration during autotuning
    integer(c_int32_t) :: grid_mode ! which communication (transpose/halo) to use to autotune process grid
    integer(c_int32_t) :: dtype ! datatype to use during autotuning
    logical(c_bool) :: allow_uneven_decompositions ! flag to control whether autotuning allows uneven decompositions (based on gdims_dist if provided, gdims otherwise)
    logical(c_bool) :: disable_nccl_backends ! flag to disable NCCL backend options during autotuning
    logical(c_bool) :: disable_nvshmem_backends ! flag to disable NVSHMEM backend options during autotuning
    real(c_double) :: skip_threshold ! threshold used to skip testing slow configurations

    ! Transpose-specific options
    logical(c_bool) :: autotune_transpose_backend ! flag to enable transpose backend autotuning
    logical(c_bool) :: transpose_use_inplace_buffers(4) ! flag to control whether transpose autotuning uses in-place or out-of-place buffers
    real(c_double) :: transpose_op_weights(4) ! multiplicative weight to apply to trial time contribution by transpose operation
    integer(c_int32_t) :: transpose_input_halo_extents(3, 4) ! input_halo_extents argument to use during autotuning by transpose operation
    integer(c_int32_t) :: transpose_output_halo_extents(3, 4) ! output_halo_extents argument to use during autotuning by transpose operation
    integer(c_int32_t) :: transpose_input_padding(3, 4) ! input_padding argument to use during autotuning by transpose operation
    integer(c_int32_t) :: transpose_output_padding(3, 4) ! output_padding argument to use during autotuning by transpose operation

    ! Halo-specific options
    logical(c_bool) :: autotune_halo_backend ! flag to enable halo backend autotuning
    integer(c_int32_t) :: halo_extents(3) ! extents for halo autotuning
    logical(c_bool) :: halo_periods(3) ! periodicity for halo autotuning
    integer(c_int32_t) :: halo_axis ! which axis pencils to use for halo autotuning
    integer(c_int32_t) :: halo_padding(3) ! padding argument for halo autotuning
  end type hipdecompGridDescAutotuneOptions

  ! Info structure containing pencil specific information
  type, bind(c) :: hipdecompPencilInfo
    integer(c_int32_t) :: shape(3)        ! pencil shape (in local order, including halo elements)
    integer(c_int32_t) :: lo(3)           ! lower bound coordinates (in local order, excluding halo elements)
    integer(c_int32_t) :: hi(3)           ! upper bound coordinates (in local order, excluding halo elements)
    integer(c_int32_t) :: order(3)        ! data layout order (e.g. 3,2,1 means memory is ordered z,y,x)
    integer(c_int32_t) :: halo_extents(3) ! halo extents by dimension (always in x,y,z order)
    integer(c_int32_t) :: padding(3)      ! padding by dimension (always in x,y,z order)
    integer(c_int64_t) :: size            ! number of elements in pencil (including halo elements)
  end type hipdecompPencilInfo

  ! interfaces

  ! hipDecomp initialization/finalization functions
  ! generic interface that takes either integer or type(MPI_Comm) communicator arguments

  interface hipdecompInit
    module procedure hipdecompInit_MPI_F, hipdecompInit_MPI_F08
  end interface hipdecompInit

  interface
    function hipdecompInit_FC(handle, mpi_comm) bind(C, name="hipdecompInit_F") result(res)
      import
      type(hipdecompHandle) :: handle
      integer, value :: mpi_comm
      integer(c_int) :: res
    end function hipdecompInit_FC
  end interface

  interface
    function hipdecompFinalize(handle) bind(C, name="hipdecompFinalize") result(res)
      import
      type(hipdecompHandle), value :: handle
      integer(c_int) :: res
    end function hipdecompFinalize
  end interface

  interface
    function hipdecompGridDescCreateC(handle, grid_desc, config, options) &
       bind(C, name="hipdecompGridDescCreate") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc) :: grid_desc
      type(hipdecompGridDescConfig) :: config
      type(hipdecompGridDescAutotuneOptions) :: options
      integer(c_int) :: res
    end function hipdecompGridDescCreateC
  end interface

  interface
    function hipdecompGridDescCreateC_nullopt(handle, grid_desc, config, options) &
       bind(C, name="hipdecompGridDescCreate") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc) :: grid_desc
      type(hipdecompGridDescConfig) :: config
      type(c_ptr), value :: options
      integer(c_int) :: res
    end function hipdecompGridDescCreateC_nullopt
  end interface

  interface
    function hipdecompGridDescDestroy(handle, grid_desc) bind(C, name="hipdecompGridDescDestroy") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      integer(c_int) :: res
    end function hipdecompGridDescDestroy
  end interface

  ! hipdecompGridDescConfig creation/manipulation functions
  interface
    function hipdecompGridDescConfigSetDefaults(config) &
      bind(C, name="hipdecompGridDescConfigSetDefaults") result(res)
      import
      type(hipdecompGridDescConfig) :: config
      integer(c_int) :: res
    end function hipdecompGridDescConfigSetDefaults
  end interface

  ! hipdecompGridDescAutotuneOptions creation/manipulation functions
  interface
    function hipdecompGridDescAutotuneOptionsSetDefaultsC(options) &
      bind(C, name="hipdecompGridDescAutotuneOptionsSetDefaults") result(res)
      import
      type(hipdecompGridDescAutotuneOptions) :: options
      integer(c_int) :: res
    end function hipdecompGridDescAutotuneOptionsSetDefaultsC
  end interface

  ! General functions
  interface
    function hipdecompGetPencilInfoC(handle, grid_desc, pencil_info, axis, halo_extents, padding) &
      bind(C, name="hipdecompGetPencilInfo") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      integer(c_int32_t), value :: axis
      integer(c_int32_t) :: halo_extents(3)
      integer(c_int32_t) :: padding(3)
      type(hipdecompPencilInfo) :: pencil_info
      integer(c_int) :: res
    end function hipdecompGetPencilInfoC
  end interface

  interface
    function hipdecompGetGridDescConfigC(handle, grid_desc, config) &
      bind(C, name="hipdecompGetGridDescConfig") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      type(hipdecompGridDescConfig) :: config
      integer(c_int) :: res
    end function hipdecompGetGridDescConfigC
  end interface

  interface
    function hipdecompGetTransposeWorkspaceSize(handle, grid_desc, workspace_size) &
      bind(C, name="hipdecompGetTransposeWorkspaceSize") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      integer(c_int64_t) :: workspace_size
      integer(c_int) :: res
    end function hipdecompGetTransposeWorkspaceSize
  end interface

  interface
    function hipdecompGetHaloWorkspaceSizeC(handle, grid_desc, axis, halo_extents, workspace_size) &
      bind(C, name="hipdecompGetHaloWorkspaceSize") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      integer(c_int32_t), value :: axis
      integer(c_int32_t) :: halo_extents(3)
      integer(c_int64_t) :: workspace_size
      integer(c_int) :: res
    end function hipdecompGetHaloWorkspaceSizeC
  end interface

  interface
    function hipdecompMallocC(handle, grid_desc, buffer, buffer_size_bytes) &
      bind(C, name="hipdecompMalloc") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      type(c_ptr) :: buffer
      integer(c_size_t), value :: buffer_size_bytes
      integer(c_int) :: res
    end function hipdecompMallocC
  end interface

  interface hipdecompMalloc
    module procedure hipdecompMallocR4, hipdecompMallocR8, hipdecompMallocC4, hipdecompMallocC8
  end interface hipdecompMalloc

  interface
    function hipdecompFreeC(handle, grid_desc, buffer) &
      bind(C, name="hipdecompFree") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      type(c_ptr), value :: buffer
      integer(c_int) :: res
    end function hipdecompFreeC
  end interface

  interface hipdecompFree
    module procedure hipdecompFreeR4, hipdecompFreeR8, hipdecompFreeC4, hipdecompFreeC8
  end interface hipdecompFree

  ! Convenience functions
  interface
    function hipdecompTransposeCommBackendToStringC(comm_backend) &
      bind(C, name="hipdecompTransposeCommBackendToString") result(res)
      import
      integer(c_int), value :: comm_backend
      type(c_ptr) :: res
    end function hipdecompTransposeCommBackendToStringC
  end interface

  interface
    function hipdecompHaloCommBackendToStringC(comm_backend) &
      bind(C, name="hipdecompHaloCommBackendToString") result(res)
      import
      integer(c_int), value :: comm_backend
      type(c_ptr) :: res
    end function hipdecompHaloCommBackendToStringC
  end interface

  interface
    function hipdecompGetDataTypeSize(dtype, dtype_size) bind(C, name="hipdecompGetDataTypeSize") result(res)
      import
      integer(c_int), value :: dtype
      integer(c_int64_t) :: dtype_size
      integer(c_int) :: res
    end function hipdecompGetDataTypeSize
  end interface

  interface
    function hipdecompGetShiftedRankC(handle, grid_desc, axis, dim, displacement, periodic, shifted_rank) &
      bind(C, name="hipdecompGetShiftedRank") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      integer(c_int32_t), value :: axis, dim, displacement
      logical(c_bool), value :: periodic
      integer(c_int32_t) :: shifted_rank
      integer(c_int) :: res
    end function hipdecompGetShiftedRankC
  end interface

  ! Transpose functions
  interface
    function hipdecompTransposeXToY_C(handle, grid_desc, input, output, work, dtype, &
                                     input_halo_extents, output_halo_extents, input_padding, &
                                     output_padding, stream) &
      bind(C, name="hipdecompTransposeXToY") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, output, work
      real(c_float) :: input(*), output(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: input_halo_extents(3), output_halo_extents(3)
      integer(c_int32_t) :: input_padding(3), output_padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompTransposeXToY_C
  end interface

  interface
    function hipdecompTransposeYToZ_C(handle, grid_desc, input, output, work, dtype, &
                                     input_halo_extents, output_halo_extents, input_padding, &
                                     output_padding, stream) &
      bind(C, name="hipdecompTransposeYToZ") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, output, work
      real(c_float) :: input(*), output(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: input_halo_extents(3), output_halo_extents(3)
      integer(c_int32_t) :: input_padding(3), output_padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompTransposeYToZ_C
  end interface

  interface
    function hipdecompTransposeZToY_C(handle, grid_desc, input, output, work, dtype, &
                                     input_halo_extents, output_halo_extents, input_padding, &
                                     output_padding, stream) &
       bind(C, name="hipdecompTransposeZToY") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, output, work
      real(c_float) :: input(*), output(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: input_halo_extents(3), output_halo_extents(3)
      integer(c_int32_t) :: input_padding(3), output_padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompTransposeZToY_C
  end interface

  interface
    function hipdecompTransposeYToX_C(handle, grid_desc, input, output, work, dtype, &
                                     input_halo_extents, output_halo_extents, input_padding, &
                                     output_padding, stream) &
      bind(C, name="hipdecompTransposeYToX") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, output, work
      real(c_float) :: input(*), output(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: input_halo_extents(3), output_halo_extents(3)
      integer(c_int32_t) :: input_padding(3), output_padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompTransposeYToX_C
  end interface

  ! Halo functions
  interface
    function hipdecompUpdateHalosX_C(handle, grid_desc, input, work, dtype, &
                                    halo_extents, halo_periods, dim, padding, stream) &
      bind(C, name="hipdecompUpdateHalosX") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, work
      real(c_float) :: input(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: halo_extents(3)
      logical(c_bool) :: halo_periods(3)
      integer(c_int32_t), value :: dim
      integer(c_int32_t) :: padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompUpdateHalosX_C
  end interface

  interface
    function hipdecompUpdateHalosY_C(handle, grid_desc, input, work, dtype, &
                                    halo_extents, halo_periods, dim, padding, stream) &
      bind(C, name="hipdecompUpdateHalosY") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, work
      real(c_float) :: input(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: halo_extents(3)
      logical(c_bool) :: halo_periods(3)
      integer(c_int32_t), value :: dim
      integer(c_int32_t) :: padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompUpdateHalosY_C
  end interface

  interface
    function hipdecompUpdateHalosZ_C(handle, grid_desc, input, work, dtype, &
                                    halo_extents, halo_periods, dim, padding, stream) &
      bind(C, name="hipdecompUpdateHalosZ") result(res)
      import
      type(hipdecompHandle), value :: handle
      type(hipdecompGridDesc), value :: grid_desc
      !dir$ ignore_tkr input, work
      real(c_float) :: input(*), work(*)
      integer(c_int), value :: dtype
      integer(c_int32_t) :: halo_extents(3)
      logical(c_bool) :: halo_periods(3)
      integer(c_int32_t), value :: dim
      integer(c_int32_t) :: padding(3)
      integer(c_intptr_t), value :: stream
      integer(c_int) :: res
    end function hipdecompUpdateHalosZ_C
  end interface

  ! Internal interface to strlen
  interface
    function strlen(str) &
      bind(C, name="strlen") result(size)
        import
        type(c_ptr), value :: str
        integer(c_int) :: size
    end function
  end interface

contains

  ! Fortran native functions/subroutines

  ! hipDecomp initialization/finalization functions
  function hipdecompInit_MPI_F(handle, comm) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    integer :: comm
    integer(c_int) :: res

    res = hipdecompInit_FC(handle, comm)
  end function hipdecompInit_MPI_F

  function hipdecompInit_MPI_F08(handle, comm) result(res)
    implicit none
    type, bind(c) :: MPI_Comm
      integer :: MPI_VAL
    end type MPI_Comm
    type(hipdecompHandle) :: handle
    type(MPI_Comm) :: comm
    integer(c_int) :: res

    res = hipdecompInit_FC(handle, comm%MPI_VAL)
  end function hipdecompInit_MPI_F08

  ! hipdecompGridDesc creation/manipulation functions
  function hipdecompGridDescAutotuneOptionsSetDefaults(options) result(res)
    type(hipdecompGridDescAutotuneOptions) :: options
    integer(c_int) :: res

    res = hipdecompGridDescAutotuneOptionsSetDefaultsC(options)

    ! Adjust halo axis entry for one-based axis indexing
    options%halo_axis = options%halo_axis + 1

  end function hipdecompGridDescAutotuneOptionsSetDefaults

  function hipdecompGridDescCreate(handle, grid_desc, config, options) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    type(hipdecompGridDescConfig) :: config
    type(hipdecompGridDescAutotuneOptions), optional :: options
    integer(c_int) :: res

    ! Adjust transpose mem order entries for zero-based indexing
    config%transpose_mem_order = config%transpose_mem_order - 1

    if (present(options)) then
      ! Adjust halo axis entry for zero-based axis indexing
      options%halo_axis = options%halo_axis - 1
      res = hipdecompGridDescCreateC(handle, grid_desc, config, options)
      ! Adjust halo axis entry for one-based axis indexing
      options%halo_axis = options%halo_axis + 1
    else
      res = hipdecompGridDescCreateC_nullopt(handle, grid_desc, config, C_NULL_PTR)
    endif

    ! Adjust transpose mem order entries for one-based indexing
    config%transpose_mem_order = config%transpose_mem_order + 1
  end function hipdecompGridDescCreate

  ! General functions
  function hipdecompGetPencilInfo(handle, grid_desc, pencil_info, axis, halo_extents, padding) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    integer :: axis  ! unit offset, so x/y/z = 1/2/3
    integer, optional:: halo_extents(3)
    integer, optional:: padding(3)
    type(hipdecompPencilInfo) :: pencil_info  ! res%order is unit offset, x/y/z = 1/2/3
    integer(c_int) :: res

    integer :: halo_extents_(3)
    integer :: padding_(3)

    halo_extents_(:) = [0, 0, 0]
    padding_(:) = [0, 0, 0]
    if (present(halo_extents)) halo_extents_ = halo_extents
    if (present(padding)) padding_ = padding
    res = hipdecompGetPencilInfoC(handle, grid_desc, pencil_info, axis - 1, halo_extents_, padding_)
    ! Update entries for Fortran indexing
    pencil_info%order = pencil_info%order + 1
    pencil_info%lo = pencil_info%lo + 1
    pencil_info%hi = pencil_info%hi + 1
  end function hipdecompGetPencilInfo

  function hipdecompGetGridDescConfig(handle, grid_desc, config) result(res)
    type(hipdecompHandle), value :: handle
    type(hipdecompGridDesc), value :: grid_desc
    type(hipdecompGridDescConfig) :: config
    integer(c_int) :: res

    res = hipdecompGetGridDescConfigC(handle, grid_desc, config)

    ! Adjust transpose mem order entries for one-based indexing
    config%transpose_mem_order = config%transpose_mem_order + 1

  end function hipdecompGetGridDescConfig

  function hipdecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents, workspace_size) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    integer :: axis
    integer :: halo_extents(3)
    integer(int64) :: workspace_size
    integer(c_int) :: res

    res = hipdecompGetHaloWorkspaceSizeC(handle, grid_desc, axis - 1, halo_extents, workspace_size)
  end function hipdecompGetHaloWorkspaceSize

  function hipdecompMallocR4(handle, grid_desc, buffer, buffer_size) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    real(real32), pointer, contiguous :: buffer(:)
    integer(int64) :: buffer_size
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    res = hipdecompMallocC(handle, grid_desc, buffer_c, buffer_size * 4)
    call c_f_pointer(buffer_c, buffer, [buffer_size])
  end function hipdecompMallocR4

  function hipdecompMallocR8(handle, grid_desc, buffer, buffer_size) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    real(real64), pointer, contiguous :: buffer(:)
    integer(int64) :: buffer_size
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    res = hipdecompMallocC(handle, grid_desc, buffer_c, buffer_size * 8)
    call c_f_pointer(buffer_c, buffer, [buffer_size])
  end function hipdecompMallocR8

  function hipdecompMallocC4(handle, grid_desc, buffer, buffer_size) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    complex(real32), pointer, contiguous :: buffer(:)
    integer(int64) :: buffer_size
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    res = hipdecompMallocC(handle, grid_desc, buffer_c, buffer_size * 8)
    call c_f_pointer(buffer_c, buffer, [buffer_size])
  end function hipdecompMallocC4

  function hipdecompMallocC8(handle, grid_desc, buffer, buffer_size) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    complex(real64), pointer, contiguous :: buffer(:)
    integer(int64) :: buffer_size
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    res = hipdecompMallocC(handle, grid_desc, buffer_c, buffer_size * 16)
    call c_f_pointer(buffer_c, buffer, [buffer_size])
  end function hipdecompMallocC8

  function hipdecompFreeR4(handle, grid_desc, buffer) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    real(real32), pointer, contiguous :: buffer(:)
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    buffer_c = c_loc(buffer)
    res = hipdecompFreeC(handle, grid_desc, buffer_c)
  end function hipdecompFreeR4

  function hipdecompFreeR8(handle, grid_desc, buffer) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    real(real64), pointer, contiguous :: buffer(:)
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    buffer_c = c_loc(buffer)
    res = hipdecompFreeC(handle, grid_desc, buffer_c)
  end function hipdecompFreeR8

  function hipdecompFreeC4(handle, grid_desc, buffer) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    complex(real32), pointer, contiguous :: buffer(:)
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    buffer_c = c_loc(buffer)
    res = hipdecompFreeC(handle, grid_desc, buffer_c)
  end function hipdecompFreeC4

  function hipdecompFreeC8(handle, grid_desc, buffer) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    complex(real64), pointer, contiguous :: buffer(:)
    integer(c_int) :: res

    type(c_ptr) :: buffer_c

    buffer_c = c_loc(buffer)
    res = hipdecompFreeC(handle, grid_desc, buffer_c)
  end function hipdecompFreeC8

  ! Convenience functions
  function hipdecompTransposeCommBackendToString(comm_backend) result(res)
    implicit none
    integer(c_int) :: comm_backend
    character(len=:), allocatable :: res
    type(c_ptr) :: cstr

    cstr = hipdecompTransposeCommBackendToStringC(comm_backend)
    call __hipdecomp_copy_c_string(cstr, res)
  end function hipdecompTransposeCommBackendToString

  function hipdecompHaloCommBackendToString(comm_backend) result(res)
    implicit none
    integer(c_int) :: comm_backend
    character(len=:), allocatable :: res
    type(c_ptr) :: cstr
    integer(c_int) :: csize

    cstr = hipdecompHaloCommBackendToStringC(comm_backend)
    call __hipdecomp_copy_c_string(cstr, res)
  end function hipdecompHaloCommBackendToString

  function hipdecompGetShiftedRank(handle, grid_desc, axis, dim, displacement, periodic, shifted_rank) &
    result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    integer :: axis, dim, displacement
    logical :: periodic
    integer(c_int32_t) :: shifted_rank
    integer(c_int) :: res
    logical(c_bool) :: periodic_c

    periodic_c = periodic
    res = hipdecompGetShiftedRankC(handle, grid_desc, axis - 1, dim - 1, displacement, periodic_c, shifted_rank)
  end function hipdecompGetShiftedRank

  ! Transpose functions
  function hipdecompTransposeXToY(handle, grid_desc, &
       input, output, work, dtype, input_halo_extents, output_halo_extents, &
       input_padding, output_padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, output, work
    real(c_float) :: input(*), output(*), work(*)
    integer :: dtype
    integer(c_intptr_t), optional :: stream
    integer, optional :: input_halo_extents(3)
    integer, optional :: output_halo_extents(3)
    integer, optional :: input_padding(3)
    integer, optional :: output_padding(3)
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    integer :: input_halo_extents_(3)
    integer :: output_halo_extents_(3)
    integer :: input_padding_(3)
    integer :: output_padding_(3)

    stream_ = 0
    input_halo_extents_(:) = [0, 0, 0]
    output_halo_extents_(:) = [0, 0, 0]
    input_padding_(:) = [0, 0, 0]
    output_padding_(:) = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(input_halo_extents)) input_halo_extents_ = input_halo_extents
    if (present(output_halo_extents)) output_halo_extents_ = output_halo_extents
    if (present(input_padding)) input_padding_ = input_padding
    if (present(output_padding)) output_padding_ = output_padding
    res = hipdecompTransposeXToY_C(handle, grid_desc, &
          input, output, work, dtype, input_halo_extents_, output_halo_extents_, &
          input_padding_, output_padding_, stream_)
  end function hipdecompTransposeXToY

  function hipdecompTransposeYToZ(handle, grid_desc, &
       input, output, work, dtype, input_halo_extents, output_halo_extents, &
       input_padding, output_padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, output, work
    real(c_float) :: input(*), output(*), work(*)
    integer :: dtype
    integer(c_intptr_t), optional :: stream
    integer, optional :: input_halo_extents(3)
    integer, optional :: output_halo_extents(3)
    integer, optional :: input_padding(3)
    integer, optional :: output_padding(3)
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    integer :: input_halo_extents_(3)
    integer :: output_halo_extents_(3)
    integer :: input_padding_(3)
    integer :: output_padding_(3)

    stream_ = 0
    input_halo_extents_(:) = [0, 0, 0]
    output_halo_extents_(:) = [0, 0, 0]
    input_padding_(:) = [0, 0, 0]
    output_padding_(:) = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(input_halo_extents)) input_halo_extents_ = input_halo_extents
    if (present(output_halo_extents)) output_halo_extents_ = output_halo_extents
    if (present(input_padding)) input_padding_ = input_padding
    if (present(output_padding)) output_padding_ = output_padding
    res = hipdecompTransposeYToZ_C(handle, grid_desc, &
          input, output, work, dtype, input_halo_extents_, output_halo_extents_, &
          input_padding_, output_padding_, stream_)
  end function hipdecompTransposeYToZ

  function hipdecompTransposeZToY(handle, grid_desc, &
       input, output, work, dtype, input_halo_extents, output_halo_extents, &
       input_padding, output_padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, output, work
    real(c_float) :: input(*), output(*), work(*)
    integer :: dtype
    integer(c_intptr_t), optional :: stream
    integer, optional :: input_halo_extents(3)
    integer, optional :: output_halo_extents(3)
    integer, optional :: input_padding(3)
    integer, optional :: output_padding(3)
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    integer :: input_halo_extents_(3)
    integer :: output_halo_extents_(3)
    integer :: input_padding_(3)
    integer :: output_padding_(3)

    stream_ = 0
    input_halo_extents_(:) = [0, 0, 0]
    output_halo_extents_(:) = [0, 0, 0]
    input_padding_(:) = [0, 0, 0]
    output_padding_(:) = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(input_halo_extents)) input_halo_extents_ = input_halo_extents
    if (present(output_halo_extents)) output_halo_extents_ = output_halo_extents
    if (present(input_padding)) input_padding_ = input_padding
    if (present(output_padding)) output_padding_ = output_padding
    res = hipdecompTransposeZToY_C(handle, grid_desc, &
          input, output, work, dtype, input_halo_extents_, output_halo_extents_, &
          input_padding_, output_padding_, stream_)
  end function hipdecompTransposeZToY

  function hipdecompTransposeYToX(handle, grid_desc, &
       input, output, work, dtype, input_halo_extents, output_halo_extents, &
       input_padding, output_padding ,stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, output, work
    real(c_float) :: input(*), output(*), work(*)
    integer :: dtype
    integer(c_intptr_t), optional :: stream
    integer, optional :: input_halo_extents(3)
    integer, optional :: output_halo_extents(3)
    integer, optional :: input_padding(3)
    integer, optional :: output_padding(3)
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    integer :: input_halo_extents_(3)
    integer :: output_halo_extents_(3)
    integer :: input_padding_(3)
    integer :: output_padding_(3)

    stream_ = 0
    input_halo_extents_(:) = [0, 0, 0]
    output_halo_extents_(:) = [0, 0, 0]
    input_padding_(:) = [0, 0, 0]
    output_padding_(:) = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(input_halo_extents)) input_halo_extents_ = input_halo_extents
    if (present(output_halo_extents)) output_halo_extents_ = output_halo_extents
    if (present(input_padding)) input_padding_ = input_padding
    if (present(output_padding)) output_padding_ = output_padding
    res = hipdecompTransposeYToX_C(handle, grid_desc, &
          input, output, work, dtype, input_halo_extents_, output_halo_extents_, &
          input_padding_, output_padding_, stream_)
  end function hipdecompTransposeYToX

  ! Halo functions
  function hipdecompUpdateHalosX(handle, grid_desc, &
       input, work, dtype, halo_extents, halo_periods, &
       dim, padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, work
    real(c_float) :: input(*), work(*)
    integer :: dtype
    integer :: halo_extents(3)
    logical :: halo_periods(3)
    integer :: dim
    integer, optional :: padding(3)
    integer(c_intptr_t), optional :: stream
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    logical(c_bool) :: halo_periods_c(3)
    integer :: padding_(3)

    halo_periods_c(:) = halo_periods

    stream_ = 0
    padding_ = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(padding)) padding_ = padding
    res = hipdecompUpdateHalosX_C(handle, grid_desc, &
          input, work, dtype, halo_extents, halo_periods_c, &
          dim - 1, padding_, stream_)
  end function hipdecompUpdateHalosX

  function hipdecompUpdateHalosY(handle, grid_desc, &
       input, work, dtype, halo_extents, halo_periods, &
       dim, padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, work
    real(c_float) :: input(*), work(*)
    integer :: dtype
    integer :: halo_extents(3)
    logical :: halo_periods(3)
    integer :: dim
    integer, optional :: padding(3)
    integer(c_intptr_t), optional :: stream
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    logical(c_bool) :: halo_periods_c(3)
    integer :: padding_(3)

    halo_periods_c(:) = halo_periods

    stream_ = 0
    padding_ = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(padding)) padding_ = padding
    res = hipdecompUpdateHalosY_C(handle, grid_desc, &
          input, work, dtype, halo_extents, halo_periods_c, &
          dim - 1, padding_, stream_)
  end function hipdecompUpdateHalosY

  function hipdecompUpdateHalosZ(handle, grid_desc, &
       input, work, dtype, halo_extents, halo_periods, &
       dim, padding, stream) result(res)
    implicit none
    type(hipdecompHandle) :: handle
    type(hipdecompGridDesc) :: grid_desc
    !dir$ ignore_tkr input, work
    real(c_float) :: input(*), work(*)
    integer :: dtype
    integer :: halo_extents(3)
    logical :: halo_periods(3)
    integer :: dim
    integer, optional :: padding(3)
    integer(c_intptr_t), optional :: stream
    integer(c_int) :: res

    integer(c_intptr_t) :: stream_
    logical(c_bool) :: halo_periods_c(3)
    integer :: padding_(3)

    halo_periods_c(:) = halo_periods

    stream_ = 0
    padding_ = [0, 0, 0]
    if (present(stream)) stream_ = stream
    if (present(padding)) padding_ = padding
    res = hipdecompUpdateHalosZ_C(handle, grid_desc, &
          input, work, dtype, halo_extents, halo_periods_c, &
          dim - 1, padding, stream_)
  end function hipdecompUpdateHalosZ

  ! Helper function to copy string
  subroutine __hipdecomp_copy_c_string(cstr, fstr)
    implicit none
    type(c_ptr) :: cstr
    character(len=:), allocatable :: fstr
    integer(c_int) :: csize

    if (c_associated(cstr)) then
      csize = strlen(cstr)
      block
        character(kind=c_char, len=csize + 1), pointer :: p
        call c_f_pointer(cstr, p)
        fstr = p(1:csize)
        nullify(p)
      end block
    else
      fstr = ' '
    endif
  end subroutine __hipdecomp_copy_c_string

end module hipdecomp
