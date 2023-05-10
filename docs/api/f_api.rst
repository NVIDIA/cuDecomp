####################
cuDecomp Fortran API
####################

These are all the types and functions available in the cuDecomp Fortran API.

Types
======================

Internal types
--------------------------------

.. _cudecompHandle_t-f-ref:

cudecompHandle
________________

.. f:type:: cudecompHandle

    A cuDecomp internal handle structure.

------

.. _cudecompGridDesc_t-f-ref:

cudecompGridDesc
__________________

.. f:type:: cudecompGridDesc

    A cuDecomp internal grid descriptor structure.

------

Grid Descriptor Configuration
-----------------------------

.. _cudecompGridDescConfig_t-f-ref:

cudecompGridDescConfig
________________________

.. f:type:: cudecompGridDescConfig

  A data structure defining configuration options for grid descriptor creation.

  :f integer gdims(3): dimensions of global data grid
  :f integer gdims_dist(3): dimensions of global data grid to use for distribution
  :f integer pdims(2): dimensions of process grid
  :f cudecompTransposeCommType transpose_comm_backend: communication backend to use for transpose communication (default: CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
  :f logical transpose_axis_contiguous(3): flag (by axis) indicating if memory should be contiguous along pencil axis (default: [false, false, false])
  :f cudecompHaloCommType halo_comm_backend: communication backend to use for halo communication (default: CUDECOMP_HALO_COMM_MPI)

------

.. _cudecompGridDescAutotuneOptions_t-f-ref:

cudecompGridDescAutotuneOptions
_________________________________

.. f:type:: cudecompGridDescAutotuneOptions

  A data structure defining autotuning options for grid descriptor creation.

  :f cudecompAutotuneGridMode grid_mode: which communication (transpose/halo) to use to autotune process grid (default: CUDECOMP_AUTOTUNE_GRID_TRANSPOSE)
  :f cudecompDataType dtype: datatype to use during autotuning (default: CUDECOMP_DOUBLE)
  :f logical allow_uneven_distributions: flag to control whether autotuning allows process grids that result in uneven distributions of elements across processes (default: true)
  :f logical disable_nccl_backends: flag to disable NCCL backend options during autotuning (default: false)
  :f logical disable_nvshmem_backends: flag to disable NVSHMEM backend options during autotuning (default: false)
  :f logical autotune_transpose_backend: flag to enable transpose backend autotuning (default: false)
  :f logical autotune_halo_backend: flag to enable halo backend autotuning (default: false)
  :f logical transpose_use_inplace_buffers: flag to control whether transpose autotuning uses in-place or out-of-place buffers (default: false)
  :f integer halo_extents(3): extents for halo autotuning (default: [0, 0, 0])
  :f logical halo_periods(3): periodicity for halo autotuning (default: [false, false, false])
  :f integer halo_axis: which axis pencils to use for halo autotuning (default: 1, X-pencils)

------

Pencil Information
-----------------------------

.. _cudecompPencilInfo_t-f-ref:

cudecompPencilInfo
____________________

.. f:type:: cudecompPencilInfo

  A data structure containing geometry information about a pencil data buffer.

  :f integer shape(3): pencil shape (in local order, including halo elements)
  :f integer lo(3): lower bound coordinates (in local order, excluding halo elements)
  :f integer hi(3): upper bound coordinates (in local order, excluding halo elements)
  :f integer order(3): data layout order (e.g. 3,2,1 means memory is ordered Z,Y,X)
  :f integer halo_extents(3): halo extents by dimension (in global order)
  :f int64 size: number of elements in pencil (including halo elements)

Communication Backends
---------------------------------

.. _cudecompTransposeCommBackend_t-f-ref:

cudecompTranposeCommBackend
_____________________________
See documention for equivalent C enumerator, :ref:`cudecompTransposeCommBackend_t-ref`.

------

.. _cudecompHaloCommBackend_t-f-ref:

cudecompHaloCommBackend
_________________________
See documention for equivalent C enumerator, :ref:`cudecompHaloCommBackend_t-ref`.

------

Additional Enumerators
---------------------------------

.. _cudecompDataType_t-f-ref:

cudecompDataType
__________________
See documention for equivalent C enumerator, :ref:`cudecompDataType_t-ref`.

------

.. _cudecompAutotuneGridMode_t-f-ref:

cudecompAutotuneGridMode
__________________________
See documention for equivalent C enumerator, :ref:`cudecompAutotuneGridMode_t-ref`.

------

.. _cudecompResult_t-f-ref:

cudecompResult
________________
See documention for equivalent C enumerator, :ref:`cudecompResult_t-ref`.

Functions
==========================

Library Initialization/Finalization
-----------------------------------

.. _cudecompInit-f-ref:

cudecompInit
____________

.. f:function:: cudecompInit(handle, mpi_comm)

  Initializes the cuDecomp library from an existing MPI communicator.

  :p cudecompHandle handle [out]: An uninitilzied cuDecomp library handle.
  :p MPI_Comm mpi_comm [in]: MPI communicator containing ranks to use with cuDecomp.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompFinalize-f-ref:

cudecompFinalize
________________

.. f:function:: cudecompFinalize(handle)

  Finalizes the cuDecomp library and frees associated resources.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

Grid Descriptor Management
-----------------------------------

.. _cudecompGridDescCreate-f-ref:

cudecompGridDescCreate
______________________

.. f:function:: cudecompGridDescCreate(handle, grid_desc, config [, options])

  Creates a cuDecomp grid descriptor for use with cuDecomp functions.

  This function creates a grid descriptor that cuDecomp requires for most library operations that perform communication or query decomposition information. This grid descriptor contains information about how the global data grid is distributed and other internal resources to facilitate communication.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [out]: An uninitalized cuDecomp grid descriptor.
  :p cudecompGridDescConfig config [inout]: A populated cuDecomp grid descriptor configuration structure. This structure defines the required attributes of the decomposition. On successful exit, fields in this structure may be updated to reflect autotuning results.
  :p cudecompGridDescAutotuneOptions [in,optional]: A populated cuDeomp grid descriptor autotune options structure. This options structure is used to control the behavior of the process grid and communication backend autotuning.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGridDescDestroy-f-ref:

cudecompGridDescDestroy
_______________________

.. f:function:: cudecompGridDescDestroy(handle, grid_desc)

  Destroys a cuDecomp grid descriptor and frees associated resources.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGridDescConfigSetDefaults-f-ref:

cudecompGridDescConfigSetDefaults
_________________________________

.. f:function:: cudecompGridDescConfigSetDefaults(config)

  Initializes a cuDecomp grid descriptor configuration structure with default values.

  This function initializes entries in a cuDecomp grid descriptor configuration structure to default values.

  :p cudecompGridDescConfig config [out]: A cuDecomp grid descriptor configuration structure.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGridDescAutotuneOptionsSetDefaults-f-ref:

cudecompGridDescAutotuneOptionsSetDefaults
__________________________________________

.. f:function:: cudecompGridDescAutotuneOptionsSetDefaults(options)

  Initializes a cuDecomp grid descriptor autotune options structure with default values.

  This function initializes entries in a cuDecomp grid descriptor autotune options structure to default values.

  :p cudecompGridDescAutotuneOptions options [out]: A cuDecomp grid descriptor autotune options structure.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

Workspace Management
----------------------------------------

.. _cudecompGetTransposeWorkspaceSize-f-ref:

cudecompGetTransposeWorkspaceSize
_________________________________
.. f:function:: cudecompGetTransposeWorkspaceSize(handle, grid_desc, workspace_size)

  Queries the required transpose workspace size, in elements, for a provided grid descriptor.

  This function queries the required workspace size, in elements, for transposition communication using a provided grid descriptor. This workspace is required to faciliate local transposition/packing/unpacking operations, or for use as a staging buffer.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p int64 workspace_size [out]: the required workspace size.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGetHaloWorkspaceSize-f-ref:

cudecompGetHaloWorkspaceSize
____________________________
.. f:function:: cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, halo_extents, workspace_size)

  Queries the required transpose workspace size, in elements, for a provided grid descriptor.

  This function queries the required workspace size, in elements, for transposition communication using a provided grid descriptor. This workspace is required to faciliate local transposition/packing/unpacking operations, or for use as a staging buffer.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p integer axis [in]: The domain axis the desired pencil is aligned with.
  :p integer halo_extents(3) [in]: An array of three integers to define halo region extents of the pencil, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side).
  :p int64 workspace_size [out]: the required workspace size.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGetDataTypeSize-f-ref:

cudecompGetDataTypeSize
_______________________
.. f:function:: cudecompGetDataTypeSize(dtype, dtype_size)

  Function to get size (in bytes) of a cuDecomp data type.

  :p cudecompDataType dtype [in]: A cudecompDataType value.
  :p int64 dtype_size [out]: the data type size in bytes.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompMalloc-f-ref:

cudecompMalloc
______________

.. f:function:: cudecompMalloc(handle, grid_desc, buffer, buffer_size)

  Allocation function for cuDecomp workspaces.

  This function should be used to allocate cuDecomp workspaces. It will select an appropriate allocator based on the communication backend information found in the provided grid descriptor. At the current time, only NVSHMEM-enabled backends require a special allocation (using nvshmem_malloc). This function is collective and should be called on all workers to avoid deadlocks. Additionally, any memory allocated using this function is invalidated if the provided grid descriptor is destroyed and care are should be taken free memory allocated using this function before the provided grid descriptor is destroyed.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T buffer(*) [out]: A Fortran pointer to device memory of type :code:`T`, where :code:`T` is one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`.
  :p int64 buffer_size [in]: size of requested allocation, in number of elements of type :code:`T`.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompFree-f-ref:

cudecompFree
____________

.. f:function:: cudecompFree(handle, grid_desc, buffer)

  Deallocation function for cuDecomp workspaces.

  This function should be used to deallocate memory allocate with :code:`cudecompMalloc`. It will select an appropriate deallocation function based on the communication backend information found in the provided grid descriptor. At the current time, only NVSHMEM-enabled backends require a special deallocation (using nvshmem_free). This function is collective and should be called on all workers to avoid deadlocks.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T buffer(*) [out]: A Fortran pointer to device memory of type :code:`T`, where :code:`T` is one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`, pointing to memory allocated with :code:`cudecompMalloc`.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

Helper Functions
----------------

.. _cudecompGetPencilInfo-f-ref:

cudecompGetPencilInfo
_____________________

.. f:function:: cudecompGetPencilInfo(handle, grid_desc, pencil_info, axis[, halo_extents])

  Collects geometry information about assigned pencils, by domain axis.

  This function queries information about the pencil assigned to the calling worker for the given axis. This information is collected in a cuDecomp pencil information structure, which can be used to access and manipuate data within the user-allocated memory buffer.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p cudecompPencilInfo pencil_info [out]: A cuDecomp pencil information structure.
  :p integer axis [in]: The domain axis the desired pencil is aligned with.
  :p integer halo_extents(3) [in, optional]: An array of three integers to define halo region extents of the pencil, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side).
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompTransposeCommBackendToString-f-ref:

cudecompTranposeCommBackendToString
___________________________________

.. f:function:: cudecompTransposeCommBackendToString(comm_backend)

  Function to get string name of transpose communication backend.

  :p cudecompTransposeCommBackend comm_backend [in]: A cuDecompTranposeCommBackend value.
  :r character(:) res: A string representation of the transpose communication backend. Will return string “ERROR” if invalid backend value is provided.

------

.. _cudecompHaloCommBackendToString-f-ref:

cudecompHaloCommBackendToString
_______________________________

.. f:function:: cudecompHaloCommBackendToString(comm_backend)

  Function to get string name of transpose communication backend.

  :p cudecompHaloCommBackend comm_backend [in]: A cuDecompHaloCommBackend value.
  :r character(:) res: A string representation of the halo communication backend. Will return string “ERROR” if invalid backend value is provided.

------

.. _cudecompGetGridDescConfig-f-ref:

cudecompGetGridDescConfig
_________________________

.. f:function:: cudecompGetGridDescConfig(handle, grid_desc, config)

  Queries the configuration used to create a grid descriptor.

  This function queries information about the pencil assigned to the calling worker for the given axis. This information is collected in a cuDecomp pencil information structure, which can be used to access and manipuate data within the user-allocated memory buffer.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p cudecompGridDescConfig config [out]: A cuDecomp grid descriptor configuration structure.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompGetShiftedRank-f-ref:

cudecompGetShiftedRank
______________________

.. f:function:: cudecompGetShiftedRank(handle, grid_desc, axis, dim, displacement, periodic, shifted_rank)

  Function to retrieve the global rank of neighboring processes.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p integer axis [in]: The domain axis the pencil is aligned with.
  :p integer dim [in]: Which pencil dimension (global indexed) to retrieve neighboring rank
  :p integer displacement [in]: Displacement of neighboring rank to retrieve. For example, 1 will retrieve the +1-th neighbor rank along dim, while -1 will retrieve the -1-th neighbor rank.
  :p logical periodic [in]: A boolean flag to indicate whether dim should be treated periodically
  :p integer shifted_rank [out]: The global rank of the requested neighbor. For non-periodic cases, a value of -1 will be written if the displacement results in a position outside the global domain.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

Transposition Functions
-----------------------

.. _cudecompTransposeXToY-f-ref:

cudecompTransposeXToY
_____________________

.. f:function:: cudecompTransposeXToY(handle, grid_desc, input, output, work, dtype[, input_halo_extents, output_halo_extents, stream])

  Function to transpose data from X-axis aligned pencils to a Y-axis aligned pencils.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in]: Device array containing input X-axis aligned pencil data.
  :p T output(*) [out]: Device array to write output Y-axis aligned pencil data. If :code:`input` and :code:`output` are the same, operation is performed in-place
  :p T work(*) [in]: Device array to use for transpose workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer input_halo_extents(3) [in,optional]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side). If not provided, input data is assumed to have no halos.
  :p integer output_halo_extents(3) [in,optional]: Similar to :code:`intput_halo_extents` but for the output data. If not provided, output data is assumed to have no halos.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompTransposeYToZ-f-ref:

cudecompTransposeYtoZ
_____________________

.. f:function:: cudecompTransposeYToZ(handle, grid_desc, input, output, work, dtype[, input_halo_extents, output_halo_extents, stream])

  Function to transpose data from Y-axis aligned pencils to a Z-axis aligned pencils.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in]: Device array containing input Y-axis aligned pencil data.
  :p T output(*) [out]: Device array to write output Z-axis aligned pencil data. If :code:`input` and :code:`output` are the same, operation is performed in-place
  :p T work(*) [in]: Device array to use for transpose workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer input_halo_extents(3) [in,optional]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side). If not provided, input data is assumed to have no halos.
  :p integer output_halo_extents(3) [in,optional]: Similar to :code:`intput_halo_extents` but for the output data. If not provided, output data is assumed to have no halos.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompTransposeZToY-f-ref:

cudecompTransposeZToY
_____________________

.. f:function:: cudecompTransposeZToY(handle, grid_desc, input, output, work, dtype[, input_halo_extents, output_halo_extents, stream])

  Function to transpose data from Z-axis aligned pencils to a Y-axis aligned pencils.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in]: Device array containing input Z-axis aligned pencil data.
  :p T output(*) [out]: Device array to write output Y-axis aligned pencil data. If :code:`input` and :code:`output` are the same, operation is performed in-place
  :p T work(*) [in]: Device array to use for transpose workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer input_halo_extents(3) [in,optional]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side). If not provided, input data is assumed to have no halos.
  :p integer output_halo_extents(3) [in,optional]: Similar to :code:`intput_halo_extents` but for the output data. If not provided, output data is assumed to have no halos.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.


------

.. _cudecompTransposeYToX-f-ref:

cudecompTransposeYToX
_____________________

.. f:function:: cudecompTransposeYToX(handle, grid_desc, input, output, work, dtype[, input_halo_extents, output_halo_extents, stream])

  Function to transpose data from Y-axis aligned pencils to a X-axis aligned pencils.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in]: Device array containing input Y-axis aligned pencil data.
  :p T output(*) [out]: Device array to write output X-axis aligned pencil data. If :code:`input` and :code:`output` are the same, operation is performed in-place
  :p T work(*) [in]: Device array to use for transpose workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer input_halo_extents(3) [in,optional]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side). If not provided, input data is assumed to have no halos.
  :p integer output_halo_extents(3) [in,optional]: Similar to :code:`intput_halo_extents` but for the output data. If not provided, output data is assumed to have no halos.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

Halo Exchange Functions
-----------------------

.. _cudecompUpdateHalosX-f-ref:

cudecompUpdateHalosX
____________________

.. f:function:: cudecompUpdateHalosX(handle, grid_desc, input, work, dtype, halo_extents, halo_periods[, stream])

  Function to perform halo communication of X-axis aligned pencil data.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in,out]: Device array containing input X-axis aligned pencil data. On successful completion, this buffer will contain the input X-axis aligned pencil data with the specified halo regions updated.
  :p T work(*) [in]: Device array to use for halo workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer halo_extents(3) [in]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side).
  :p logical halo_periods(3) [in]: An array of three boolean values to define halo periodicity of the input data, in global order. If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompUpdateHalosY-f-ref:

cudecompUpdateHalosY
____________________

.. f:function:: cudecompUpdateHalosY(handle, grid_desc, input, work, dtype, halo_extents, halo_periods[, stream])

  Function to perform halo communication of Y-axis aligned pencil data.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in,out]: Device array containing input Y-axis aligned pencil data. On successful completion, this buffer will contain the input X-axis aligned pencil data with the specified halo regions updated.
  :p T work(*) [in]: Device array to use for halo workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer halo_extents(3) [in]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side).
  :p logical halo_periods(3) [in]: An array of three boolean values to define halo periodicity of the input data, in global order. If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.

------

.. _cudecompUpdateHalosZ-f-ref:

cudecompUpdateHalosZ
____________________

.. f:function:: cudecompUpdateHalosZ(handle, grid_desc, input, work, dtype, halo_extents, halo_periods[, stream])

  Function to perform halo communication of Z-axis aligned pencil data.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`complex(real32)`, :code:`complex(real64)`. The data access for this operation is controlled via :code:`dtype`, irrespective of :code:`T`.

  :p cudecompHandle handle [in]: The initialized cuDecomp library handle
  :p cudecompGridDesc grid_desc [in]: A cuDecomp grid descriptor.
  :p T input(*) [in,out]: Device array containing input Z-axis aligned pencil data. On successful completion, this buffer will contain the input X-axis aligned pencil data with the specified halo regions updated.
  :p T work(*) [in]: Device array to use for halo workspace.
  :p cudecompDataType dtype [in]: The :code:`cudecompDataType` to use for the operation.
  :p integer halo_extents(3) [in]: An array of three integers to define halo region extents of the input data, in global order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements, one element on each side).
  :p logical halo_periods(3) [in]: An array of three boolean values to define halo periodicity of the input data, in global order. If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
  :p integer(cuda_stream_kind) stream [in, optional]: CUDA stream to enqueue GPU operations into. If not provided, operations are enqueued in the default stream.
  :r cudecompResult res: :code:`CUDECOMP_RESULT_SUCCESS` on success or error code on failure.
