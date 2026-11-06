###################
cuDecomp Python API
###################

.. py:currentmodule:: cudecomp

These are all the types and functions available in the cuDecomp Python API.
See :ref:`python-api-ref` for array, datatype, stream, and resource conventions.
Methods raise exceptions on failure instead of returning C API result codes.

Types
=====

Internal types
--------------

.. py:class:: Handle(mpi_comm)

   Collectively initialize cuDecomp from an existing MPI communicator.

   :param mpi_comm: Initialized ``mpi4py.MPI.Comm`` containing participating ranks.

   Select the CUDA device before construction. Keep the communicator valid
   until :py:meth:`finalize`.

.. py:class:: GridDesc

   Grid descriptor returned by :py:meth:`Handle.create_grid`; not directly
   constructible. Retains its Python handle until released. Library resources
   require explicit :py:meth:`destroy`.

Grid Descriptor Configuration
-----------------------------

.. py:class:: GridDescConfig()

   A data structure defining configuration options for grid descriptor creation.
   The constructor initializes its fields to defaults.
   Array properties are writable NumPy views; whole-field assignment
   accepts tuples, lists, or NumPy arrays. See :doc:`basic_usage` for examples
   and :ref:`cudecompGridDescConfig_t-ref` for configuration constraints.

   :ivar gdims: Dimensions of global data grid in X/Y/Z order, shape ``(3,)`` (default: all zeros).
   :ivar gdims_dist: Dimensions of global data grid to use for distribution,
      shape ``(3,)`` (default: all zeros, use ``gdims``).
   :ivar pdims: Dimensions of process grid, P_rows and P_cols, shape ``(2,)``
      (default: both zero, request autotuning).
   :ivar rank_order: Process grid rank assignment order
      (default: ``RankOrder.DEFAULT``, row-major).
   :ivar transpose_comm_backend: Communication backend to use for transpose communication
      (default: ``TransposeCommBackend.MPI_P2P``).
   :ivar transpose_axis_contiguous: Flags by axis indicating if memory should be
      contiguous along the pencil axis, shape ``(3,)`` (default: all ``False``).
   :ivar transpose_mem_order: Memory orders by pencil axis, shape ``(3, 3)``.
      Each row specifies a permutation of ``(0, 1, 2)`` from fastest to slowest
      and overrides ``transpose_axis_contiguous``. All rows must be set when used
      (default: all ``-1``, unset).
   :ivar halo_comm_backend: Communication backend to use for halo communication
      (default: ``HaloCommBackend.MPI``).

.. py:class:: GridDescAutotuneOptions()

   A data structure defining autotuning options for grid descriptor creation.
   The constructor initializes its fields to defaults.
   Array properties have the same assignment behavior as
   :py:class:`GridDescConfig`. See :doc:`autotuning` for usage and
   :ref:`cudecompGridDescAutotuneOptions_t-ref` for configuration constraints.

   :ivar n_warmup_trials: Number of warmup trials for each tested configuration during autotuning (default: ``3``).
   :ivar n_trials: Number of timed trials for each tested configuration during autotuning (default: ``5``).
   :ivar grid_mode: Communication type (transpose/halo) to use to autotune the process grid
      (default: ``AutotuneGridMode.TRANSPOSE``).
   :ivar dtype: Data type to use during autotuning; accepts a :py:class:`DataType`
      member (default: ``DataType.DOUBLE``).
   :ivar allow_uneven_decompositions: Allow process grids that result in uneven distributions of elements across processes
      (default: ``True``).
   :ivar disable_mpi_backends: Disable MPI backend options during autotuning (default: ``False``).
   :ivar disable_nccl_backends: Disable NCCL backend options during autotuning (default: ``False``).
   :ivar disable_nvshmem_backends: Disable NVSHMEM backend options during autotuning (default: ``False``).
   :ivar skip_threshold: Skip a configuration if ``skip_threshold * t > t_best``,
      where ``t`` is its first trial time and ``t_best`` the best average time
      (default: ``0.0``, disabled).
   :ivar autotune_transpose_backend: Enable transpose backend tuning (default: ``False``).
   :ivar transpose_use_inplace_buffers: Flags controlling in-place or out-of-place transpose buffers by operation,
      shape ``(4,)`` (default: all ``False``).
   :ivar transpose_op_weights: Multiplicative weights for trial time contributions by transpose operation, shape ``(4,)``
      (default: all ``1.0``).
   :ivar transpose_input_halo_extents: Input halo extent arguments to use during autotuning by transpose operation,
      shape ``(4, 3)`` (default: all zeros).
   :ivar transpose_output_halo_extents: Output halo extent arguments to use during autotuning by transpose operation,
      shape ``(4, 3)`` (default: all zeros).
   :ivar transpose_input_padding: Input padding arguments to use during autotuning by transpose operation,
      shape ``(4, 3)`` (default: all zeros).
   :ivar transpose_output_padding: Output padding arguments to use during autotuning by transpose operation,
      shape ``(4, 3)`` (default: all zeros).
   :ivar autotune_halo_backend: Enable halo backend tuning (default: ``False``).
   :ivar halo_extents: Symmetric halo extents in global X/Y/Z order,
      shape ``(3,)`` (default: all zeros).
   :ivar halo_periods: Periodicity in global X/Y/Z order, shape ``(3,)``
      (default: all ``False``).
   :ivar halo_axis: Pencil axis for halo tuning: 0=X, 1=Y, 2=Z (default: ``0``).
   :ivar halo_padding: Padding in global X/Y/Z order, shape ``(3,)``
      (default: all zeros).

   Transpose array entries use operation order X-to-Y, Y-to-Z, Z-to-Y, Y-to-X.
   Halo and padding rows use global X/Y/Z order.

Pencil Information
------------------

.. py:class:: PencilInfo

   A data structure containing geometry information about a pencil data buffer,
   returned by :py:meth:`GridDesc.get_pencil_info`.
   Properties are read-only. Array properties are tuples; shapes and bounds use
   Python C-order.
   See :ref:`python-pencil-layout-ref` for local/global indexing conventions.

   :ivar shape: Pencil shape in local C-order, including halos and padding.
   :ivar lo: Zero-based inclusive lower interior bounds in local order.
   :ivar hi: Zero-based inclusive upper interior bounds in local order.
   :ivar order: Global axis corresponding to each local array dimension.
   :ivar halo_extents: Symmetric halo extents in global X/Y/Z order.
   :ivar padding: Upper-end padding extents in global X/Y/Z order.
   :ivar size: Number of elements in the pencil, including halos and padding.

Communication Backends
----------------------

.. py:class:: TransposeCommBackend

   Members: ``MPI_P2P``, ``MPI_P2P_PL``, ``MPI_A2A``, ``NCCL``, ``NCCL_PL``,
   ``NVSHMEM``, ``NVSHMEM_PL``, ``NVSHMEM_SM``.
   See :ref:`cudecompTransposeCommBackend_t-ref` for backend behavior and
   restrictions. Enum availability does not imply the backend was built.

.. py:class:: HaloCommBackend

   Members: ``MPI``, ``MPI_BLOCKING``, ``NCCL``, ``NVSHMEM``,
   ``NVSHMEM_BLOCKING``. See :ref:`cudecompHaloCommBackend_t-ref`.

Additional Enumerators
----------------------

.. py:class:: DataType

   Members: ``FLOAT`` (float32), ``DOUBLE`` (float64), ``FLOAT_COMPLEX``
   (complex64), ``DOUBLE_COMPLEX`` (complex128).

.. py:class:: AutotuneGridMode

   Members: ``TRANSPOSE`` and ``HALO``. See :ref:`cudecompAutotuneGridMode_t-ref`.

.. py:class:: RankOrder

   Members: ``DEFAULT``, ``ROW_MAJOR``, ``COL_MAJOR``.
   See :ref:`cudecompRankOrder_t-ref`.

.. py:class:: Result

   C API result constants: ``SUCCESS``, ``INVALID_USAGE``, ``NOT_SUPPORTED``,
   ``INTERNAL_ERROR``, ``CUDA_ERROR``, ``CUTENSOR_ERROR``, ``MPI_ERROR``,
   ``NCCL_ERROR``, ``NVSHMEM_ERROR``, ``NVML_ERROR``. Public Python operations
   return objects or ``None`` and raise exceptions instead of returning these codes.

Functions
=========

Library Initialization/Finalization
-----------------------------------

Construct :py:class:`Handle` with an MPI communicator to initialize cuDecomp.

.. py:method:: Handle.finalize()

   Collectively finalize the handle and release its resources. Destroy
   grids before finalizing the handle. Repeated calls are no-ops.

   :returns: ``None``.

Grid Descriptor Management
--------------------------

.. py:method:: Handle.create_grid(config, autotune_options=None)

   Collectively create and return a :py:class:`GridDesc`.

   :param config: Initialized :py:class:`GridDescConfig`. Grid creation may
      update its properties with the selected configuration.
   :param autotune_options: :py:class:`GridDescAutotuneOptions`, or ``None``
      to disable autotuning (equivalent to passing ``NULL`` in C). If
      ``config.pdims`` is ``(0, 0)``, options are required; otherwise
      :py:exc:`InvalidUsageError` is raised.
   :returns: The new grid descriptor.

.. py:method:: GridDesc.destroy()

   Collectively destroy the descriptor. Repeated calls are no-ops; other methods
   on a destroyed grid raise :py:exc:`InvalidUsageError`.

   :returns: ``None``.

Helper Functions
----------------

.. py:method:: GridDesc.get_config()

   Return an independent :py:class:`GridDescConfig` containing the resolved
   configuration. Mutating this copy does not reconfigure the grid.

.. py:method:: GridDesc.get_pencil_info(axis, halo_extents=None, padding=None)

   Collect geometry information about the pencil assigned to the calling process
   for the given axis. Returns a :py:class:`PencilInfo` object.

   :param axis: Pencil orientation: 0=X, 1=Y, 2=Z.
   :param halo_extents: Three nonnegative symmetric extents in global X/Y/Z
      order. ``None`` means no halos.
   :param padding: Three nonnegative upper-end padding extents in global
      X/Y/Z order. ``None`` means no padding.

.. py:method:: GridDesc.get_shifted_rank(axis, dim, displacement, periodic)

   Return the neighbor's rank in the handle communicator, or ``-1`` if the
   neighbor does not exist at a nonperiodic boundary.

   :param axis: Pencil orientation: 0=X, 1=Y, 2=Z.
   :param dim: Global direction to shift in: 0=X, 1=Y, 2=Z.
   :param displacement: Signed integer neighbor displacement.
   :param periodic: Boolean selecting periodic boundaries in ``dim``.

.. py:function:: get_data_type_size(dtype)

   Get the size (in bytes) of a cuDecomp :py:class:`DataType`.

.. py:function:: transpose_comm_backend_to_string(backend)

   Return the display name of a :py:class:`TransposeCommBackend` member (for
   example ``"NCCL (pipelined)"``), not the enum member name.

.. py:function:: halo_comm_backend_to_string(backend)

   Return the display name of a :py:class:`HaloCommBackend` member, not the enum
   member name.

Transposition Methods
---------------------

All transpose methods are collective and use automatically managed workspaces.
Input and output arrays must be C-contiguous and match their respective pencil
query shapes, including halos and padding. These requirements also apply to
in-place operations.

.. py:method:: GridDesc.transpose_x_to_y(input, out=None, dtype=None, input_halo_extents=None, output_halo_extents=None, input_padding=None, output_padding=None, stream=None)

   Transpose data from X-axis aligned pencils to Y-axis aligned pencils.

   :param input: Input pencil array, as described in :ref:`python-api-ref`.
   :param out: Writable output pencil array. ``None`` allocates a CuPy array
      and requires a CuPy input.
   :param dtype: ``DataType`` member; ``None`` infers it
      from the input array. No conversion is performed.
   :param input_halo_extents: Three input halo extents in global order;
      ``None`` means no halos.
   :param output_halo_extents: Three output halo extents in global order;
      ``None`` means no halos.
   :param input_padding: Three input padding extents in global order;
      ``None`` means no padding.
   :param output_padding: Three output padding extents in global order;
      ``None`` means no padding.
   :param stream: CUDA stream handle, object implementing ``__cuda_stream__``,
      or object with integer ``ptr``. ``None`` uses CuPy's current stream for
      CuPy inputs and CUDA stream 0 otherwise. See :ref:`python-api-ref` for
      ordering requirements.
   :returns: The supplied ``out`` object or a new C-contiguous CuPy output.

   Only interior values are transposed; output halos and padding are not initialized.
   Operations are asynchronous with respect to the host.

.. py:method:: GridDesc.transpose_y_to_z(input, out=None, dtype=None, input_halo_extents=None, output_halo_extents=None, input_padding=None, output_padding=None, stream=None)

   Transpose data from Y-axis aligned pencils to Z-axis aligned pencils. Parameters and returns are as in
   :py:meth:`GridDesc.transpose_x_to_y`.

.. py:method:: GridDesc.transpose_z_to_y(input, out=None, dtype=None, input_halo_extents=None, output_halo_extents=None, input_padding=None, output_padding=None, stream=None)

   Transpose data from Z-axis aligned pencils to Y-axis aligned pencils. Parameters and returns are as in
   :py:meth:`GridDesc.transpose_x_to_y`.

.. py:method:: GridDesc.transpose_y_to_x(input, out=None, dtype=None, input_halo_extents=None, output_halo_extents=None, input_padding=None, output_padding=None, stream=None)

   Transpose data from Y-axis aligned pencils to X-axis aligned pencils. Parameters and returns are as in
   :py:meth:`GridDesc.transpose_x_to_y`.

Halo Exchange Methods
---------------------

All halo methods are collective, update arrays in place, and use automatically
managed workspaces. Input arrays must be writable, C-contiguous, and match the
corresponding pencil query shape, including halos and padding.

.. py:method:: GridDesc.update_halos_x(input, halo_extents, dim, dtype=None, halo_periods=None, padding=None, stream=None)

   Perform halo communication of X-axis aligned pencil data.

   :param input: Writable input pencil array matching ``get_pencil_info(0,
      halo_extents, padding)``.
   :param halo_extents: Three nonnegative symmetric extents in global X/Y/Z order.
   :param dim: Global update direction: 0=X, 1=Y, 2=Z.
   :param dtype: ``DataType`` member; ``None`` infers it
      from the array. No conversion is performed.
   :param halo_periods: Three booleans selecting periodic boundaries in global
      order. ``None`` means all nonperiodic.
   :param padding: Three padding extents in global order; ``None`` means none.
   :param stream: CUDA stream handle, object implementing ``__cuda_stream__``,
      or object with integer ``ptr``; ``None`` uses CuPy's current stream for
      CuPy inputs and CUDA stream 0 otherwise. See :ref:`python-api-ref` for
      ordering requirements.
   :returns: The same ``input`` object. Operations are asynchronous with respect
      to the host.

   Halos with no neighbor at a nonperiodic boundary are unchanged. Padding
   elements are not part of the halo update.

.. py:method:: GridDesc.update_halos_y(input, halo_extents, dim, dtype=None, halo_periods=None, padding=None, stream=None)

   Perform halo communication of Y-axis aligned pencil data. Parameters and returns are as in
   :py:meth:`GridDesc.update_halos_x`, with input matching the Y-pencil query.

.. py:method:: GridDesc.update_halos_z(input, halo_extents, dim, dtype=None, halo_periods=None, padding=None, stream=None)

   Perform halo communication of Z-axis aligned pencil data. Parameters and returns are as in
   :py:meth:`GridDesc.update_halos_x`, with input matching the Z-pencil query.

Errors and Version
==================

Arrays that do not meet cuDecomp's requirements (dtype, shape, contiguity,
writability, byte order, or masks) raise :py:exc:`InvalidUsageError`.
Arguments of the wrong kind and malformed CUDA array-interface metadata raise
``TypeError``, ``ValueError``, or ``OverflowError``; errors raised by array
providers may propagate as well. Missing required CUDA array-interface fields
raise ``KeyError``. With ``dtype=None``, an input whose dtype is not a supported
cuDecomp type raises ``TypeError``.
C API result codes are translated into the exceptions below.

Numeric arguments use pybind11's standard conversions, including corresponding
NumPy scalar types. Integer arguments do not accept floating-point values.
Fixed-size sequence arguments must have the required length;
argument conversion failures raise ``TypeError``.

.. py:exception:: CuDecompError

   Base class for cuDecomp errors; derives from ``RuntimeError``.

.. py:exception:: InvalidUsageError

   Invalid API usage, arrays that do not meet cuDecomp's requirements, or use
   of a finalized handle or destroyed grid. Derives from :py:exc:`CuDecompError`.

.. py:exception:: NotSupportedError

   Unsupported configuration. Derives from :py:exc:`CuDecompError`.

.. py:exception:: InternalError

   Internal library failure. Derives from :py:exc:`CuDecompError`.

.. py:exception:: CudaError
                  CutensorError
                  MpiError
                  NcclError
                  NvshmemError
                  NvmlError

   Errors reported by the corresponding dependency. Each derives from
   :py:exc:`CuDecompError`.

.. py:data:: __version__

   cuDecomp package version.
