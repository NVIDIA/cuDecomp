.. _python-api-ref:

Python API (experimental)
=========================

The ``cudecomp`` package provides Python bindings for cuDecomp. This API is
experimental and may change between releases.

For prerequisites and source, editable, or wheel installation, see
`Python build instructions in the README
<https://github.com/NVIDIA/cuDecomp#python-experimental>`_.

.. toctree::
   :maxdepth: 2

   python/basic_usage
   python/autotuning
   python/api
   python/cpu_only

Array and datatype conventions
------------------------------

GPU operations accept C-contiguous, three-dimensional arrays exposing
``__cuda_array_interface__``. Array shapes must match the corresponding
:py:meth:`~cudecomp.GridDesc.get_pencil_info` query, including the same halos and
padding. NumPy host arrays are not accepted by GPU builds.

.. list-table:: Supported datatypes
   :header-rows: 1

   * - Array dtype
     - :py:class:`cudecomp.DataType`
   * - ``float32``
     - ``FLOAT``
   * - ``float64``
     - ``DOUBLE``
   * - ``complex64``
     - ``FLOAT_COMPLEX``
   * - ``complex128``
     - ``DOUBLE_COMPLEX``

Operation ``dtype`` arguments accept a ``DataType`` member or ``None`` to infer
the type from the input array. NumPy dtype objects, scalar types, and strings
are not accepted as dtype arguments. An explicit dtype must agree with the
arrays and does not convert data. Big-endian data and non-``None`` validity masks
are unsupported. CUDA Array Interface versions 0 through 3 are accepted.

Transposes allocate a CuPy output when ``out`` is omitted; this requires a CuPy
input. Allocation uses CuPy's standard memory pool on the operation stream,
regardless of the configured allocator. Supply ``out`` for caller-allocated
storage or other CUDA array providers. Opaque storage requires an explicit dtype
with a matching element size. Output and halo-update arrays must be writable.
Framework differentiation is not supported.

Streams, devices, and lifetimes
-------------------------------

Transpose and halo ``stream`` arguments accept ``None``, an integer CUDA stream
handle, an object implementing the CUDA stream protocol (``__cuda_stream__``),
such as a ``cuda.core`` or CuPy stream, or an object with an integer ``ptr``
attribute. Omitting the argument or passing ``None`` uses the current stream of
the input array's package, as in nvmath-python: CuPy's current stream for CuPy
inputs, and CUDA default stream 0 for other array providers.

Operations are asynchronous with respect to the host. Callers must order buffer
accesses before and after each operation using the same stream or explicit CUDA
dependencies. Array producer-stream metadata does not establish dependencies.
Keep buffers and caller-owned streams alive until queued work completes.

Select the CUDA device before constructing a handle. Use compatible devices
for the handle, arrays, and streams; automatic CuPy allocation uses the current
device. The binding does not select a device from an input array.

MPI and resource management
---------------------------

cuDecomp requires an initialized MPI communicator, which must remain valid for
the handle's lifetime. Importing ``mpi4py.MPI`` normally initializes MPI;
cuDecomp does not initialize or finalize MPI. Initialization, grid management,
finalization, and communication follow the C API collective requirements.

Call :py:meth:`~cudecomp.GridDesc.destroy` and
:py:meth:`~cudecomp.Handle.finalize` explicitly and consistently across ranks,
before finalizing MPI. Python garbage collection does not perform collective
cleanup. The binding can release the GIL; serialize use and cleanup of shared
handles/grids across Python threads. Operations on one handle must not overlap.

Workspaces are managed automatically and reused across calls; warm up operations
before measuring steady-state performance. See :ref:`cuda-graph-capture-ref` for
C API capture requirements.
