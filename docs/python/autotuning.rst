.. _python-autotuning-ref:

Autotuning
====================
cuDecomp can autotune process grid dimensions and communication backends.
See :ref:`autotuning-process-ref` for the search process shared by all bindings.

Autotuning usage
----------------
In this section, we will use a modified version of the :doc:`basic usage example <basic_usage>`,
explaining the changes required to enable the autotuning feature. The complete
`Python autotuning example
<https://github.com/NVIDIA/cuDecomp/blob/main/examples/python/basic_usage/basic_usage_autotune.py>`_
can be run with ``mpirun -n 4 python basic_usage_autotune.py``.

Creating a grid descriptor with autotuning enabled
__________________________________________________
Enabling autotuning requires additional steps prior to the grid descriptor creation.

To enable process grid autotuning, instead of setting :code:`pdims` entries in the configuration object to fixed values,
initialize them to zero. This indicates to cuDecomp that process grid autotuning is desired. Otherwise, if the :code:`pdims`
entries are set to desired process grid dimensions, process grid autotuning is disabled and these will remain fixed during
the autotuning process.

.. code-block:: python

   config.pdims = (0, 0)  # P_rows, P_cols

In addition to this modification of ``pdims``, an autotuning options object,
:py:class:`cudecomp.GridDescAutotuneOptions`, must be created, populated, and passed to
:py:meth:`cudecomp.Handle.create_grid`. Its constructor initializes all fields to defaults.

.. code-block:: python

   options = cudecomp.GridDescAutotuneOptions()

First, let's go over general autotuning options that affect process grid and communication backend autotuning.

The :code:`n_warmup_trials` and :code:`n_trials` entries in the options object control the number of warmup
and timed trials run for each tested configuration respectively. Here we set them to their default values.

.. code-block:: python

   options.n_warmup_trials = 3
   options.n_trials = 5

The :code:`dtype` entry in the options object controls which data type cuDecomp will use for autotuning.
This property takes a :py:class:`cudecomp.DataType` member.

.. code-block:: python

   options.dtype = cudecomp.DataType.DOUBLE

The :code:`disable_mpi_backends`, :code:`disable_nccl_backends`, and :code:`disable_nvshmem_backends` entries are
boolean flags controlling whether the autotuner will test transpose and halo communication backends using the MPI,
NCCL, or NVSHMEM libraries respectively. These exclusions are applied after any backend environment-variable filter.
By default, these flags are set to false and all three backend families are enabled.

.. code-block:: python

   options.disable_mpi_backends = False
   options.disable_nccl_backends = False
   options.disable_nvshmem_backends = False

The :code:`skip_threshold` entry allows the autotuner to rapidly skip slow performing configurations. In particular,
the autotuner will skip testing a configuration if :code:`skip_threshold * t > t_best`, where :code:`t` is the duration
of the first timed trial for the configuration and :code:`t_best` is the average trial time of the current best configuration.
By default, the threshold is set to zero which disables any skipping. More aggressive skipping can be useful in cases where exhaustive
testing of all possible configurations is too expensive.

.. code-block:: python

   options.skip_threshold = 0.0

Moving on, these are the options specific to process grid autotuning.

The :code:`grid_mode` entry controls which type of communication (transpose or halo) to use to autotune the
process grid dimensions (see :ref:`cudecompAutotuneGridMode_t-ref`). By default, transpose communication is
used.

.. code-block:: python

   options.grid_mode = cudecomp.AutotuneGridMode.TRANSPOSE

The :code:`allow_uneven_decompositions` entry is a boolean flag controlling whether the autotuner will test
process grid dimensions that result in uneven distributions of data (i.e. grids where pencil shapes are not
identical across ranks). By default, this flag is set to :code:`True` and uneven distributions are allowed.

.. code-block:: python

   options.allow_uneven_decompositions = True

Next, these are the options specific to transpose communication backend autotuning.

The :code:`autotune_transpose_backend` entry is a boolean flag controlling whether the autotuner will autotune
the communication backend used for transposes. By default, this flag is :code:`False` and the transpose communication
backend is fixed to the value set within the configuration object during the autotuning process. In
this example, we set it to true to enable transpose backend autotuning.

.. code-block:: python

   options.autotune_transpose_backend = True

The :code:`transpose_use_inplace_buffers` entry is an array of boolean flags that controls whether the transpose
communication during autotuning is performed in-place or out-of-place, on a per operation basis. This choice can impact transpose
performance due to some optimized paths that skip intermediate local operations in some situations,
depending on the input/output buffer locations.

For example, :code:`GridDesc.transpose_x_to_y` can be a no-op if:

  1. the process grid yields a decomposition with :math:`XY`-slabs (i.e. distributed along the :math:`Z`-axis only)
  2. the :math:`X`- and :math:`Y`-pencils are not in the permuted :code:`axis_contiguous` layout
  3. the transposition is performed in-place

In this configuration, the :math:`X`- and :math:`Y`-pencil buffers are in identical layouts and
contain the same data elements from the global grid. Since the transpose is in-place, the input
is already the output buffer, and no operation is performed. In contrast, an out-of-place transpose
would require a copy of data between the input and output buffers.

In this example, we use in-place buffers for all transpose operations so we can set all elements of :code:`transpose_use_inplace_buffers` to :code:`True`.
By default, the entries are set to :code:`False` and out-of-place buffers are used during autotuning.

.. code-block:: python

   options.transpose_use_inplace_buffers[:] = True

The :code:`transpose_op_weights` entry is an array of floating point weights that enable adjusting the
contribution of the different transpose operations to the trial timings used by the autotuner. By default,
the trial timings used by the autotuner are an unweighted sum of the X-to-Y, Y-to-Z, Z-to-Y, and Y-to-X transpose timings.
The entries in :code:`transpose_op_weights` are multiplicative weights that are applied to the
contribution of each transpose operation to the total trial timing.
This option is meant for programs that may invoke the different transpose operations an unequal
number of times and may want the autotuner to emphasize the more frequently invoked transpose operations
when measuring the performance of a backend and process grid configuration. For example, setting
the weight to :code:`0.0` for one of the transpose operations will indicate to the autotuner that the timing
of that operation should not contribute to the trial time sum. On a related note, the autotuner will skip running
any transpose operation with a weight of :code:`0.0` for efficiency.
In this example, we autotune using the full set of transpose
operations, and therefore set all elements of :code:`transpose_op_weights` to :code:`1.0`.
We should note that this is the default behavior, and thus there is no need to explicitly set
the elements to :code:`1.0` generally.

.. code-block:: python

   options.transpose_op_weights[:] = 1.0

There are additional entries to supply per transpose operation :code:`input_halo_extents`, :code:`output_halo_extents`,
:code:`input_padding`, and :code:`output_padding` arguments for use during autotuning: :code:`transpose_input_halo_extents`, :code:`transpose_output_halo_extents`,
:code:`transpose_input_padding`, and :code:`transpose_output_padding` respectively. These entries default to supplying zero halo extent and padding
arguments to the transpose operations. In this example, we want the autotuner to supply non-zero :code:`input_halo_extents` and :code:`output_halo_extents` arguments
to the transpose operations involving the :math:`X`-pencil.

.. code-block:: python

   options.transpose_input_halo_extents[0] = [1, 1, 1]   # X-to-Y input.
   options.transpose_output_halo_extents[3] = [1, 1, 1]  # Y-to-X output.

All other transpose operations do not have non-zero halo extents or padding arguments, so we leave those entries as their default values.

Lastly, these are the options specific to halo communication backend autotuning.

The :code:`autotune_halo_backend` entry is a boolean flag controlling whether the autotuner will autotune
the communication backend used for halo exchanges. By default, this flag is :code:`False` and the halo communication
backend is fixed to the value set within the configuration object during the autotuning process. In
this example, we set it to true to enable halo backend autotuning.

.. code-block:: python

   options.autotune_halo_backend = True

The :code:`halo_extents`, :code:`halo_periods`, and :code:`halo_axis` define the halo configuration to use during halo autotuning.
See documentation on the halo communication routines, like :ref:`cudecompUpdateHalosX-ref` for descriptions of these
fields. In this example, we autotune for :math:`X`-pencil halo exchanges with one halo element in each direction
with periodic boundaries.

.. code-block:: python

   options.halo_axis = 0
   options.halo_extents = (1, 1, 1)
   options.halo_periods = (True, True, True)

There is an additional entry, :code:`halo_padding`, to supply the padding argument to use during halo autotuning.
This entry defaults to supplying zero padding. In this example, we do not supply non-zero padding for the halo operations
so we leave this entry default valued.

With the grid descriptor configuration and autotuning options objects created and populated,
we can now create the grid descriptor with autotuning.

.. code-block:: python

   grid = handle.create_grid(config, autotune_options=options)

Autotuner output and querying results
_________________________________________________
When autotuning is enabled, cuDecomp reports trial timings and the selected configuration.
The output is the same for all bindings; see :ref:`autotuning-output-ref` for an
example and explanation of the reported timings and selections.

After the autotuning process is complete, the grid descriptor is created and ready to use. Entries in the configuration object provided
to :code:`Handle.create_grid` corresponding
to autotuned fields (:code:`pdims`, :code:`transpose_comm_backend`, and :code:`halo_comm_backend`) are updated to reflect the autotuning
selections. Thus, one can also run the following code like the following to inspect and report the final configuration used:

.. code-block:: python

   if rank == 0:
       print(f"running on {config.pdims[0]} x {config.pdims[1]} process grid...")
       print(f"running using {cudecomp.transpose_comm_backend_to_string(config.transpose_comm_backend)} transpose backend...")
       print(f"running using {cudecomp.halo_comm_backend_to_string(config.halo_comm_backend)} halo backend...")

As autotuning only impacts grid descriptor creation, the rest of the usage of the library is unchanged from that illustrated in
the basic usage section.
