.. _autotuning-section-ref:

Autotuning
====================
One of the main features available in cuDecomp is the ability to perform
runtime autotuning of the process grid dimensions used to partition the
global domain and communication backends used for transpose and/or halo
communication. This feature enables users to run the library using the
best performing configuration for a given global domain size, number of tasks,
and compute cluster topology. The autotuner aims to select decomposition and
communication backend options that minimize transpose and halo communication time.


Autotuning process
------------------
The autotuning process in cuDecomp can be logically split into two categories:

  1. process grid autotuning
  2. communication backend autotuning

Process grid autotuning refers to selecting the :math:`P_{\text{rows}} \times P_{\text{cols}}`
of the process distribution, among the possible combinations with :math:`P_{\text{rows}} \times P_{\text{cols}} = N_{\text{GPU}}`.

Communication backend autotuning refers to selecting the transpose and/or halo backends used for communication between processes.

With all autotuning options enabled (i.e. full autotuning), cuDecomp will run the autotuning process in two phases.

During the first phase, cuDecomp will run all possible pairs of process grid dimensions
and communication backends of a user-selected type (either transpose or halo communication), identifying and selecting the pair that achieves the lowest average runtime out of 5 measured trials.
For transpose communication backends, the trial time is the time it takes to complete the
full set of transposes (XToY, YToZ, ZToY, YToX). For halo communication backends, the trial time is the time it takes to
complete a full set of halo updates (in all three grid directions), for a user-defined halo configuration.

Once a process grid and communication backend is selected that minimizes communication time of the user-selected type, the autotuner
will run a second phase to select a communication backend for the unselected communication type. In this phase, the process grid selected during the first phase is fixed and the backend with the minimum average trial runtime over 5 measured trials is selected.

In lieu of autotuning all options, users can also fix the process grid or communication options to limit the autotuning (for example,
fixing the process grid and autotuning the transpose and halo communication backends only).

We will illustrate this process in the following sections.


Autotuning usage
----------------
In this section, we will use a modified version of the basic usage example, explaining the changes required to enable the
autotuning feature.

Creating a grid descriptor with autotuning enabled
__________________________________________________
Enabling autotuning requires additional steps prior to the grid descriptor creation.

To enable process grid autotuning, instead of setting :code:`pdims` entries in the configuration struct to fixed values,
initialize them to zero. This indicates to cuDecomp that process grid autotuning is desired. Otherwise, if the :code:`pdims`
entries are set to desired process grid dimensions, process grid autotuning is disabled and these will remain fixed during
the autotuning process.

.. tabs::

  .. code-tab:: c++

    config.pdims[0] = 0; // P_rows
    config.pdims[1] = 0; // P_cols

  .. code-tab:: fortran

    config%pdims = [0, 0] ! [P_rows, P_cols]

In addition to this modification of :code:`pdims`, an autotuning options structure, :ref:`cudecompGridDescAutotuneOptions_t-ref` must be created,
populated, and passed as an additional argument to :ref:`cudecompGridDescCreate-ref`.

Create an uninitialized autotune option struct and initialize it to defaults using :ref:`cudecompGridDescAutotuneOptionsSetDefaults-ref`. Initializing this struct to default values is required to ensure no entries are left uninitialized.

.. tabs::

  .. code-tab:: c++

    cudecompGridDescAutotuneOptions_t options;
    CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));

  .. code-tab:: fortran

    type(cudecompGridDescAutotuneOptions) :: options
    ...

    istat = cudecompGridDescAutotuneOptionsSetDefaults(options)
    call CHECK_CUDECOMP_EXIT(istat)


First, let's go over general autotuning options that effect process grid and communication backend autotuning.

The :code:`n_warmup_trials` and :code:`n_trials` entries in the options struct control the number of warmup
and timed trials run for each tested configuration respectively. Here we set them to their default values.

.. tabs::

  .. code-tab:: c++

    options.n_warmup_trials = 3;
    options.n_trials = 5;

  .. code-tab:: fortran

    options%n_warmup_trials = 3
    options%n_trials = 5


The :code:`dtype` entry in the options struct controls which data type cuDecomp will use for autotuning.

.. tabs::

  .. code-tab:: c++

    options.dtype = CUDECOMP_DOUBLE;

  .. code-tab:: fortran

    options%dtype = CUDECOMP_DOUBLE

The :code:`disable_nccl_backends` and :code:`disable_nvshmem_backends` entries are boolean flags controlling whether
the autotuner will test transpose and halo communication backends using the NCCL or NVSHMEM libraries respectively.
By default, these flags are set to false and NCCL and NVSHMEM backends are enabled.

.. tabs::

  .. code-tab:: c++

    options.disable_nccl_backends = false;
    options.disable_nvshmem_backends = false;

  .. code-tab:: fortran

    options%disable_nccl_backends = .false.
    options%disable_nvshmem_backends = .false.

The :code:`skip_threshold` entry allows the autotuner to rapidly skip slow performing configurations. In particular,
the autotuner will skip testing a configuration if :code:`skip_threshold * t > t_best`, where :code:`t` is the duration
of the first timed trial for the configuration and :code:`t_best` is the average trial time of the current best configuration. 
By default, the threshold is set to zero which disables any skipping. More aggressive skipping can be useful in cases where exhaustive
testing of all possible configurations is too expensive.

.. tabs::

  .. code-tab:: c++

    options.skip_threshold = 0.0;

  .. code-tab:: fortran

    options%skip_threshold = 0.0

Moving on, these are the options specific to process grid autotuning.

The :code:`grid_mode` entry controls which type of communication (transpose or halo) to use to autotune the
process grid dimensions (see :ref:`cudecompAutotuneGridMode_t-ref`). By default, transpose communication is
used.

.. tabs::

  .. code-tab:: c++

    options.grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;

  .. code-tab:: fortran

    options%grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE

The :code:`allow_uneven_decompositions` entry is a boolean flag controlling whether the autotuner will test
process grid dimensions that result in uneven distributions of data (i.e. grids where pencil shapes are not
identical across ranks). By default, this flag is set to :code:`true` and uneven distributions are allowed.

.. tabs::

  .. code-tab:: c++

    options.allow_uneven_decompositions = true;

  .. code-tab:: fortran

    options%allow_uneven_decompositions = .true.

Next, these are the options specific to transpose communication backend autotuning.

The :code:`autotune_transpose_backend` entry is a boolean flag controlling whether the autotuner will autotune
the communication backend used for transposes. By default, this flag is :code:`false` and the transpose communication
backend is fixed to the value set within the configuration struct during the autotuning process. In
this example, we set it to true to enable transpose backend autotuning.

.. tabs::

  .. code-tab:: c++

    options.autotune_transpose_backend = true;

  .. code-tab:: fortran

    options%autotune_transpose_backend = .true.

The :code:`transpose_use_inplace_buffers` entry is an array of boolean flags that controls whether the transpose
communication during autotuning is performed in-place or out-of-place, on a per operation basis. This choice can impact transpose
performance due to some optimized paths that skip intermediate local operations in some situations,
depending on the input/output buffer locations.

For example, :code:`cudecompTransposeXToY` can be a no-op if:

  1. the process grid yields a decomposition with :math:`XY`-slabs (i.e. distributed along the :math:`Z`-axis only)
  2. the :math:`X`- and :math:`Y`-pencils are not in the permuted :code:`axis_contiguous` layout
  3. the transposition is performed in-place

In this configuration, the :math:`X`- and :math:`Y`-pencil buffers are in identical layouts and
contain the same data elements from the global grid. Since the transpose is in-place, the input
is already the output buffer, and no operation is performed. In contrast, an out-of-place transpose
would require a copy of data between the input and output buffers.

In this example, we use in-place buffers for all transpose operations so we can set all elements of :code:`transpose_use_inplace_buffers` to :code:`true`.
By default, the entries are set to :code:`false` and out-of-place buffers are used during autotuning.

.. tabs::

  .. code-tab:: c++

    options.transpose_use_inplace_buffers[0] = true; // use in-place buffers for X-to-Y transpose
    options.transpose_use_inplace_buffers[1] = true; // use in-place buffers for Y-to-Z transpose
    options.transpose_use_inplace_buffers[2] = true; // use in-place buffers for Z-to-Y transpose
    options.transpose_use_inplace_buffers[3] = true; // use in-place buffers for Y-to-X transpose

  .. code-tab:: fortran

    options%transpose_use_inplace_buffers(1) = .true. ! use in-place buffers for X-to-Y transpose
    options%transpose_use_inplace_buffers(2) = .true. ! use in-place buffers for Y-to-Z transpose
    options%transpose_use_inplace_buffers(3) = .true. ! use in-place buffers for Z-to-Y transpose
    options%transpose_use_inplace_buffers(4) = .true. ! use in-place buffers for Y-to-X transpose

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

.. tabs::

  .. code-tab:: c++

    options.transpose_op_weights[0] = 1.0; // apply 1.0 multiplier to X-to-Y transpose timings
    options.transpose_op_weights[1] = 1.0; // apply 1.0 multiplier to Y-to-Z transpose timings
    options.transpose_op_weights[2] = 1.0; // apply 1.0 multiplier to Z-to-Y transpose timings
    options.transpose_op_weights[3] = 1.0; // apply 1.0 multiplier to Y-to-X transpose timings

  .. code-tab:: fortran

    options%transpose_op_weights(1) = 1.0 ! apply 1.0 multiplier to X-to-Y transpose timings
    options%transpose_op_weights(2) = 1.0 ! apply 1.0 multiplier to Y-to-Z transpose timings
    options%transpose_op_weights(3) = 1.0 ! apply 1.0 multiplier to Z-to-Y transpose timings
    options%transpose_op_weights(4) = 1.0 ! apply 1.0 multiplier to Y-to-X transpose timings

Lastly, these are the options specific to halo communication backend autotuning.

The :code:`autotune_halo_backend` entry is a boolean flag controlling whether the autotuner will autotune
the communication backend used for halo exchanges. By default, this flag is :code:`false` and the halo communication
backend is fixed to the value set within the configuration struct during the autotuning process. In
this example, we set it to true to enable halo backend autotuning.

.. tabs::

  .. code-tab:: c++

    options.autotune_halo_backend = true;

  .. code-tab:: fortran

    options%autotune_halo_backend = .true.

The :code:`halo_extents`, :code:`halo_periods`, and :code:`halo_axis` define the halo configuration to use during halo autotuning.
See documentation on the halo communication routines, like :ref:`cudecompUpdateHalosX-ref` for descriptions of these
fields. In this example, we autotune for :math:`X`-pencil halo exchanges with one halo element in each direction
with periodic boundaries.

.. tabs::

  .. code-tab:: c++

    options.halo_axis = 0;

    options.halo_extents[0] = 1;
    options.halo_extents[1] = 1;
    options.halo_extents[2] = 1;

    options.halo_periods[0] = true;
    options.halo_periods[1] = true;
    options.halo_periods[2] = true;

  .. code-tab:: fortran

    options%halo_axis = 1

    options%halo_extents = [1, 1, 1]

    options%halo_periods = [.true., .true., .true]

With the grid descriptor configuration and autotuning options structures created and populated,
we can now create the grid descriptor with autotuning.

.. tabs::

  .. code-tab:: c++

    cudecompGridDesc_t grid_desc;
    CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  .. code-tab:: fortran

    istat = cudecompGridDescCreate(handle, grid_desc, config, options)
    call CHECK_CUDECOMP_EXIT(istat)

Autotuner output and querying results
_________________________________________________
When autotuning is enabled, cuDecomp will produce additional output to stdout to report
on the autotuning process, providing trial timings and similar information on the
tested configurations.

For example, running this example on a 4 GPU system will produce output as follows.
First, the autotuner will run the first phase, which in this case is performing process
grid autotuning and transpose backend autotuning (since we set :code:`grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE`.
The output generated from this phase will be like the following:

.. code-block:: none
  :emphasize-lines: 2-7, 58

  CUDECOMP: Running transpose autotuning...
  CUDECOMP:       grid: 1 x 4, backend: MPI_P2P
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.266102/0.276084/0.270158/0.003797
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.018432/0.026624/0.020941/0.002208
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.101376/0.110592/0.104806/0.002341
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.095232/0.101376/0.097229/0.001602
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.015360/0.020480/0.017459/0.001354
  CUDECOMP:       grid: 1 x 4, backend: MPI_P2P (pipelined)
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.456339/0.483480/0.467483/0.011253
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.018432/0.024576/0.020531/0.001354
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.188416/0.196608/0.191488/0.002243
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.194560/0.229376/0.207411/0.011130
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.016384/0.022528/0.019046/0.001498
  CUDECOMP:       grid: 1 x 4, backend: MPI_A2A
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.253752/0.275133/0.262857/0.006987
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.017408/0.021504/0.019661/0.001054

  ...

  CUDECOMP:       grid: 2 x 2, backend: MPI_P2P
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.302244/0.306223/0.303693/0.001211
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.067584/0.078848/0.072704/0.003123
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.068608/0.081920/0.073165/0.003569
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.060416/0.067584/0.063949/0.002278
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.059392/0.077824/0.069530/0.005864
  CUDECOMP:       grid: 2 x 2, backend: MPI_P2P (pipelined)
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.346133/0.354265/0.350742/0.002535
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.073728/0.087040/0.080538/0.005184
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.072704/0.093184/0.082586/0.006470
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.072704/0.080896/0.076851/0.002941
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.070656/0.098304/0.083558/0.008859
  CUDECOMP:       grid: 2 x 2, backend: MPI_A2A
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.289410/0.320966/0.298509/0.011557
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.064512/0.074752/0.070093/0.003197
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.065536/0.084992/0.073011/0.005610

  ...

  CUDECOMP:       grid: 4 x 1, backend: MPI_P2P
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.227092/0.233280/0.229325/0.002050
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.092160/0.099328/0.095181/0.001956
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.011264/0.016384/0.013005/0.001556
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.009216/0.012288/0.010240/0.000971
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.083968/0.095232/0.087910/0.003042
  CUDECOMP:       grid: 4 x 1, backend: MPI_P2P (pipelined)
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.355253/0.363846/0.358656/0.003062
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.147456/0.155648/0.150938/0.002736
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.011264/0.015360/0.013363/0.001354
  CUDECOMP:       TransposeZY time min/max/avg/std [ms]: 0.010240/0.014336/0.011878/0.001566
  CUDECOMP:       TransposeYX time min/max/avg/std [ms]: 0.152576/0.172032/0.158720/0.005909
  CUDECOMP:       grid: 4 x 1, backend: MPI_A2A
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.220565/0.226522/0.224512/0.002049
  CUDECOMP:       TransposeXY time min/max/avg/std [ms]: 0.075776/0.095232/0.086118/0.008030
  CUDECOMP:       TransposeYZ time min/max/avg/std [ms]: 0.010240/0.015360/0.012493/0.001539

  ...

  CUDECOMP: SELECTED: grid: 4 x 1, backend: NCCL, Avg. time 0.138808
  CUDECOMP: transpose autotuning time [s]: 1.589209

The first highlighted block of output shows the autotuning trial results for one configuration tested, in this case,
a :math:`1 \times 4` process grid paired with the :code:`MPI_P2P` (i.e. :code:`CUDECOMP_TRANSPOSE_COMM_MPI_P2P`) transpose
communication backend. The total time to complete all transposes is listed first, with the minimum, maximum, average, and
standard deviation over the trials printed. Following this, a further breakdown of the transpose timings by operation is listed
to provide additional insight into the performance.

The autotuner then proceeds to try other possible process grid and transpose communication backend pairs, in this case
continuing on to test :math:`2 \times 2` process grid options and :math:`1 \times 4` process grid options. After all the
configurations are tested, the autotuner selects the process grid and transpose communication backend pair that
achieves the lowest average trial time, and reports the selection, shown by the highlighted line at the end of the block.
In this case, it selected a :math:`4 \times 1` process grid using the NCCL (i.e. :code:`CUDECOMP_TRANSPOSE_COMM_NCCL`) backend.

If autotuning of the other type of communication is requested, the autotuning procedure moves onto the second phase,
selecting the best communication backend for this communication using the process grid selected in the first phase.
In this example, the second phase of autotuning is done to select a halo communication backend to use on the selected
:math:`4 \times 1` process grid.


.. code-block:: none
  :emphasize-lines: 3-4, 13

  CUDECOMP: Running halo autotuning...
  CUDECOMP: Autotune halo axis: x
  CUDECOMP:       grid: 4 x 1, halo backend: MPI
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.068239/0.074815/0.070960/0.002477
  CUDECOMP:       grid: 4 x 1, halo backend: MPI (blocking)
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.073353/0.085638/0.077625/0.003406
  CUDECOMP:       grid: 4 x 1, halo backend: NCCL
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.053063/0.063682/0.057200/0.003232
  CUDECOMP:       grid: 4 x 1, halo backend: NVSHMEM
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.050031/0.052668/0.051291/0.000742
  CUDECOMP:       grid: 4 x 1, halo backend: NVSHMEM (blocking)
  CUDECOMP:       Total time min/max/avg/std [ms]: 0.063190/0.067428/0.065849/0.001215
  CUDECOMP: SELECTED: grid: 4 x 1, halo backend: NVSHMEM, Avg. time [s] 0.051291
  CUDECOMP: halo autotuning time [s]: 0.227950

The first highlighted block shows the results for one configuration tested, in this case the :code:`MPI` (i.e. :code:`CUDECOMP_HALO_COMM_MPI`)
halo communication backend operating on a :math:`4 \times 1` process grid. The total time to complete the full set of halo exchanges for
the user-selected pencil axis is reported similar to the transpose trials. The autotuner proceeds to test all the other halo
communication backend options, selects the one achieving the lowest average trial time, and reports the selection in the final highlighted
line. In this case, it selected the NVSHMEM halo communication backend (i.e. :code:`CUDECOMP_HALO_COMM_NVSHMEM`).

After the autotuning process is complete, the grid descriptor is created and ready to use. Entries in the configuration struct provided
to :code:`cudecompGridDescCreate` corresponding
to autotuned fields (:code:`pdims`, :code:`transpose_comm_backend`, and :code:`halo_comm_backend`) are updated to reflect the autotuning
selections. Thus, one can also run the following code like the following to inspect and report the final configuration used:

.. tabs::

  .. code-tab:: c++

    if (rank == 0) {
      printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
      printf("running using %s transpose backend...\n",
             cudecompTransposeCommBackendToString(config.transpose_comm_backend));
      printf("running using %s halo backend...\n",
             cudecompHaloCommBackendToString(config.halo_comm_backend));
    }

  .. code-tab:: fortran

    if (rank == 0) then
      write(*,"('running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
      write(*,"('running using ', a, ' transpose backend ...')") &
                cudecompTransposeCommBackendToString(config%transpose_comm_backend)
      write(*,"('running using ', a, ' halo backend ...')") &
                cudecompHaloCommBackendToString(config%halo_comm_backend)
    endif

As autotuning only impacts grid descriptor creation, the rest of the usage of the library is unchanged from that illustrated in
the basic usage section.
