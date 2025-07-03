.. _env-var-section-ref:

Environment Variables
==============================

The following section lists the environment variables available to configure the cuDecomp library.

CUDECOMP_ENABLE_NCCL_UBR
------------------------
(since v0.4.0, requires NCCL v2.19 or newer)

:code:`CUDECOMP_ENABLE_NCCL_UBR` controls whether cuDecomp registers its communication buffers with the NCCL library using :code:`ncclCommRegister`/:code:`ncclCommDeregister` (i.e., user buffer registration).
Registration can improve NCCL send/receive performance in some scenarios. See the `User Buffer Registration <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html>`_
section of the NCCL documentation for more details.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.

CUDECOMP_ENABLE_CUMEM
------------------------
(since v0.5.0, requires CUDA 12.3 driver/toolkit or newer)

:code:`CUDECOMP_ENABLE_CUMEM` controls whether cuDecomp uses :code:`cuMem*` APIs to allocate fabric-registered workspace buffers via :code:`cudecompMalloc`. This option can improve the performance of
some MPI distributions on multi-node NVLink (MNNVL) capable systems.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.

CUDECOMP_ENABLE_CUDA_GRAPHS
---------------------------
(since v0.5.1, requires CUDA 11.1 driver/toolkit or newer)

:code:`CUDECOMP_ENABLE_CUDA_GRAPHS` controls whether cuDecomp uses CUDA Graphs APIs to capture/replay packing operations for pipelined backends. This option can improve the launch efficiency
and communication overlap of packing kernels in large scale cases.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.

CUDECOMP_ENABLE_PERFORMANCE_REPORTING
-------------------------------------
(since v0.5.1)

:code:`CUDECOMP_ENABLE_PERFORMANCE_REPORTING` controls the level of performance reporting cuDecomp prints to the console. This option requires a device synchronization after each transpose operation to capture event timings which can impact performance.

The following values are supported:

- :code:`0`: Performance reporting disabled
- :code:`1`: Final performance summary only - prints a comprehensive table with averaged performance statistics for all transpose operation configurations when the grid descriptor is destroyed
- :code:`2`: Verbose reporting - prints both per-operation performance reports and the final performance summary

Default setting is off (:code:`0`).

CUDECOMP_PERFORMANCE_REPORT_SAMPLES
-----------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_SAMPLES` controls the number of performance samples to keep for the final performance report. This setting determines the size of the circular buffer used to store timing measurements for each transpose configuration.

Default setting is :code:`20` samples. Valid range is 1-1000 samples.

CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES
------------------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES` controls the number of initial samples to ignore for each transpose configuration. This helps exclude outliers from GPU warmup, memory allocation, and other initialization effects from the final performance statistics.

Default setting is :code:`2` warmup samples. Valid range is 0-100 samples. Setting this to 0 disables warmup sample filtering.
