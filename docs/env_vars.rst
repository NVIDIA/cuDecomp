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

CUDECOMP_ENABLE_PERFORMANCE_REPORT
------------------------------------
(since v0.5.1)

:code:`CUDECOMP_ENABLE_PERFORMANCE_REPORT` controls whether cuDecomp performance reporting is enabled.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.

CUDECOMP_PERFORMANCE_REPORT_DETAIL
----------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_DETAIL` controls the verbosity of performance reporting when :code:`CUDECOMP_ENABLE_PERFORMANCE_REPORT` is enabled. This setting determines whether individual sample data is printed in addition to the aggregated performance summary.

The following values are supported:

- :code:`0`: Aggregated report only - prints only the summary table with averaged performance statistics (default)
- :code:`1`: Per-sample reporting on rank 0 - prints individual sample data for each transpose/halo configuration, but only from rank 0
- :code:`2`: Per-sample reporting on all ranks - prints individual sample data for each transpose/halo configuration from all ranks, gathered and sorted by rank on rank 0

Default setting is :code:`0`.

CUDECOMP_PERFORMANCE_REPORT_SAMPLES
-----------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_SAMPLES` controls the number of performance samples to keep for the final performance report. This setting determines the size of the circular buffer used to store timing measurements for each transpose/halo configuration.

Default setting is :code:`20` samples.

CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES
------------------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES` controls the number of initial samples to ignore for each transpose/halo configuration. This helps exclude outliers from GPU warmup, memory allocation, and other initialization effects from the final performance statistics.

Default setting is :code:`3` warmup samples. Setting this to 0 disables warmup sample filtering.

CUDECOMP_PERFORMANCE_REPORT_WRITE_DIR
-------------------------------------
(since v0.5.1)

:code:`CUDECOMP_PERFORMANCE_REPORT_WRITE_DIR` controls the directory where CSV performance reports are written when :code:`CUDECOMP_ENABLE_PERFORMANCE_REPORT` is enabled. When this variable is set, cuDecomp will write performance data to CSV files in the specified directory.

CSV files are created with descriptive names encoding the grid configuration, for example:
:code:`cudecomp-perf-report-transpose-aggregated-tcomm_1-hcomm_1-pdims_2x2-gdims_256x256x256-memorder_012012012.csv`

The following CSV files are generated:

- Aggregated transpose performance data
- Aggregated halo performance data
- Per-sample transpose data (when :code:`CUDECOMP_PERFORMANCE_REPORT_DETAIL` > 0)
- Per-sample halo data (when :code:`CUDECOMP_PERFORMANCE_REPORT_DETAIL` > 0)

Each CSV file includes grid configuration information as comments at the top, followed by performance data in comma-separated format.

Default setting is unset (no CSV files written). Setting this variable to a directory path will enable CSV file output.
