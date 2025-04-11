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

CUDECOMP_ENABLE_NVSHMEM_SYNC
----------------------------
(since v0.5.1, requires NVSHMEM 2.5.0 or newer)

:code:`CUDECOMP_ENABLE_NVSHMEM_SYNC` controls whether cuDecomp uses NVSHMEM synchronization APIs in communication backends using NVSHMEM. This option makes these backends CPU synchronization free which
can improve performance in some cases.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.
