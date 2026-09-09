.. _cuda-graph-capture-ref:

CUDA Graph Capture
==================

cuDecomp transpose and halo operations can be captured as part of a caller-owned CUDA Graph. This allows an
application to capture a larger workflow containing cuDecomp operations and replay it with the standard CUDA Graph
APIs.

Compatibility
-------------

.. list-table:: External capture compatibility
   :header-rows: 1

   * - Communication family
     - Supported
     - Notes
   * - MPI
     - No
     - Performs host-side MPI communication and CUDA stream synchronization.
   * - NCCL
     - Yes
     - Includes the pipelined transpose backend.
   * - NVSHMEM
     - Yes
     - Includes the pipelined and device-initiated transpose backends; requires NVSHMEM 2.9 or newer. NVSHMEM 2.6
       through 2.8 remain supported for normal, uncaptured operation.

External capture is separate from cuDecomp's internal graph optimization. The
:code:`CUDECOMP_ENABLE_CUDA_GRAPHS` environment variable must be disabled. If it is enabled, a cuDecomp operation
called during external capture returns :code:`CUDECOMP_RESULT_NOT_SUPPORTED`. Performance reporting must also be
disabled during external capture.

Workspace preparation
---------------------

Caller-owned workspace can be used directly during capture. The workspace must remain valid until all graph
executables and graphs that reference it have been destroyed.

Automatic workspace requires a completed warmup before capture. Run each operation once, using the same datatype and
configuration that will be captured, and synchronize its stream before beginning capture. The warmup allocates enough
workspace and establishes any required communication-buffer registrations. An operation captured before its automatic
workspace is ready returns :code:`CUDECOMP_RESULT_NOT_SUPPORTED`.

The automatic workspace allocation domain is frozen when it is first used during capture. A later operation that
would require the allocation to grow returns :code:`CUDECOMP_RESULT_NOT_SUPPORTED`. Warm up the largest operation that
the handle will execute before capturing any operation, or use a separate handle for operations with different
workspace requirements.

Execution and lifetime requirements
-----------------------------------

All participating ranks must capture the same collective operation and must consistently use caller-owned or
automatic workspace. Operations using the same cuDecomp handle must not overlap in execution, including eager
operations and graph replays. Order them on the same stream or with explicit CUDA dependencies. Applications that
require concurrent cuDecomp operations must use separate handles.

Complete all graph launches and destroy graph executables and graphs before destroying the associated grid descriptor
or finalizing its cuDecomp handle.

Example
-------

The following example warms up an automatically managed workspace, captures one transpose, and instantiates the
resulting graph. Error checking is omitted for brevity.

.. code-block:: c++

  cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                        CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, stream);
  cudaStreamSynchronize(stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  cudecompTransposeXToY(handle, grid_desc, input, output, CUDECOMP_WORKSPACE_AUTO,
                        CUDECOMP_FLOAT, nullptr, nullptr, nullptr, nullptr, stream);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  cudaGraphLaunch(graph_exec, stream);
  cudaStreamSynchronize(stream);

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
