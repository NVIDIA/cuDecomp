.. _nvshmem-section-ref:

Working with NVSHMEM-enabled builds
===================================
When cuDecomp is built with NVSHMEM enabled, there are a few concepts and potential pitfalls that
users should be aware of which we will highlight in this section.


Controlling the symmetric heap size
-----------------------------------
In general, NVSHMEM operations requires memory it operates on to be allocated on its `symmetric heap`, via
:code:`nvshmem_malloc`. While cuDecomp attempts to hide this complexity behind :code:`cudecompMalloc`, it is important
to understand that memory allocated for usage with NVSHMEM comes out of a separate memory pool than all other
CUDA allocations. At a high-level, NVSHMEM will preallocate this symmetric heap on each GPU when it is initialized, 
with the heap size set by the `NVSHMEM_SYMMETRIC_SIZE <https://docs.nvidia.com/hpc-sdk/nvshmem/api/docs/gen/env.html#c.NVSHMEM_SYMMETRIC_SIZE>`_ environment variable.
As such, it is important to set the symmetric heap size to a value that is large enough for any necessary allocations from cuDecomp,
but not much larger as that will waste GPU memory space.

To help with this, the code will produce warnings like the following

.. code-block::

  CUDECOMP:WARN: Attempting an NVSHMEM allocation of 2147483648 bytes but *approximately* 1073741824 free bytes of 1073741824 total bytes of symmetric heap space available. If the allocation fails, set NVSHMEM_SYMMETRIC_SIZE >= 2147483648 and try again.

if the library detects NVSHMEM allocations that may exceed the symmetric heap size, and suggests an appropriate value for :code:`NVSHMEM_SYMMETRIC_SIZE`.

MPI compatibility
-----------------
As noted in the NVSHMEM documentation `here <https://docs.nvidia.com/hpc-sdk/nvshmem/api/docs/faq.html#interoperability-with-mpi-faqs>`_,
memory allocated on the symmetric heap may lead to crashes when used in MPI calls with some MPI implementations, especially when
CUDA VMM features in NVSHMEM are enabled. We strongly encourage users to set :code:`NVSHMEM_DISABLE_CUDA_VMM=1` when using cuDecomp
with NVSHMEM enabled. However, this is not always sufficient and MPI can still crash when passed NVSHMEM allocated memory.

Due to this, cuDecomp attempts to avoid using NVSHMEM-allocated memory with MPI where possible but it can arise in a couple of situations:

#. During autotuning, if there is not enough memory to allocate separate workspace buffers for NVSHMEM and non-NVSHMEM backends for testing,
   the library will attempt to run MPI backends using NVSHMEM allocated memory, printing the following message:

   .. code-block::

     CUDECOMP:WARN: Cannot allocate separate workspace for non-NVSHMEM backends during autotuning. Using NVSHMEM allocated workspace for all backends, which may cause issues for some MPI implementations. See documentation for more details and suggested workarounds.

   If this crashes on your system due to MPI incompatibilities, you should disable NVSHMEM backends for your problem configuration.

#. :code:`cudecompMalloc` will allocate memory based on both the transpose and halo communication backends. If one of those backends is
   NVSHMEM-enabled, it will allocate memory using `nvshmem_malloc`. If the other communication backend is using MPI, you may end up
   using an NVSHMEM-allocated workspace for that communication. If this is detected, cuDecomp will print the following message:

   .. code-block::

     CUDECOMP:WARN: A work buffer allocated with nvshmem_malloc (via cudecompMalloc) is being used with an MPI communication backend. This may cause issues with some MPI implementations. See the documentation for additional details and possible workarounds if you encounter issues.

   If this crashes on your system due to MPI incompatibilities, the suggested workaround is to use two separate grid descriptors,
   one for the transpose communication, and one for the halo communication, in order to avoid mixing workspace allocation types in
   the case where one communication type uses MPI and the others uses NVSHMEM.

