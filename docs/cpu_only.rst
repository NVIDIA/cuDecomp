CPU-only build
==============

Configure with ``-DCUDECOMP_BUILD_CPU_ONLY=ON`` for a limited, toolkit-free
debug/test build requiring C++17 and MPI (plus a Fortran compiler for the optional
bindings). This mode is intended for debugging and testing only, not for production
use. The API is unchanged except for the following differences:

* Library, function, and Fortran module names are unchanged. Use separate CPU/GPU
  build directories and installation prefixes, and only one variant per process.
* Buffers and ``cudecompMalloc`` allocations reside in host memory. Operations
  complete synchronously; stream arguments are ignored.
* Only MPI backends are supported. Set both ``pdims`` explicitly, with their product
  equal to the communicator size. Autotuning is unsupported; pass null autotuning
  options or disable tuning.
* Fortran bindings use host arrays and standard C pointers instead of device
  arrays/pointers.
* CUDA Graphs, performance reports, NCCL, NVSHMEM, and NVTX are unsupported.
