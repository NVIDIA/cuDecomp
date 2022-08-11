# cuDecomp

An Adaptive Pencil Decomposition Library for NVIDIA GPUs

## Introduction

cuDecomp is a library for managing 1D (slab) and 2D (pencil) parallel decompositions of 3D Cartesian spatial domains on NVIDIA GPUs, with routines to perform global transpositions and halo communications. The library is inspired by the [2DECOMP&FFT Fortran library](https://github.com/xcompact3d/2decomp-fft), a popular decomposition library for numerical simulation codes, with a similar set of available transposition routines. While 2DECOMP&FFT and similar libraries in the past have been written to target CPU systems, this library is designed for GPU systems, leveraging CUDA-aware MPI and additional communication libraries optimized for GPUs, like the [NVIDIA Collective Communication Library (NCCL)](https://github.com/NVIDIA/nccl) and [NVIDIA OpenSHMEM Library (NVSHMEM)](https://developer.nvidia.com/nvshmem).

Please refer to the [documentation](https://nvidia.github.io/cuDecomp/) for additional information on the library and usage details.

This library is currently in a research-oriented state, and has been released as a companion to a paper presented at the PASC22 conference ([link](https://dl.acm.org/doi/10.1145/3539781.3539797)). We are making it available here as it can be useful in other applications outside of this study or as a benchmarking tool and usage example for various GPU communication libraries to perform transpose and halo communication.

Please contact us or open a GitHub issue if you are interested in using this library in your own solvers and have questions on usage and/or feature requests.

## Build

To build the library, you must first create a configuration file to point the installed to dependent library paths and enable/disable features.
 See the default [`nvhpcsdk.conf`](configs/nvhpcsdk.conf) for an example of settings to build the library using the [NVHPC SDK compilers and libraries](https://developer.nvidia.com/hpc-sdk).
The [`configs/`](configs) directory also contains several sample build configuration files for a number of GPU compute clusters, like Perlmutter, Summit, and Marconi 100.

With this configuration file created, you can build the library using the command

```shell
$ make -j CONFIGFILE=<path to your configuration file>
```

The library will be compiled and installed in a newly created `build/` directory.

### Dependencies
We strongly recommend building this library using NVHPC SDK compilers and libraries, as the SDK contains all required dependencies for this library and is the focus of our testing. Fortran features are only supported using NVHPC SDK compilers.

One exception is NVSHMEM, which uses a bootstrapping layer that depends on your MPI installation. The NVSHMEM library packaged within NVHPC
supports OpenMPI only. If you require usage of a different MPI implementation (e.g. Spectrum MPI or Cray MPICH), you need to either build
NVSHMEM against your desired MPI implementation, or build a custom MPI bootstrap layer. Please refer to this [NVSHMEM documentation section](https://docs.nvidia.com/hpc-sdk/nvshmem/install-guide/index.html#use-nvshmem-mpi) for more details.

Additionally, this library utilizes CUDA-aware MPI and is only compatible with MPI libraries with these features enabled.

## License
This library is released under a BSD 3-clause license, which can be found in [LICENSE](license).

