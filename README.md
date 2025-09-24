# cuDecomp

An Adaptive Pencil Decomposition Library for NVIDIA GPUs

## Introduction

cuDecomp is a library for managing 1D (slab) and 2D (pencil) parallel decompositions of 3D Cartesian spatial domains on NVIDIA GPUs, with routines to perform global transpositions and halo communications. The library is inspired by the [2DECOMP&FFT Fortran library](https://github.com/xcompact3d/2decomp-fft), a popular decomposition library for numerical simulation codes, with a similar set of available transposition routines. While 2DECOMP&FFT and similar libraries in the past have been written to target CPU systems, this library is designed for GPU systems, leveraging CUDA-aware MPI and additional communication libraries optimized for GPUs, like the [NVIDIA Collective Communication Library (NCCL)](https://github.com/NVIDIA/nccl) and [NVIDIA OpenSHMEM Library (NVSHMEM)](https://developer.nvidia.com/nvshmem).

Please refer to the [documentation](https://nvidia.github.io/cuDecomp/) for additional information on the library and usage details.

This library is currently in a research-oriented state, and has been released as a companion to a paper presented at the PASC22 conference ([link](https://dl.acm.org/doi/10.1145/3539781.3539797)). We are making it available here as it can be useful in other applications outside of this study or as a benchmarking tool and usage example for various GPU communication libraries to perform transpose and halo communication.

Please contact us or open a GitHub issue if you are interested in using this library in your own solvers and have questions on usage and/or feature requests.

## Build
You can build this library using CMake. A CMake build of the library without additional examples/tests can be completed using the following commands
```shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```
There are several build variables available to configure the CMake build which can be found at the top of the project [`CMakeLists.txt`](CMakeLists.txt) file. As an example,
to configure the build to compile additional examples and enable NVSHMEM backends, you can run the following CMake command
```shell
$ cmake -DCUDECOMP_BUILD_EXTRAS=1 -DCUDECOMP_ENABLE_NVSHMEM=1 ..
```


### Dependencies
We strongly recommend building this library using NVHPC SDK compilers and libraries, as the SDK contains all required dependencies for this library and is the focus of our testing. Fortran features are only supported using NVHPC SDK compilers.

One exception is cuDecomp builds using NVSHMEM versions older than v3.0, which require the use of a bootstrapping layer that depends on your MPI distribution. The NVSHMEM library packaged within NVHPC SDK
supports OpenMPI only. If you require usage of a different MPI implementation (e.g. Spectrum MPI or Cray MPICH), you need to either build
NVSHMEM against your desired MPI implementation, or build a custom MPI bootstrap layer separately. Please refer to this [NVSHMEM documentation section](https://docs.nvidia.com/hpc-sdk/nvshmem/install-guide/index.html#use-nvshmem-mpi) for more details.
For cuDecomp builds using NVSHMEM v3.0+, this additional MPI boostrapping layer is no longer required.

Additionally, this library utilizes CUDA-aware MPI and is only compatible with MPI libraries with these features enabled.

## License
This library is released under an Apache 2.0 license, which can be found in [LICENSE](LICENSE).

