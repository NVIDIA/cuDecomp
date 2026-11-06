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


### GPU dependencies
The NVHPC SDK remains the primary tested environment and provides all required
GPU dependencies, but the C++ library does not require the NVHPC C++ compiler.
GPU Fortran bindings require `nvfortran`. CMake 3.18 or newer is required.
When no C++ compiler or toolchain is selected explicitly, CMake prefers
`nvc++` when it is available.

The CUDA Toolkit location is inferred from the selected NVIDIA CUDA compiler.
To select a specific CUDA installation, set `CUDACXX` or
`CMAKE_CUDA_COMPILER` when configuring a fresh build directory. When building
the Fortran bindings against an external CUDA Toolkit, also set
`NVHPC_CUDA_HOME` or `NVCOMPILER_CUDA_HOME` to an installation with the same
CUDA major and minor version.

CMake automatically searches system locations and paths exposed by the current
environment, including `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`. If
dependencies cannot be found there, use `CUDECOMP_CUTENSOR_HOME`/`CUTENSOR_ROOT`,
`CUDECOMP_NCCL_HOME`/`NCCL_ROOT`, or `CUDECOMP_NVSHMEM_HOME`/`NVSHMEM_ROOT`. The
`CUDECOMP_*_HOME` setting takes precedence when both forms are provided.

One exception is cuDecomp builds using NVSHMEM versions older than v3.0, which require the use of a bootstrapping layer that depends on your MPI distribution. The NVSHMEM library packaged within NVHPC SDK
supports OpenMPI only. If you require usage of a different MPI implementation (e.g. Spectrum MPI or Cray MPICH), you need to either build
NVSHMEM against your desired MPI implementation, or build a custom MPI bootstrap layer separately. Please refer to this [NVSHMEM documentation section](https://docs.nvidia.com/hpc-sdk/nvshmem/install-guide/index.html#use-nvshmem-mpi) for more details.
For cuDecomp builds using NVSHMEM v3.0+, this additional MPI boostrapping layer is no longer required.

Additionally, this library utilizes CUDA-aware MPI and is only compatible with MPI libraries with these features enabled.

### Python (experimental)

The Python API requires Python 3.10 or newer and the same CUDA-aware MPI and GPU
dependencies described above. Install from the repository root:

```shell
python -m pip install .
```

Package builds use CMake and a C++17 compiler, with Fortran bindings and examples
disabled. Pip installs the build requirements (`scikit-build-core` and
`pybind11>=2.12`) and runtime dependencies (`numpy>=1.23` and `mpi4py>=3.1`).
The MPI implementation loaded by mpi4py must be ABI-compatible with the one used
to build cuDecomp.

Install CuPy separately for GPU examples and automatic transpose output
allocation. Choose the package matching your CUDA environment, for example:

```shell
python -m pip install cupy-cuda13x  # Use cupy-cuda12x for CUDA 12.
```

Editable installs and local source/wheel builds use the same GPU configuration:

```shell
python -m pip install -e .
python -m pip install build
python -m build
```

`python -m build` creates an sdist and builds a wheel from it. The wheel includes
the Python extension and cuDecomp shared library, but relies on the MPI, CUDA,
NCCL, cuTENSOR, and enabled optional libraries in the runtime environment. Local
wheels are intended for compatible systems or cluster environments, rather than
as portable binary distributions.

Dependency locations and optional features can be set through environment
variables such as `CUDECOMP_NCCL_HOME`, `CUDECOMP_CUTENSOR_HOME`, and
`CUDECOMP_ENABLE_NVSHMEM`. Other CMake settings can be passed through pip, for
example `python -m pip install . -Ccmake.define.CUDECOMP_CUDA_CC_LIST=89`.

See the [Python documentation](docs/python_api.rst) for basic usage, autotuning,
and the API reference.

## License
This library is released under an Apache 2.0 license, which can be found in [LICENSE](LICENSE).
