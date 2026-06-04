# Tests

This directory contains cuDecomp functional tests. The primary routine
development test workflow is the CTest suite under [`ctest/`](ctest/),
which covers public API behavior plus transpose and halo correctness for the
backends available in the current build. When Fortran bindings are enabled, the
CTest workflow also includes focused Fortran API coverage plus MPI functional
coverage across the Fortran dtype-specialized executables. The
legacy C++ and Fortran executables are still available for broader sweeps and
targeted configuration testing.

## CTest Suite

### Build

The default project build does not build tests. Build the CTest suite with
`CUDECOMP_BUILD_TESTS=ON`:

```shell
mkdir -p build
cd build
cmake -DCUDECOMP_BUILD_TESTS=ON ..
make -j"$(nproc)"
```

`CUDECOMP_BUILD_TESTS=ON` also builds the legacy C++/Fortran test executables
under this directory. The C++ CTest executables use GoogleTest, which is found
with `find_package(GTest)` by default. If a system GoogleTest package is
unavailable, configure with `-DCUDECOMP_TEST_FETCH_GTEST=ON` to fetch the pinned
test dependency.

### Running Tests

List registered tests with:

```shell
cd build
ctest -N
```

The default test order is API, regular transpose tests, regular halo tests, then
specialized CUDA Graphs, NCCL user-buffer-registration tests, and focused
Fortran API/functional tests when Fortran bindings are enabled. NVSHMEM tests are
registered only with NVSHMEM-enabled builds.

Useful labels:

| Label | Tests selected |
| --- | --- |
| `api` | Public C API and focused Fortran API behavior tests |
| `transpose` | All transpose correctness tests |
| `halo` | All halo correctness tests |
| `fortran` | Focused Fortran API and MPI functional tests across Fortran dtypes, when built |
| `mpi` | MPI-backend tests |
| `nccl` | NCCL-backend tests |
| `nvshmem` | NVSHMEM-backend tests, when built |
| `cuda_graphs` | CUDA Graphs functional coverage |
| `nccl_ubr` | NCCL user buffer registration functional coverage |

Run tests by name or label:

```shell
cd build
ctest --output-on-failure -R "cudecomp_(api|transpose_mpi|halo_mpi)$"
ctest --output-on-failure -L mpi
ctest --output-on-failure -L cuda_graphs
```

### GPU Requirements

The CTest suites require GPUs. They run four MPI ranks by default and work best
on systems with four visible GPUs available. Test setup fails when no CUDA
device is visible.

On systems with fewer than four visible GPUs, CUDA MPS is required so multiple
local MPI ranks can share a GPU. Set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to
`100 / nranks` for the local ranks sharing a GPU. For the default four-rank
CTest suites on one GPU, this value is `25`.

NCCL tests using MPS also require NCCL 2.30 or newer and
`NCCL_MULTI_RANK_GPU_ENABLE=1`. If these NCCL-specific requirements are not met,
the NCCL cases skip so MPI-capable systems can still run the non-NCCL tests.

## Legacy Executables

The CTest suites above are the recommended routine development tests. The legacy
runners remain useful for broader manual sweeps or for targeting a specific
configuration that is not part of the focused CTest matrix.

The testing executables accept a number of flags to control the configuration of the test (run `cc/transpose_test -h` or
`cc/halo_test -h` for a listing of available options). You can use these binaries to test particular configurations of cuDecomp (i.e.
global grid, process grid, communication backends, datatype, etc.) to verify functionality.

There is a [`test_runner.py`](test_runner.py) script that runs sweeps of these tests across a number of different configurations, defined
in [`test_config.yaml`](test_config.yaml). This is mostly for internal use, but usage of this script closely matches that of
the [`benchmark_runner`](../benchmark/benchmark_runner.py) script, documented [here](../benchmark/README.md).
