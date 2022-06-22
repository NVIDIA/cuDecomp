# Tests
This subdirectory contains tests for both the transpose and halo communication routines in cuDecomp, in both C++ and Fortran.
The testing executables accept a number of flags to control the configuration of the test (run `cc/transpose_test -h` or
`cc/halo_test -h` for a listing of available options). You can use these binaries to test particular configurations of cuDecomp (i.e.
global grid, process grid, communication backends, datatype, etc.) to verify functionality.

There is a [`test_runner.py`](test_runner.py) script that runs sweeps of these tests across a number of different configurations, defined
in [`test_config.yaml`](test_config.yaml). This is mostly for internal use, but usage of this script closely matches that of
the [`benchmark_runner`](../benchmark/benchmark_runner.py) script, documented [here](../benchmark/README.md).
