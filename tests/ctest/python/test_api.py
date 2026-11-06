# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import re
import weakref
from pathlib import Path

import numpy as np
from mpi4py import MPI

import cudecomp

K_GDIMS = (9, 10, 11)
K_PDIMS = (2, 2)
K_HALO_EXTENTS = (1, 2, 1)
K_PADDING = (1, 0, 2)

EXPECTED_X_PENCIL = (
    ((10, 9, 12), (0, 0, 0), (5, 4, 8), 1080),
    ((9, 9, 12), (6, 0, 0), (10, 4, 8), 972),
    ((10, 9, 12), (0, 5, 0), (5, 9, 8), 1080),
    ((9, 9, 12), (6, 5, 0), (10, 9, 8), 972),
)


def assert_raises(expected_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except expected_type as exc:
        return exc
    except Exception as exc:  # noqa: BLE001 - keep this dependency-free smoke test simple.
        raise AssertionError(f"expected {expected_type.__name__}, got {type(exc).__name__}") from exc

    raise AssertionError(f"expected {expected_type.__name__}")


def check_enum_values(enum_type, expected):
    for name, value in expected.items():
        actual = getattr(enum_type, name)
        assert int(actual) == value, f"{enum_type.__name__}.{name} has value {int(actual)}, expected {value}"


def header_version():
    header = Path(__file__).resolve().parents[3] / "include" / "cudecomp_version.h"
    source = header.read_text(encoding="utf-8")
    version = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define CUDECOMP_{part}\s+(\d+)$", source, re.MULTILINE)
        assert match is not None, f"missing CUDECOMP_{part} macro"
        version.append(match.group(1))
    return ".".join(version)


def test_version():
    cpu_only = cudecomp.__version__.endswith("+cpu")
    assert cudecomp.__version__ == header_version() + ("+cpu" if cpu_only else "")


def test_enum_values():
    check_enum_values(
        cudecomp.TransposeCommBackend,
        {
            "MPI_P2P": 1,
            "MPI_P2P_PL": 2,
            "MPI_A2A": 3,
            "NCCL": 4,
            "NCCL_PL": 5,
            "NVSHMEM": 6,
            "NVSHMEM_PL": 7,
            "NVSHMEM_SM": 8,
        },
    )
    check_enum_values(
        cudecomp.HaloCommBackend,
        {
            "MPI": 1,
            "MPI_BLOCKING": 2,
            "NCCL": 3,
            "NVSHMEM": 4,
            "NVSHMEM_BLOCKING": 5,
        },
    )
    check_enum_values(
        cudecomp.DataType,
        {
            "FLOAT": -1,
            "DOUBLE": -2,
            "FLOAT_COMPLEX": -3,
            "DOUBLE_COMPLEX": -4,
        },
    )
    check_enum_values(
        cudecomp.AutotuneGridMode,
        {
            "TRANSPOSE": 0,
            "HALO": 1,
        },
    )
    check_enum_values(
        cudecomp.RankOrder,
        {
            "DEFAULT": 0,
            "ROW_MAJOR": 1,
            "COL_MAJOR": 2,
        },
    )
    check_enum_values(
        cudecomp.Result,
        {
            "SUCCESS": 0,
            "INVALID_USAGE": 1,
            "NOT_SUPPORTED": 2,
            "INTERNAL_ERROR": 3,
            "CUDA_ERROR": 4,
            "CUTENSOR_ERROR": 5,
            "MPI_ERROR": 6,
            "NCCL_ERROR": 7,
            "NVSHMEM_ERROR": 8,
            "NVML_ERROR": 9,
        },
    )


def test_exception_hierarchy():
    assert issubclass(cudecomp.CuDecompError, RuntimeError)
    for error_type in (
        cudecomp.InvalidUsageError,
        cudecomp.NotSupportedError,
        cudecomp.InternalError,
        cudecomp.CudaError,
        cudecomp.CutensorError,
        cudecomp.MpiError,
        cudecomp.NcclError,
        cudecomp.NvshmemError,
        cudecomp.NvmlError,
    ):
        assert issubclass(error_type, cudecomp.CuDecompError)


def test_grid_desc_config_defaults():
    config = cudecomp.GridDescConfig()

    np.testing.assert_array_equal(config.gdims, (0, 0, 0))
    np.testing.assert_array_equal(config.gdims_dist, (0, 0, 0))
    np.testing.assert_array_equal(config.pdims, (0, 0))
    assert config.rank_order == cudecomp.RankOrder.DEFAULT
    assert config.transpose_comm_backend == cudecomp.TransposeCommBackend.MPI_P2P
    np.testing.assert_array_equal(config.transpose_axis_contiguous, (False, False, False))
    np.testing.assert_array_equal(config.transpose_mem_order, ((-1, -1, -1), (-1, -1, -1), (-1, -1, -1)))
    assert config.halo_comm_backend == cudecomp.HaloCommBackend.MPI


def test_grid_desc_config_setters():
    config = cudecomp.GridDescConfig()

    config.gdims = [9, 10, 11]
    config.gdims_dist = (8, 9, 10)
    config.pdims = [2, 2]
    config.rank_order = cudecomp.RankOrder.COL_MAJOR
    config.transpose_comm_backend = cudecomp.TransposeCommBackend.MPI_A2A
    config.transpose_axis_contiguous = (True, False, True)
    config.transpose_mem_order = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    config.halo_comm_backend = cudecomp.HaloCommBackend.MPI_BLOCKING

    np.testing.assert_array_equal(config.gdims, (9, 10, 11))
    config.gdims += 1
    np.testing.assert_array_equal(config.gdims, (10, 11, 12))
    np.testing.assert_array_equal(config.gdims_dist, (8, 9, 10))
    np.testing.assert_array_equal(config.pdims, (2, 2))
    assert config.rank_order == cudecomp.RankOrder.COL_MAJOR
    assert config.transpose_comm_backend == cudecomp.TransposeCommBackend.MPI_A2A
    np.testing.assert_array_equal(config.transpose_axis_contiguous, (True, False, True))
    np.testing.assert_array_equal(config.transpose_mem_order, ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    assert config.halo_comm_backend == cudecomp.HaloCommBackend.MPI_BLOCKING


def test_grid_desc_config_rejects_invalid_fixed_size_values():
    config = cudecomp.GridDescConfig()
    assert_raises(TypeError, setattr, config, "gdims", (1, 2))
    assert_raises(TypeError, setattr, config, "transpose_mem_order", ((0, 1, 2), (0, 1), (0, 1, 2)))


def test_grid_desc_autotune_defaults():
    options = cudecomp.GridDescAutotuneOptions()

    assert options.n_warmup_trials == 3
    assert options.n_trials == 5
    assert options.grid_mode == cudecomp.AutotuneGridMode.TRANSPOSE
    assert options.dtype == cudecomp.DataType.DOUBLE
    assert options.allow_uneven_decompositions is True
    assert options.disable_mpi_backends is False
    assert options.disable_nccl_backends is False
    assert options.disable_nvshmem_backends is False
    assert options.skip_threshold == 0.0
    assert options.autotune_transpose_backend is False
    np.testing.assert_array_equal(options.transpose_use_inplace_buffers, (False, False, False, False))
    np.testing.assert_array_equal(options.transpose_op_weights, (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_array_equal(options.transpose_input_halo_extents, ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    np.testing.assert_array_equal(options.transpose_output_halo_extents, ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    np.testing.assert_array_equal(options.transpose_input_padding, ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    np.testing.assert_array_equal(options.transpose_output_padding, ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    assert options.autotune_halo_backend is False
    np.testing.assert_array_equal(options.halo_extents, (0, 0, 0))
    np.testing.assert_array_equal(options.halo_periods, (False, False, False))
    assert options.halo_axis == 0
    np.testing.assert_array_equal(options.halo_padding, (0, 0, 0))


def test_grid_desc_autotune_setters():
    options = cudecomp.GridDescAutotuneOptions()

    options.n_warmup_trials = 0
    options.n_trials = 1
    options.grid_mode = cudecomp.AutotuneGridMode.HALO
    options.dtype = cudecomp.DataType.FLOAT
    options.allow_uneven_decompositions = False
    options.disable_mpi_backends = True
    options.disable_nccl_backends = True
    options.disable_nvshmem_backends = True
    options.skip_threshold = 2.5
    options.autotune_transpose_backend = True
    options.transpose_use_inplace_buffers = (True, False, True, False)
    options.transpose_op_weights = (1.0, 2.0, 3.0, 4.0)
    options.transpose_input_halo_extents = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    options.transpose_output_halo_extents = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 2))
    options.transpose_input_padding = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
    options.transpose_output_padding = ((12, 11, 10), (9, 8, 7), (6, 5, 4), (3, 2, 1))
    options.autotune_halo_backend = True
    options.halo_extents = (1, 2, 3)
    options.halo_periods = (False, True, False)
    options.halo_axis = 2
    options.halo_padding = (3, 2, 1)

    assert options.n_warmup_trials == 0
    assert options.n_trials == 1
    assert options.grid_mode == cudecomp.AutotuneGridMode.HALO
    assert options.dtype == cudecomp.DataType.FLOAT
    assert options.allow_uneven_decompositions is False
    assert options.disable_mpi_backends is True
    assert options.disable_nccl_backends is True
    assert options.disable_nvshmem_backends is True
    assert options.skip_threshold == 2.5
    assert options.autotune_transpose_backend is True
    np.testing.assert_array_equal(options.transpose_use_inplace_buffers, (True, False, True, False))
    np.testing.assert_array_equal(options.transpose_op_weights, (1.0, 2.0, 3.0, 4.0))
    np.testing.assert_array_equal(options.transpose_input_halo_extents, ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    np.testing.assert_array_equal(options.transpose_output_halo_extents, ((2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 2)))
    np.testing.assert_array_equal(options.transpose_input_padding, ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    np.testing.assert_array_equal(options.transpose_output_padding, ((12, 11, 10), (9, 8, 7), (6, 5, 4), (3, 2, 1)))
    assert options.autotune_halo_backend is True
    np.testing.assert_array_equal(options.halo_extents, (1, 2, 3))
    np.testing.assert_array_equal(options.halo_periods, (False, True, False))
    assert options.halo_axis == 2
    np.testing.assert_array_equal(options.halo_padding, (3, 2, 1))


def test_grid_desc_autotune_rejects_invalid_fixed_size_values():
    options = cudecomp.GridDescAutotuneOptions()
    assert_raises(TypeError, setattr, options, "transpose_op_weights", (1.0, 1.0, 1.0))
    assert_raises(TypeError, setattr, options, "transpose_input_halo_extents", ((0, 0, 0),) * 3)


def test_builtin_numeric_conversions():
    config = cudecomp.GridDescConfig()
    config.gdims = (np.int32(9), np.int64(10), 11)
    config.pdims = np.array([2, 2], dtype=np.int64)
    config.transpose_mem_order = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
    config.transpose_axis_contiguous = (np.bool_(True), 0, False)
    np.testing.assert_array_equal(config.gdims, (9, 10, 11))
    np.testing.assert_array_equal(config.pdims, (2, 2))
    np.testing.assert_array_equal(config.transpose_mem_order, [[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    np.testing.assert_array_equal(config.transpose_axis_contiguous, (True, False, False))

    options = cudecomp.GridDescAutotuneOptions()
    options.n_trials = np.int64(5)
    options.skip_threshold = np.float32(0.5)
    options.allow_uneven_decompositions = np.bool_(False)
    options.disable_mpi_backends = 1
    options.halo_periods = (np.bool_(True), 0, False)
    options.transpose_op_weights = (np.float32(0.5), np.float64(1.5), 2, 3.0)
    assert options.n_trials == 5
    assert options.skip_threshold == 0.5
    assert options.allow_uneven_decompositions is False
    assert options.disable_mpi_backends is True
    np.testing.assert_array_equal(options.halo_periods, (True, False, False))
    np.testing.assert_array_equal(options.transpose_op_weights, (0.5, 1.5, 2.0, 3.0))

    # Integer fields reject out-of-range values and floating-point dimensions.
    for value in (2**31, -(2**31) - 1):
        assert_raises(TypeError, setattr, config, "gdims", (value, 10, 11))
        assert_raises(TypeError, setattr, options, "n_trials", value)
    for value in (1.5, np.float32(1.5)):
        assert_raises(TypeError, setattr, config, "gdims", (value, 10, 11))
        assert_raises(TypeError, setattr, options, "n_trials", value)
    assert_raises(TypeError, setattr, config, "gdims", np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(config.gdims, (9, 10, 11))
    assert options.n_trials == 5


def test_fixed_size_array_views():
    fields = (
        (cudecomp.GridDescConfig, "gdims", (3,), np.int32),
        (cudecomp.GridDescConfig, "gdims_dist", (3,), np.int32),
        (cudecomp.GridDescConfig, "pdims", (2,), np.int32),
        (cudecomp.GridDescConfig, "transpose_axis_contiguous", (3,), np.bool_),
        (cudecomp.GridDescConfig, "transpose_mem_order", (3, 3), np.int32),
        (cudecomp.GridDescAutotuneOptions, "transpose_use_inplace_buffers", (4,), np.bool_),
        (cudecomp.GridDescAutotuneOptions, "transpose_op_weights", (4,), np.float64),
        (cudecomp.GridDescAutotuneOptions, "transpose_input_halo_extents", (4, 3), np.int32),
        (cudecomp.GridDescAutotuneOptions, "transpose_output_halo_extents", (4, 3), np.int32),
        (cudecomp.GridDescAutotuneOptions, "transpose_input_padding", (4, 3), np.int32),
        (cudecomp.GridDescAutotuneOptions, "transpose_output_padding", (4, 3), np.int32),
        (cudecomp.GridDescAutotuneOptions, "halo_extents", (3,), np.int32),
        (cudecomp.GridDescAutotuneOptions, "halo_periods", (3,), np.bool_),
        (cudecomp.GridDescAutotuneOptions, "halo_padding", (3,), np.int32),
    )
    for owner_type, name, shape, dtype in fields:
        owner = owner_type()
        view = getattr(owner, name)
        assert isinstance(view, np.ndarray)
        assert view.shape == shape
        assert view.dtype == np.dtype(dtype)
        assert view.flags.writeable

        # Independent getters share storage, and whole-field setters update existing views.
        view[...] = 1
        np.testing.assert_array_equal(getattr(owner, name), np.ones(shape, dtype=dtype))
        setattr(owner, name, np.zeros(shape, dtype=dtype))
        np.testing.assert_array_equal(view, np.zeros(shape, dtype=dtype))


def test_convenience_functions():
    assert cudecomp.get_data_type_size(cudecomp.DataType.FLOAT) == 4
    assert cudecomp.get_data_type_size(cudecomp.DataType.DOUBLE) == 8
    assert cudecomp.get_data_type_size(cudecomp.DataType.FLOAT_COMPLEX) == 8
    assert cudecomp.get_data_type_size(cudecomp.DataType.DOUBLE_COMPLEX) == 16

    assert cudecomp.transpose_comm_backend_to_string(cudecomp.TransposeCommBackend.MPI_P2P) == "MPI_P2P"
    assert cudecomp.halo_comm_backend_to_string(cudecomp.HaloCommBackend.MPI) == "MPI"


def test_array_view_lifetime():
    for owner_type, name in (
        (cudecomp.GridDescConfig, "gdims"),
        (cudecomp.GridDescAutotuneOptions, "transpose_input_halo_extents"),
    ):
        owner = owner_type()
        view = getattr(owner, name)
        owner_ref = weakref.ref(owner)
        saved = view[1:]
        del owner, view
        gc.collect()
        assert owner_ref() is not None
        saved[...] = 1
        np.testing.assert_array_equal(getattr(owner_ref(), name)[1:], saved)
        del saved
        gc.collect()
        assert owner_ref() is None


def distributed_config():
    config = cudecomp.GridDescConfig()
    config.gdims[:] = K_GDIMS
    config.pdims[:] = K_PDIMS
    return config


def assert_config_matches(actual, expected):
    np.testing.assert_array_equal(actual.gdims, expected.gdims)
    np.testing.assert_array_equal(actual.gdims_dist, expected.gdims_dist)
    np.testing.assert_array_equal(actual.pdims, expected.pdims)
    assert actual.rank_order == expected.rank_order
    assert actual.transpose_comm_backend == expected.transpose_comm_backend
    np.testing.assert_array_equal(actual.transpose_axis_contiguous, expected.transpose_axis_contiguous)
    np.testing.assert_array_equal(actual.transpose_mem_order, expected.transpose_mem_order)
    assert actual.halo_comm_backend == expected.halo_comm_backend


def test_grid_creation_metadata_and_destroy():
    rank = MPI.COMM_WORLD.Get_rank()
    handle = cudecomp.Handle(MPI.COMM_WORLD)
    grid = None
    options_grid = None
    try:
        config = distributed_config()
        config.rank_order = cudecomp.RankOrder.ROW_MAJOR
        grid = handle.create_grid(config)
        assert isinstance(grid, cudecomp.GridDesc)

        resolved = grid.get_config()
        assert_config_matches(resolved, config)

        pinfo = grid.get_pencil_info(axis=0, halo_extents=K_HALO_EXTENTS, padding=K_PADDING)
        assert isinstance(pinfo, cudecomp.PencilInfo)
        shape, lo, hi, size = EXPECTED_X_PENCIL[rank]
        assert pinfo.shape == shape
        assert pinfo.lo == lo
        assert pinfo.hi == hi
        assert pinfo.order == (2, 1, 0)
        assert pinfo.halo_extents == K_HALO_EXTENTS
        assert pinfo.padding == K_PADDING
        assert pinfo.size == size

        assert grid.get_shifted_rank(axis=0, dim=1, displacement=1, periodic=False) == (2, 3, -1, -1)[rank]

        options = cudecomp.GridDescAutotuneOptions()
        options_grid = handle.create_grid(distributed_config(), autotune_options=options)
        assert isinstance(options_grid.get_config(), cudecomp.GridDescConfig)

        options_grid.destroy()
        options_grid = None

        grid.destroy()
        grid.destroy()
        assert_raises(cudecomp.InvalidUsageError, grid.get_config)
        grid = None
    finally:
        if options_grid is not None:
            options_grid.destroy()
        if grid is not None:
            grid.destroy()
        handle.finalize()

    handle.finalize()
    assert_raises(cudecomp.InvalidUsageError, handle.create_grid, distributed_config())


def test_memory_order_view_reaches_native_grid():
    handle = cudecomp.Handle(MPI.COMM_WORLD)
    grid = None
    try:
        config = distributed_config()
        config.transpose_mem_order[:] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        config.transpose_mem_order[0] = [2, 1, 0]
        grid = handle.create_grid(config)
        np.testing.assert_array_equal(grid.get_config().transpose_mem_order, config.transpose_mem_order)
        options = cudecomp.GridDescAutotuneOptions()
        options.halo_extents[:] = [1, 2, 1]
        for axis in range(3):
            pinfo = grid.get_pencil_info(axis, halo_extents=options.halo_extents)
            assert pinfo.halo_extents == (1, 2, 1)
            assert grid.get_pencil_info(axis).order == tuple(config.transpose_mem_order[axis, ::-1])
    finally:
        if grid is not None:
            grid.destroy()
        handle.finalize()


def main():
    if MPI.COMM_WORLD.Get_size() != 4:
        raise RuntimeError(f"Python API test requires 4 ranks, got {MPI.COMM_WORLD.Get_size()}")

    if not cudecomp.__version__.endswith("+cpu"):
        import cupy as cp

        local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        cp.cuda.Device(local_comm.Get_rank() % cp.cuda.runtime.getDeviceCount()).use()
        local_comm.Free()

    test_version()
    test_enum_values()
    test_exception_hierarchy()
    test_grid_desc_config_defaults()
    test_grid_desc_config_setters()
    test_grid_desc_config_rejects_invalid_fixed_size_values()
    test_grid_desc_autotune_defaults()
    test_grid_desc_autotune_setters()
    test_grid_desc_autotune_rejects_invalid_fixed_size_values()
    test_builtin_numeric_conversions()
    test_fixed_size_array_views()
    test_convenience_functions()
    test_array_view_lifetime()
    test_grid_creation_metadata_and_destroy()
    test_memory_order_view_reaches_native_grid()


if __name__ == "__main__":
    main()
