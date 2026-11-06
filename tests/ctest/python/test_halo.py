# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudecomp

CPU_ONLY = cudecomp.__version__.endswith("+cpu")

try:
    from mpi4py import MPI
except ImportError as exc:
    raise RuntimeError("mpi4py is required for cudecomp_python_halo") from exc

if CPU_ONLY:
    import numpy as array_module
else:
    try:
        import cupy as array_module
    except ImportError as exc:
        raise RuntimeError("CuPy is required for cudecomp_python_halo") from exc


K_GDIMS = (9, 10, 11)
K_PDIMS = (2, 2)
K_HALO_EXTENTS = (1, 2, 1)
K_PADDING = (1, 1, 2)
K_NONPERIODIC_HALOS = (False, False, False)
K_MIXED_HALOS = (True, False, False)  # Not a palindrome, so reversed axis order is detected.
K_MIXED_MEM_ORDER = ((2, 1, 0), (2, 1, 0), (2, 1, 0))
K_DTYPES = (
    (array_module.float32, cudecomp.DataType.FLOAT),
    (array_module.float64, cudecomp.DataType.DOUBLE),
    (array_module.complex64, cudecomp.DataType.FLOAT_COMPLEX),
    (array_module.complex128, cudecomp.DataType.DOUBLE_COMPLEX),
)


class CudaArrayInterfaceObject:
    def __init__(self, interface):
        self.__cuda_array_interface__ = interface


def assert_raises(expected_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except expected_type as exc:
        return exc
    except Exception as exc:  # noqa: BLE001 - keep this dependency-free smoke test simple.
        raise AssertionError(f"expected {expected_type.__name__}, got {type(exc).__name__}") from exc

    raise AssertionError(f"expected {expected_type.__name__}")


def distributed_config(*, rank_order=None, transpose_mem_order=None):
    config = cudecomp.GridDescConfig()
    config.gdims = K_GDIMS
    config.pdims = K_PDIMS
    config.halo_comm_backend = cudecomp.HaloCommBackend.MPI
    if rank_order is not None:
        config.rank_order = rank_order
    if transpose_mem_order is not None:
        config.transpose_mem_order = transpose_mem_order
    return config


def with_grid(function, *, config=None):
    handle = cudecomp.Handle(MPI.COMM_WORLD)
    grid = None
    try:
        grid = handle.create_grid(config if config is not None else distributed_config())
        function(grid)
    finally:
        if grid is not None:
            grid.destroy()
        handle.finalize()


def is_internal(pinfo, local):
    return (
        local[0] >= pinfo.halo_extents[pinfo.order[0]]
        and local[0] < pinfo.shape[0] - pinfo.halo_extents[pinfo.order[0]] - pinfo.padding[pinfo.order[0]]
        and local[1] >= pinfo.halo_extents[pinfo.order[1]]
        and local[1] < pinfo.shape[1] - pinfo.halo_extents[pinfo.order[1]] - pinfo.padding[pinfo.order[1]]
        and local[2] >= pinfo.halo_extents[pinfo.order[2]]
        and local[2] < pinfo.shape[2] - pinfo.halo_extents[pinfo.order[2]] - pinfo.padding[pinfo.order[2]]
    )


def is_padding(pinfo, local):
    return (
        local[0] >= pinfo.shape[0] - pinfo.padding[pinfo.order[0]]
        or local[1] >= pinfo.shape[1] - pinfo.padding[pinfo.order[1]]
        or local[2] >= pinfo.shape[2] - pinfo.padding[pinfo.order[2]]
    )


def local_index_to_coordinate(pinfo, index):
    return (
        index // (pinfo.shape[1] * pinfo.shape[2]),
        (index // pinfo.shape[2]) % pinfo.shape[1],
        index % pinfo.shape[2],
    )


def global_coordinate(pinfo, local):
    global_coord = [0, 0, 0]
    global_coord[pinfo.order[0]] = local[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]]
    global_coord[pinfo.order[1]] = local[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]]
    global_coord[pinfo.order[2]] = local[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]]
    return global_coord


def global_linear_index(global_coord):
    return global_coord[0] + K_GDIMS[0] * (global_coord[1] + global_coord[2] * K_GDIMS[1])


def initialize_pencil_values(pinfo):
    values = []
    for index in range(pinfo.size):
        local = local_index_to_coordinate(pinfo, index)
        if is_internal(pinfo, local):
            values.append(float(global_linear_index(global_coordinate(pinfo, local))))
        else:
            values.append(-1.0)
    return values


def initialize_reference_values(pinfo, halo_periods):
    values = []
    for index in range(pinfo.size):
        local = local_index_to_coordinate(pinfo, index)
        global_coord = global_coordinate(pinfo, local)
        unset = is_padding(pinfo, local)

        for dim in range(3):
            if 0 <= global_coord[dim] < K_GDIMS[dim]:
                continue
            if halo_periods[dim]:
                global_coord[dim] = global_coord[dim] % K_GDIMS[dim]
            else:
                unset = True

        values.append(-1.0 if unset else float(global_linear_index(global_coord)))
    return values


def backend_empty(shape, dtype=array_module.float32):
    return array_module.empty(shape, dtype=dtype)


def new_stream():
    if CPU_ONLY:
        return 12345
    return array_module.cuda.Stream(non_blocking=True)


def pencil_array(pinfo, dtype=array_module.float32):
    return array_module.asarray(initialize_pencil_values(pinfo), dtype=dtype).reshape(pinfo.shape)


def assert_pencil_matches(array, pinfo, halo_periods):
    actual = array.ravel(order="C")
    if not CPU_ONLY:
        actual = array_module.asnumpy(actual)
    expected = initialize_reference_values(pinfo, halo_periods)

    for index, expected_value in enumerate(expected):
        actual_value = actual[index]
        assert actual_value == expected_value, (
            f"rank {MPI.COMM_WORLD.Get_rank()} mismatch at local index {index}: "
            f"expected {expected_value}, got {actual_value}"
        )


def generic_cuda_array(array, *, read_only=False):
    interface = dict(array.__cuda_array_interface__)
    interface["data"] = (array.data.ptr, read_only)
    return CudaArrayInterfaceObject(interface)


def update_method(grid, axis):
    return (grid.update_halos_x, grid.update_halos_y, grid.update_halos_z)[axis]


def run_halo_axis(
    grid,
    axis,
    *,
    halo_periods=K_MIXED_HALOS,
    padding=None,
    array_dtype=array_module.float32,
    dtype=None,
    stream=None,
    generic=False,
):
    pinfo = grid.get_pencil_info(axis, halo_extents=K_HALO_EXTENTS, padding=padding)
    data = pencil_array(pinfo, dtype=array_dtype)
    if stream is not None and not CPU_ONLY:
        array_module.cuda.get_current_stream().synchronize()
    input_object = generic_cuda_array(data) if generic else data
    method = update_method(grid, axis)

    for dim in range(3):
        returned = method(
            input_object,
            halo_extents=K_HALO_EXTENTS,
            dim=dim,
            dtype=dtype,
            halo_periods=halo_periods,
            padding=padding,
            stream=stream,
        )
        assert returned is input_object

    if stream is not None and not CPU_ONLY:
        stream.synchronize()
    assert_pencil_matches(data, pinfo, halo_periods if halo_periods is not None else K_NONPERIODIC_HALOS)


def run_halo_scenario(grid, *, padding=None, array_dtype=array_module.float32, dtype=None, use_stream=False):
    for axis in range(3):
        run_halo_axis(
            grid,
            axis,
            halo_periods=K_MIXED_HALOS,
            array_dtype=array_dtype,
            dtype=dtype,
            padding=padding,
            stream=new_stream() if use_stream else None,
        )


def test_halo_default_padding():
    def run(grid):
        for array_dtype, _ in K_DTYPES:
            run_halo_scenario(grid, array_dtype=array_dtype)

    with_grid(run)


def test_halo_custom_layout_explicit_padding_stream():
    config = distributed_config(
        rank_order=cudecomp.RankOrder.COL_MAJOR,
        transpose_mem_order=K_MIXED_MEM_ORDER,
    )

    def run(grid):
        for array_dtype, dtype in K_DTYPES:
            run_halo_scenario(
                grid,
                padding=K_PADDING,
                array_dtype=array_dtype,
                dtype=dtype,
                use_stream=True,
            )

    with_grid(run, config=config)


def test_generic_cuda_array_with_explicit_dtype():
    def run(grid):
        run_halo_axis(grid, axis=2, halo_periods=K_MIXED_HALOS, dtype=cudecomp.DataType.FLOAT, generic=True)

    with_grid(run)


def test_halo_default_periods():
    def run(grid):
        run_halo_axis(grid, axis=0, halo_periods=None)

    with_grid(run)


def test_input_validation_errors():
    def run(grid):
        x_info = grid.get_pencil_info(0, halo_extents=K_HALO_EXTENTS, padding=K_PADDING)
        x = pencil_array(x_info)
        kwargs = {"halo_extents": K_HALO_EXTENTS, "dim": 0, "padding": K_PADDING}

        assert_raises(cudecomp.InvalidUsageError, grid.update_halos_x, x, dtype=cudecomp.DataType.DOUBLE, **kwargs)
        assert_raises(TypeError, grid.update_halos_x, x, dtype="float32", **kwargs)
        assert_raises(cudecomp.InvalidUsageError, grid.update_halos_x, backend_empty((x_info.size,)), **kwargs)

        x_shape = x_info.shape
        noncontiguous = backend_empty((x_shape[0], x_shape[1], x_shape[2] * 2))[:, :, ::2]
        assert_raises(cudecomp.InvalidUsageError, grid.update_halos_x, noncontiguous, **kwargs)

        if CPU_ONLY:
            x.flags.writeable = False
            read_only_input = x
        else:
            read_only_input = generic_cuda_array(x, read_only=True)
        assert_raises(cudecomp.InvalidUsageError, grid.update_halos_x, read_only_input, **kwargs)

    with_grid(run)


def test_halo_argument_validation_errors():
    def run(grid):
        x_info = grid.get_pencil_info(0, halo_extents=K_HALO_EXTENTS, padding=K_PADDING)
        x = pencil_array(x_info)

        assert_raises(TypeError, grid.update_halos_x, x, dim=0)
        assert_raises(TypeError, grid.update_halos_x, x, halo_extents=(1, 2), dim=0)
        assert_raises(TypeError, grid.update_halos_x, x, halo_extents=K_HALO_EXTENTS, dim=0, halo_periods=(True,))
        assert_raises(TypeError, grid.update_halos_x, x, halo_extents=K_HALO_EXTENTS, dim=0, padding=(1, 2))

    with_grid(run)


def main():
    if MPI.COMM_WORLD.Get_size() != 4:
        raise RuntimeError(f"halo MPI test requires 4 ranks, got {MPI.COMM_WORLD.Get_size()}")

    if not CPU_ONLY:
        local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        array_module.cuda.Device(local_comm.Get_rank() % array_module.cuda.runtime.getDeviceCount()).use()
        local_comm.Free()

    test_halo_default_padding()
    test_halo_custom_layout_explicit_padding_stream()
    if not CPU_ONLY:
        test_generic_cuda_array_with_explicit_dtype()
    test_halo_default_periods()
    test_input_validation_errors()
    test_halo_argument_validation_errors()


if __name__ == "__main__":
    main()
