# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudecomp

CPU_ONLY = cudecomp.__version__.endswith("+cpu")

try:
    from mpi4py import MPI
except ImportError as exc:
    raise RuntimeError("mpi4py is required for cudecomp_python_transpose") from exc

if CPU_ONLY:
    import numpy as array_module
else:
    try:
        import cupy as array_module
    except ImportError as exc:
        raise RuntimeError("CuPy is required for cudecomp_python_transpose") from exc


K_GDIMS = (9, 10, 11)
K_PDIMS = (2, 2)
K_INPUT_HALO_EXTENTS = (1, 2, 1)
K_OUTPUT_HALO_EXTENTS = (2, 1, 1)
K_Z_HALO_EXTENTS = (1, 1, 2)
K_INPUT_PADDING = (1, 1, 2)
K_OUTPUT_PADDING = (2, 1, 1)
K_Z_PADDING = (1, 0, 1)
K_MIXED_MEM_ORDER = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
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


def distributed_config(*, rank_order=None, transpose_mem_order=None, transpose_axis_contiguous=None):
    config = cudecomp.GridDescConfig()
    config.gdims = K_GDIMS
    config.pdims = K_PDIMS
    config.transpose_comm_backend = cudecomp.TransposeCommBackend.MPI_P2P
    if rank_order is not None:
        config.rank_order = rank_order
    if transpose_mem_order is not None:
        config.transpose_mem_order = transpose_mem_order
    if transpose_axis_contiguous is not None:
        config.transpose_axis_contiguous = transpose_axis_contiguous
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


def local_index_to_coordinate(pinfo, index):
    return (
        index // (pinfo.shape[1] * pinfo.shape[2]),
        (index // pinfo.shape[2]) % pinfo.shape[1],
        index % pinfo.shape[2],
    )


def initialize_pencil_values(pinfo):
    values = []
    for index in range(pinfo.size):
        local = local_index_to_coordinate(pinfo, index)
        if not is_internal(pinfo, local):
            values.append(-1.0)
            continue

        global_coord = [0, 0, 0]
        global_coord[pinfo.order[0]] = local[0] + pinfo.lo[0] - pinfo.halo_extents[pinfo.order[0]]
        global_coord[pinfo.order[1]] = local[1] + pinfo.lo[1] - pinfo.halo_extents[pinfo.order[1]]
        global_coord[pinfo.order[2]] = local[2] + pinfo.lo[2] - pinfo.halo_extents[pinfo.order[2]]
        values.append(float(global_coord[0] + K_GDIMS[0] * (global_coord[1] + global_coord[2] * K_GDIMS[1])))
    return values


def backend_empty(shape, dtype=array_module.float32):
    return array_module.empty(shape, dtype=dtype)


def new_stream():
    if CPU_ONLY:
        return 12345
    return array_module.cuda.Stream(non_blocking=True)


def pencil_array(pinfo, dtype=array_module.float32):
    return array_module.asarray(initialize_pencil_values(pinfo), dtype=dtype).reshape(pinfo.shape)


def assert_pencil_matches(array, pinfo):
    actual = array.ravel(order="C")
    if not CPU_ONLY:
        actual = array_module.asnumpy(actual)
    expected = initialize_pencil_values(pinfo)

    for index, expected_value in enumerate(expected):
        local = local_index_to_coordinate(pinfo, index)
        if not is_internal(pinfo, local):
            continue
        actual_value = actual[index]
        assert actual_value == expected_value, (
            f"rank {MPI.COMM_WORLD.Get_rank()} mismatch at local index {index}: "
            f"expected {expected_value}, got {actual_value}"
        )


def generic_cuda_array(array, *, typestr=None, read_only=False):
    interface = dict(array.__cuda_array_interface__)
    interface["data"] = (array.data.ptr, read_only)
    if typestr is not None:
        interface["typestr"] = typestr
    return CudaArrayInterfaceObject(interface)


def transpose_infos(grid, *, use_optional_args):
    if not use_optional_args:
        return (
            grid.get_pencil_info(0),
            grid.get_pencil_info(1),
            grid.get_pencil_info(2),
        )
    return (
        grid.get_pencil_info(0, halo_extents=K_INPUT_HALO_EXTENTS, padding=K_INPUT_PADDING),
        grid.get_pencil_info(1, halo_extents=K_OUTPUT_HALO_EXTENTS, padding=K_OUTPUT_PADDING),
        grid.get_pencil_info(2, halo_extents=K_Z_HALO_EXTENTS, padding=K_Z_PADDING),
    )


def transpose_kwargs(input_info, output_info, *, dtype=None, stream=None, use_optional_args=False):
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if stream is not None:
        kwargs["stream"] = stream
    if use_optional_args:
        kwargs["input_halo_extents"] = input_info.halo_extents
        kwargs["output_halo_extents"] = output_info.halo_extents
        kwargs["input_padding"] = input_info.padding
        kwargs["output_padding"] = output_info.padding
    return kwargs


def run_transpose_sequence(
    grid,
    *,
    use_optional_args=False,
    explicit_out=False,
    array_dtype=array_module.float32,
    dtype=None,
    stream=None,
):
    x_info, y_info, z_info = transpose_infos(grid, use_optional_args=use_optional_args)
    current = pencil_array(x_info, dtype=array_dtype)
    if stream is not None and not CPU_ONLY:
        array_module.cuda.get_current_stream().synchronize()
    sequence = (
        (grid.transpose_x_to_y, x_info, y_info),
        (grid.transpose_y_to_z, y_info, z_info),
        (grid.transpose_z_to_y, z_info, y_info),
        (grid.transpose_y_to_x, y_info, x_info),
    )

    for method, input_info, output_info in sequence:
        kwargs = transpose_kwargs(
            input_info,
            output_info,
            dtype=dtype,
            stream=stream,
            use_optional_args=use_optional_args,
        )
        expected_output = None
        if explicit_out:
            expected_output = backend_empty(output_info.shape, dtype=array_dtype)
            kwargs["out"] = expected_output

        current = method(current, **kwargs)
        if stream is not None and not CPU_ONLY:
            stream.synchronize()
        if expected_output is not None:
            assert current is expected_output
        else:
            assert isinstance(current, array_module.ndarray)
            assert current.shape == output_info.shape
            assert current.flags.c_contiguous
        assert current.dtype == array_module.dtype(array_dtype)
        assert_pencil_matches(current, output_info)


def test_transpose_default_layout_minimal_args():
    def run(grid):
        for array_dtype, _ in K_DTYPES:
            run_transpose_sequence(grid, array_dtype=array_dtype)

    with_grid(run)


def test_transpose_custom_layout_explicit_args():
    config = distributed_config(
        rank_order=cudecomp.RankOrder.COL_MAJOR,
        transpose_mem_order=K_MIXED_MEM_ORDER,
    )

    def run(grid):
        for array_dtype, dtype in K_DTYPES:
            run_transpose_sequence(
                grid,
                use_optional_args=True,
                explicit_out=True,
                array_dtype=array_dtype,
                dtype=dtype,
                stream=new_stream(),
            )

    with_grid(run, config=config)


def test_transpose_in_place_views():
    def run(grid):
        x_info, y_info, z_info = transpose_infos(grid, use_optional_args=True)
        storage = backend_empty((max(x_info.size, y_info.size, z_info.size),))
        x, y, z = (storage[:info.size].reshape(info.shape) for info in (x_info, y_info, z_info))
        x[...] = pencil_array(x_info)

        for method, input_array, output_array, input_info, output_info in (
            (grid.transpose_x_to_y, x, y, x_info, y_info),
            (grid.transpose_y_to_z, y, z, y_info, z_info),
            (grid.transpose_z_to_y, z, y, z_info, y_info),
            (grid.transpose_y_to_x, y, x, y_info, x_info),
        ):
            returned = method(
                input_array,
                out=output_array,
                **transpose_kwargs(input_info, output_info, use_optional_args=True),
            )
            assert returned is output_array
            assert_pencil_matches(output_array, output_info)

    with_grid(run)


def test_transpose_axis_contiguous_layout():
    config = distributed_config(transpose_axis_contiguous=(True, True, True))

    def run(grid):
        x_info, y_info, _ = transpose_infos(grid, use_optional_args=False)
        x = pencil_array(x_info)
        y = grid.transpose_x_to_y(x)
        assert y.shape == y_info.shape
        assert_pencil_matches(y, y_info)

    with_grid(run, config=config)


def test_transpose_generic_cuda_array_explicit_dtype():
    class ArrayProvider:
        def __init__(self, storage):
            self.storage = storage

        @property
        def __cuda_array_interface__(self):
            return self.storage.__cuda_array_interface__

    def run(grid):
        x_info, y_info, _ = transpose_infos(grid, use_optional_args=True)
        x = pencil_array(x_info)
        out_storage = backend_empty(y_info.shape)
        generic_input = ArrayProvider(x)
        generic_out = ArrayProvider(out_storage)

        returned = grid.transpose_x_to_y(
            generic_input,
            out=generic_out,
            dtype=cudecomp.DataType.FLOAT,
            input_halo_extents=x_info.halo_extents,
            output_halo_extents=y_info.halo_extents,
            input_padding=x_info.padding,
            output_padding=y_info.padding,
        )
        assert returned is generic_out
        assert_pencil_matches(out_storage, y_info)

    with_grid(run)


def test_transpose_stream_interop():
    import numpy as np

    def run(grid):
        zero_extents = np.zeros(3, dtype=np.int64)
        x_info = grid.get_pencil_info(np.int32(0), halo_extents=zero_extents, padding=zero_extents)
        y_info = grid.get_pencil_info(np.int64(1))
        for axis in (0.5, np.float32(0.5)):
            assert_raises(TypeError, grid.get_pencil_info, axis)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)
        stream = None if CPU_ONLY else array_module.cuda.Stream(non_blocking=True)

        class StreamObject:
            def __init__(self, ptr):
                self.ptr = ptr

        class CudaStreamProtocol:
            def __init__(self, value):
                self.value = value

            def __cuda_stream__(self):
                return self.value

        pointer = 0 if CPU_ONLY else stream.ptr
        for stream_arg in (pointer, StreamObject(pointer), CudaStreamProtocol((0, pointer))):
            # Reset out so each stream form must produce the result itself; interior values are non-negative.
            out.fill(-1)
            if stream is not None:
                array_module.cuda.get_current_stream().synchronize()
            returned = grid.transpose_x_to_y(
                x, out=out, stream=stream_arg, input_halo_extents=zero_extents
            )
            assert returned is out
            if stream is not None:
                stream.synchronize()
            assert_pencil_matches(out, y_info)

        if not CPU_ONLY:
            previous_stream = array_module.cuda.get_current_stream()
            caller_stream = array_module.cuda.Stream(non_blocking=True)
            caller_stream.use()
            try:
                allocated_out = grid.transpose_x_to_y(x, stream=stream)
                assert array_module.cuda.get_current_stream().ptr == caller_stream.ptr
                stream.synchronize()
                assert_pencil_matches(allocated_out, y_info)
            finally:
                stream.synchronize()
                previous_stream.use()

        if not CPU_ONLY:
            # stream=None follows CuPy's current stream for CuPy inputs. out=None allocates from CuPy's default pool
            # on the operation stream, so the pool's per-stream arena identifies the stream that was used.
            pool = array_module.get_default_memory_pool()
            current_stream = array_module.cuda.Stream(non_blocking=True)
            with current_stream:
                allocated_out = grid.transpose_x_to_y(x)
            current_stream.synchronize()
            assert_pencil_matches(allocated_out, y_info)
            del allocated_out
            total_bytes = pool.total_bytes()
            pool.free_all_blocks(stream=current_stream)
            assert pool.total_bytes() < total_bytes, "stream=None did not use CuPy's current stream"

        for stream_arg in (
            True,
            object(),
            StreamObject(True),
            CudaStreamProtocol((1, pointer)),
            CudaStreamProtocol(pointer),
            CudaStreamProtocol((0, True)),
        ):
            assert_raises(TypeError, grid.transpose_x_to_y, x, out=out, stream=stream_arg)
        for stream_arg in (-1, 1 << 64, StreamObject(-1), CudaStreamProtocol((0, -1))):
            assert_raises(ValueError, grid.transpose_x_to_y, x, out=out, stream=stream_arg)

    with_grid(run)


def test_cuda_array_interface_validation():
    def run(grid):
        x_info = grid.get_pencil_info(0)
        y_info = grid.get_pencil_info(1)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)

        interface = dict(x.__cuda_array_interface__)
        del interface["data"]
        assert_raises(KeyError, grid.transpose_x_to_y, CudaArrayInterfaceObject(interface), out=out)

        # Malformed interface metadata is a ValueError; well-formed arrays cuDecomp cannot use are invalid usage.
        for expected_type, field, value in (
            (ValueError, "version", 4),
            (ValueError, "strides", (4,)),
            (ValueError, "typestr", "<f"),
            (ValueError, "typestr", 4),
            (ValueError, "version", "3"),
            (ValueError, "data", (0,)),
            (ValueError, "data", (-1, False)),
            (ValueError, "shape", ("a", 1, 1)),
            (ValueError, "strides", "abc"),
            (cudecomp.InvalidUsageError, "mask", object()),
            (cudecomp.InvalidUsageError, "typestr", ">f4"),
        ):
            interface = dict(x.__cuda_array_interface__)
            interface[field] = value
            assert_raises(expected_type, grid.transpose_x_to_y, CudaArrayInterfaceObject(interface), out=out)

        interface = dict(x.__cuda_array_interface__)
        interface["strides"] = x.strides
        assert grid.transpose_x_to_y(CudaArrayInterfaceObject(interface), out=out) is out
        assert_pencil_matches(out, y_info)

        interface = dict(x.__cuda_array_interface__)
        interface.pop("strides", None)
        interface.pop("mask", None)
        assert grid.transpose_x_to_y(CudaArrayInterfaceObject(interface), out=out) is out
        assert_pencil_matches(out, y_info)

        # Opaque storage is usable when its item size matches an explicit dtype.
        returned = grid.transpose_x_to_y(generic_cuda_array(x, typestr="|V4"), out=out, dtype=cudecomp.DataType.FLOAT)
        assert returned is out
        assert_pencil_matches(out, y_info)

    with_grid(run)


def test_cuda_array_singleton_strides():
    config = distributed_config()
    config.gdims = (2, 2, 2)

    def run(grid):
        x_info = grid.get_pencil_info(0)
        y_info = grid.get_pencil_info(1)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)
        assert 1 in x.shape
        interface = dict(x.__cuda_array_interface__)
        interface["strides"] = tuple(
            2 * stride if extent == 1 else stride
            for extent, stride in zip(x.shape, x.strides)
        )
        assert grid.transpose_x_to_y(CudaArrayInterfaceObject(interface), out=out) is out
        assert_pencil_matches(out, y_info)

    with_grid(run, config=config)


def test_input_validation_errors():
    def run(grid):
        x_info = grid.get_pencil_info(0)
        y_info = grid.get_pencil_info(1)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)

        if CPU_ONLY:
            mismatched_array = CudaArrayInterfaceObject({})
            expected_array = "NumPy host array"
        else:
            import numpy
            mismatched_array = numpy.empty(x_info.shape, dtype=numpy.float32)
            expected_array = "__cuda_array_interface__"
        for argument in ("input", "out"):
            error = assert_raises(
                TypeError, grid.transpose_x_to_y,
                mismatched_array if argument == "input" else x,
                out=mismatched_array if argument == "out" else out,
            )
            assert expected_array in str(error)
            assert f"'{argument}'" in str(error)

        if CPU_ONLY:
            import numpy
            nonnative_dtype = numpy.dtype(">f4" if numpy.little_endian else "<f4")
            assert_raises(
                cudecomp.InvalidUsageError, grid.transpose_x_to_y, backend_empty(x_info.shape, nonnative_dtype), out=out
            )

        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=out, dtype=cudecomp.DataType.DOUBLE)
        assert_raises(TypeError, grid.transpose_x_to_y, backend_empty(x_info.shape, array_module.int32), out=out)
        for dtype in ("float32", array_module.float32, array_module.dtype("float32"), True, -1):
            assert_raises(TypeError, grid.transpose_x_to_y, x, out=out, dtype=dtype)
        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, backend_empty((x_info.size,)), out=out)

        x_shape = x_info.shape
        noncontiguous_input = backend_empty((x_shape[0], x_shape[1], x_shape[2] * 2))[:, :, ::2]
        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, noncontiguous_input, out=out)

        if not CPU_ONLY:
            short_input = generic_cuda_array(x, typestr="|V1")
            assert_raises(
                cudecomp.InvalidUsageError, grid.transpose_x_to_y, short_input, out=out, dtype=cudecomp.DataType.FLOAT
            )
            typeless_input = generic_cuda_array(x, typestr="|V4")
            assert_raises(TypeError, grid.transpose_x_to_y, typeless_input, out=out)

    with_grid(run)


def test_output_validation_errors():
    def run(grid):
        x_info = grid.get_pencil_info(0)
        y_info = grid.get_pencil_info(1)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)

        assert_raises(
            cudecomp.InvalidUsageError, grid.transpose_x_to_y, x,
            out=backend_empty(y_info.shape, array_module.float64),
        )
        assert_raises(
            cudecomp.InvalidUsageError, grid.transpose_x_to_y, x,
            out=backend_empty(y_info.shape, array_module.int32),
        )
        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=backend_empty((y_info.size,)))

        y_shape = y_info.shape
        noncontiguous_out = backend_empty((y_shape[0], y_shape[1], y_shape[2] * 2))[:, :, ::2]
        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=noncontiguous_out)

        if CPU_ONLY:
            out.flags.writeable = False
            assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=out)
        else:
            read_only_out = generic_cuda_array(out, read_only=True)
            assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=read_only_out)
            short_out = generic_cuda_array(out, typestr="|V1")
            assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=short_out)
            generic_input = generic_cuda_array(x)
            assert_raises(TypeError, grid.transpose_x_to_y, generic_input)

    with_grid(run)


def test_destroyed_grid_validation():
    def run(grid):
        x = pencil_array(grid.get_pencil_info(0))
        out = backend_empty(grid.get_pencil_info(1).shape)
        grid.destroy()
        assert_raises(cudecomp.InvalidUsageError, grid.transpose_x_to_y, x, out=out)

    with_grid(run)


def test_transpose_argument_validation_errors():
    def run(grid):
        x_info = grid.get_pencil_info(0)
        y_info = grid.get_pencil_info(1)
        x = pencil_array(x_info)
        out = backend_empty(y_info.shape)

        assert_raises(
            TypeError,
            grid.transpose_x_to_y,
            x,
            out=out,
            dtype=cudecomp.DataType.FLOAT,
            input_halo_extents=(1, 2),
        )
        assert_raises(
            TypeError,
            grid.transpose_x_to_y,
            x,
            out=out,
            dtype=cudecomp.DataType.FLOAT,
            output_halo_extents=(1, 2),
        )
        assert_raises(
            TypeError,
            grid.transpose_x_to_y,
            x,
            out=out,
            dtype=cudecomp.DataType.FLOAT,
            input_padding=(1, 2),
        )
        assert_raises(
            TypeError,
            grid.transpose_x_to_y,
            x,
            out=out,
            dtype=cudecomp.DataType.FLOAT,
            output_padding=(1, 2),
        )

    with_grid(run)


def main():
    if MPI.COMM_WORLD.Get_size() != 4:
        raise RuntimeError(f"transpose MPI test requires 4 ranks, got {MPI.COMM_WORLD.Get_size()}")

    if not CPU_ONLY:
        local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        array_module.cuda.Device(local_comm.Get_rank() % array_module.cuda.runtime.getDeviceCount()).use()
        local_comm.Free()

    test_transpose_default_layout_minimal_args()
    test_transpose_custom_layout_explicit_args()
    test_transpose_in_place_views()
    test_transpose_axis_contiguous_layout()
    test_transpose_stream_interop()
    if not CPU_ONLY:
        test_transpose_generic_cuda_array_explicit_dtype()
        test_cuda_array_interface_validation()
        test_cuda_array_singleton_strides()
    test_input_validation_errors()
    test_output_validation_errors()
    test_transpose_argument_validation_errors()
    test_destroyed_grid_validation()


if __name__ == "__main__":
    main()
