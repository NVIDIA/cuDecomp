# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

import cupy as cp
from mpi4py import MPI
import numpy as np

import cudecomp

# Maximum error accepted by verification; matches the C++ benchmark's double-precision tolerance.
TOLERANCE = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(description="Run a distributed 3D C2C FFT example with cuDecomp.")
    parser.add_argument("--gdims", type=int, nargs=3, metavar=("X", "Y", "Z"), default=(256, 256, 256))
    parser.add_argument("--nwarmup", type=int, default=3)
    parser.add_argument("--ntrials", type=int, default=5)
    args = parser.parse_args()
    if any(dim <= 0 for dim in args.gdims):
        parser.error("--gdims entries must be positive")
    if args.nwarmup < 0:
        parser.error("--nwarmup must be nonnegative")
    if args.ntrials <= 0:
        parser.error("--ntrials must be positive")
    args.gdims = tuple(args.gdims)
    return args


def process_timings(comm, times):
    times = sorted(times)
    time_min = comm.allreduce(times[0], op=MPI.MIN)
    time_max = comm.allreduce(times[-1], op=MPI.MAX)
    time_avg = comm.allreduce(sum(times) / len(times), op=MPI.SUM) / comm.Get_size()
    return time_min, time_max, time_avg


def main():
    args = parse_args()

    # Initialize MPI and start up cuDecomp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    cp.cuda.Device(local_rank).use()

    handle = cudecomp.Handle(comm)

    # Create cuDecomp grid
    config = cudecomp.GridDescConfig()
    config.pdims = (0, 0)  # P_rows, P_cols
    config.gdims = args.gdims  # X, Y, Z
    config.transpose_axis_contiguous = (True, True, True)

    options = cudecomp.GridDescAutotuneOptions()
    options.autotune_transpose_backend = True
    options.dtype = cudecomp.DataType.DOUBLE_COMPLEX

    grid = handle.create_grid(config, options)

    if rank == 0:
        print(f"Running on {config.pdims[0]} x {config.pdims[1]} process grid...")
        print(f"Running on {config.gdims[0]} x {config.gdims[1]} x {config.gdims[2]} (complex) spatial grid...")
        backend = config.transpose_comm_backend
        print(f"Running with {cudecomp.transpose_comm_backend_to_string(backend)}...")

    # Create initial data array, in X-pencil configuration.
    pinfo_x = grid.get_pencil_info(0)
    data = cp.empty(pinfo_x.shape, dtype=cp.complex128)

    # Create reference X-pencil input as one global Fourier mode. The forward C2C FFT should produce
    # prod(gdims) at this mode and roundoff-level values everywhere else.
    x_data_ref = cp.empty(pinfo_x.shape, dtype=cp.complex128)

    gdims = config.gdims
    mode = tuple(k % n for k, n in zip((3, 5, 7), gdims))
    fft_size = np.prod(gdims)

    coord0 = cp.arange(pinfo_x.shape[0], dtype=cp.float64) + pinfo_x.lo[0]
    coord1 = cp.arange(pinfo_x.shape[1], dtype=cp.float64) + pinfo_x.lo[1]
    coord2 = cp.arange(pinfo_x.shape[2], dtype=cp.float64) + pinfo_x.lo[2]

    wave0 = cp.exp(2j * cp.pi * mode[pinfo_x.order[0]] * coord0 / gdims[pinfo_x.order[0]])
    wave1 = cp.exp(2j * cp.pi * mode[pinfo_x.order[1]] * coord1 / gdims[pinfo_x.order[1]])
    wave2 = cp.exp(2j * cp.pi * mode[pinfo_x.order[2]] * coord2 / gdims[pinfo_x.order[2]])

    x_data_ref[...] = wave0[:, None, None]
    x_data_ref *= wave1[None, :, None]
    x_data_ref *= wave2[None, None, :]

    # Create reference transformed Z-pencil output.
    pinfo_z = grid.get_pencil_info(2)
    z_data_ref = cp.zeros(pinfo_z.shape, dtype=cp.complex128)
    mode_local_z = tuple(mode[pinfo_z.order[i]] - pinfo_z.lo[i] for i in range(3))
    if all(0 <= mode_local_z[i] < pinfo_z.shape[i] for i in range(3)):
        z_data_ref[mode_local_z] = fft_size


    start_events = [cp.cuda.Event() for _ in range(args.ntrials)]
    stop_events = [cp.cuda.Event() for _ in range(args.ntrials)]
    for n in range(args.nwarmup + args.ntrials):
        data[...] = x_data_ref

        if n >= args.nwarmup:
            start_events[n - args.nwarmup].record()

        # Run forward C2C FFT sequence
        # Note that we reuse the data reference through this sequence to ensure we
        # don't maintain multiple pencil array allocations (in different orientations)
        data = cp.fft.fft(data, norm="backward")
        data = grid.transpose_x_to_y(data)
        data = cp.fft.fft(data, norm="backward")
        data = grid.transpose_y_to_z(data)
        data = cp.fft.fft(data, norm="backward")

        if n == 0:
            # Verify the transformed Z-pencil result.
            local_error = float(cp.max(cp.abs(data - z_data_ref)).get())
            global_error = comm.allreduce(local_error, op=MPI.MAX)
            global_relative_error_fwd = global_error / fft_size

        # Run inverse C2C FFT sequence
        data = cp.fft.ifft(data, norm="forward")
        data = grid.transpose_z_to_y(data)
        data = cp.fft.ifft(data, norm="forward")
        data = grid.transpose_y_to_x(data)
        data = cp.fft.ifft(data, norm="forward")

        if n >= args.nwarmup:
            stop_events[n - args.nwarmup].record()

        # Scale round-trip output by FFT size
        data /= fft_size

        if n == 0:
            # Verify the round-trip X-pencil result.
            local_error = float(cp.max(cp.abs(data - x_data_ref)).get())
            global_relative_error_bwd = comm.allreduce(local_error, op=MPI.MAX)

    fft_times = []
    for start, stop in zip(start_events, stop_events):
        stop.synchronize()
        # Divide by two to report one FFT time from the measured forward-and-inverse round trip.
        fft_times.append(cp.cuda.get_elapsed_time(start, stop) / 2)

    fft_time_min, fft_time_max, fft_time_avg = process_timings(comm, fft_times)
    flop_count = 5.0 * fft_size * np.log2(fft_size) * 1e-9
    fft_throughputs = [flop_count / (time * 1e-3) for time in fft_times]
    fft_throughput_min, fft_throughput_max, fft_throughput_avg = process_timings(comm, fft_throughputs)

    if rank == 0:
        print(f"Results summary:")
        print(f"\tFFT size: {config.gdims[0]} x {config.gdims[1]} x {config.gdims[2]}")
        print(f"\tFFT mode: C2C")
        print(f"\tPrecision: double")
        print(f"\tProcess grid: {config.pdims[0]} x {config.pdims[1]}")
        print(f"\tComm backend: {cudecomp.transpose_comm_backend_to_string(backend)}")
        print(f"\tTime min/max/avg [ms]: {fft_time_min:.6f}/{fft_time_max:.6f}/{fft_time_avg:.6f}")
        print(
            f"\tThroughput min/max/avg [GFLOPS/s]: "
            f"{fft_throughput_min:.6f}/{fft_throughput_max:.6f}/{fft_throughput_avg:.6f}"
        )
        print(f"\tMax relative error: {global_relative_error_fwd:.6e} (fwd), {global_relative_error_bwd:.6e} (bwd)")

    # Fail like the C++ benchmark when verification exceeds its double-precision tolerance. The errors are
    # reduced across ranks, so every rank takes the same branch.
    failed = not (global_relative_error_fwd <= TOLERANCE and global_relative_error_bwd <= TOLERANCE)
    if failed and rank == 0:
        print(f"FAILURE: max relative error exceeds tolerance {TOLERANCE:.1e}")

    # Cleanup resources
    grid.destroy()
    handle.finalize()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
