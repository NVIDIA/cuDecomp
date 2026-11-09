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
    parser = argparse.ArgumentParser(description="Run a distributed 3D R2C FFT example with cuDecomp.")
    parser.add_argument("--gdims", type=int, nargs=3, metavar=("X", "Y", "Z"), default=(256, 256, 256))
    parser.add_argument("--nwarmup", type=int, default=3)
    parser.add_argument("--ntrials", type=int, default=5)
    args = parser.parse_args()
    if any(dim <= 0 for dim in args.gdims):
        parser.error("--gdims entries must be positive")
    if args.gdims[0] <= 6 or args.gdims[1] <= 5 or args.gdims[2] <= 7:
        parser.error("--gdims must be larger than the verification mode (3, 5, 7)")
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

    # Create cuDecomp grid for complex spectral pencils.
    gdims_r = args.gdims
    gdims_c = (gdims_r[0] // 2 + 1, gdims_r[1], gdims_r[2])

    config_c = cudecomp.GridDescConfig()
    config_c.pdims = (0, 0)  # P_rows, P_cols
    config_c.gdims = gdims_c  # X, Y, Z
    config_c.transpose_axis_contiguous = (True, True, True)

    options = cudecomp.GridDescAutotuneOptions()
    options.autotune_transpose_backend = True
    options.dtype = cudecomp.DataType.DOUBLE_COMPLEX

    grid_c = handle.create_grid(config_c, options)

    # Create matching real X-pencil grid without in-place R2C padding.
    config_r = cudecomp.GridDescConfig()
    config_r.pdims = config_c.pdims
    config_r.gdims = gdims_r
    config_r.rank_order = config_c.rank_order
    config_r.transpose_comm_backend = config_c.transpose_comm_backend
    config_r.transpose_axis_contiguous = config_c.transpose_axis_contiguous
    grid_r = handle.create_grid(config_r)

    if rank == 0:
        print(f"Running on {config_c.pdims[0]} x {config_c.pdims[1]} process grid...")
        print(f"Running on {gdims_r[0]} x {gdims_r[1]} x {gdims_r[2]} (real) spatial grid...")
        print(f"Running on {gdims_c[0]} x {gdims_c[1]} x {gdims_c[2]} (complex) spectral grid...")
        backend = config_c.transpose_comm_backend
        print(f"Running with {cudecomp.transpose_comm_backend_to_string(backend)}...")

    # Create initial real X-pencil data.
    pinfo_x_r = grid_r.get_pencil_info(0)
    data = cp.empty(pinfo_x_r.shape, dtype=cp.float64)

    # Create reference X-pencil input as one real global Fourier mode. The forward R2C FFT should produce
    # prod(gdims) / 2 at this mode and roundoff-level values everywhere else in the stored half spectrum.
    x_data_ref = cp.empty(pinfo_x_r.shape, dtype=cp.float64)

    mode = (3, 5, 7)
    fft_size = np.prod(gdims_r)

    coord0 = cp.arange(pinfo_x_r.shape[0], dtype=cp.float64) + pinfo_x_r.lo[0]
    coord1 = cp.arange(pinfo_x_r.shape[1], dtype=cp.float64) + pinfo_x_r.lo[1]
    coord2 = cp.arange(pinfo_x_r.shape[2], dtype=cp.float64) + pinfo_x_r.lo[2]

    phase0 = 2 * cp.pi * mode[pinfo_x_r.order[0]] * coord0 / gdims_r[pinfo_x_r.order[0]]
    phase1 = 2 * cp.pi * mode[pinfo_x_r.order[1]] * coord1 / gdims_r[pinfo_x_r.order[1]]
    phase2 = 2 * cp.pi * mode[pinfo_x_r.order[2]] * coord2 / gdims_r[pinfo_x_r.order[2]]

    x_data_ref[...] = cp.cos(phase0[:, None, None] + phase1[None, :, None] + phase2[None, None, :])

    # Create reference transformed Z-pencil output.
    pinfo_z_c = grid_c.get_pencil_info(2)
    z_data_ref = cp.zeros(pinfo_z_c.shape, dtype=cp.complex128)
    mode_local_z = tuple(mode[pinfo_z_c.order[i]] - pinfo_z_c.lo[i] for i in range(3))
    if all(0 <= mode_local_z[i] < pinfo_z_c.shape[i] for i in range(3)):
        z_data_ref[mode_local_z] = fft_size / 2

    start_events = [cp.cuda.Event() for _ in range(args.ntrials)]
    stop_events = [cp.cuda.Event() for _ in range(args.ntrials)]
    for n in range(args.nwarmup + args.ntrials):
        data[...] = x_data_ref

        if n >= args.nwarmup:
            start_events[n - args.nwarmup].record()

        # Run forward R2C FFT sequence.
        data = cp.fft.rfft(data, axis=-1, norm="backward")
        data = grid_c.transpose_x_to_y(data)
        data = cp.fft.fft(data, axis=-1, norm="backward")
        data = grid_c.transpose_y_to_z(data)
        data = cp.fft.fft(data, axis=-1, norm="backward")

        if n == 0:
            # Verify the transformed Z-pencil result.
            local_error = float(cp.max(cp.abs(data - z_data_ref)).get())
            global_error = comm.allreduce(local_error, op=MPI.MAX)
            global_relative_error_fwd = global_error / fft_size

        # Run inverse C2R FFT sequence.
        data = cp.fft.ifft(data, axis=-1, norm="forward")
        data = grid_c.transpose_z_to_y(data)
        data = cp.fft.ifft(data, axis=-1, norm="forward")
        data = grid_c.transpose_y_to_x(data)
        data = cp.fft.irfft(data, n=gdims_r[0], axis=-1, norm="forward")

        if n >= args.nwarmup:
            stop_events[n - args.nwarmup].record()

        # Scale round-trip output by FFT size.
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
        print(f"\tFFT size: {gdims_r[0]} x {gdims_r[1]} x {gdims_r[2]}")
        print(f"\tFFT mode: R2C")
        print(f"\tPrecision: double")
        print(f"\tProcess grid: {config_c.pdims[0]} x {config_c.pdims[1]}")
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

    # Cleanup resources.
    grid_r.destroy()
    grid_c.destroy()
    handle.finalize()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
